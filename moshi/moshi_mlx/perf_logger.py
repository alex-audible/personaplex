# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Performance logging framework for PersonaPlex inference benchmarking.

Provides precise per-component, per-frame timing with GPU memory tracking
and summary statistics. Designed to instrument the real-time speech-to-speech
inference loop to establish baselines for MLX porting.

Usage:
    perf = PerfLogger(enabled=True, backend="mps", device_name="Apple M3 96GB")

    for frame in inference_loop():
        perf.frame_start()
        encode(...)
        perf.mark("mimi_encode")
        tokens = lm.step(...)
        perf.mark("lm_forward")
        decode(tokens)
        perf.mark("mimi_decode")
        perf.frame_end()

    perf.export_json("perf_output.json")
    summary = perf.summary()
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np


@dataclass
class FrameRecord:
    """Timing and memory data for a single inference frame."""
    frame_idx: int
    timestamp_unix: float
    timings_ms: Dict[str, float] = field(default_factory=dict)
    missed: bool = False
    memory_mb: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame record to a plain dict for JSON export."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp_unix": self.timestamp_unix,
            "timings_ms": dict(self.timings_ms),
            "missed": self.missed,
            "memory_mb": dict(self.memory_mb),
        }


class PerfLogger:
    """Per-frame performance logger with GPU sync and memory tracking.

    When ``enabled=False``, every public method is a no-op (zero-cost guard).

    Args:
        enabled: Whether logging is active. When False, all methods return
            immediately with no side effects.
        backend: Compute backend - ``"mps"`` or ``"cuda"``. Controls which
            synchronization primitive is called before timing marks.
        device_name: Human-readable device string for summary metadata
            (e.g. ``"Apple M3 96GB"``).
        frame_budget_ms: The real-time budget per frame in milliseconds.
            Frames exceeding this are flagged as ``missed``. Default 80.0 ms
            (matching the Mimi codec frame rate of 12.5 Hz).
    """

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "mps",
        device_name: Optional[str] = None,
        frame_budget_ms: float = 80.0,
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.device_name = device_name or _detect_device_name()
        self.frame_budget_ms = frame_budget_ms

        # Internal state
        self._frames: List[FrameRecord] = []
        self._frame_idx: int = 0
        self._frame_start_time: float = 0.0
        self._last_mark_time: float = 0.0
        self._current_frame: Optional[FrameRecord] = None

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Flush the GPU command queue to get accurate host-side timing.

        Calls the appropriate synchronization barrier based on the backend:
        - ``mps``: ``torch.mps.synchronize()``
        - ``cuda``: ``torch.cuda.synchronize()``
        """
        if not self.enabled:
            return
        import torch
        if self.backend == "mps":
            torch.mps.synchronize()
        elif self.backend == "cuda":
            torch.cuda.synchronize()
        # For "cpu" or unknown backends, no sync needed.

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def frame_start(self) -> None:
        """Begin timing a new inference frame.

        Records the frame index, unix timestamp, and a high-resolution
        start time. Must be paired with a subsequent :meth:`frame_end`.
        """
        if not self.enabled:
            return
        self.sync()
        now = time.perf_counter()
        self._frame_start_time = now
        self._last_mark_time = now
        self._current_frame = FrameRecord(
            frame_idx=self._frame_idx,
            timestamp_unix=time.time(),
        )

    def mark(self, label: str) -> None:
        """Record the elapsed time since the last mark (or frame_start).

        Each mark captures the duration of one logical component within a
        frame (e.g. ``"mimi_encode"``, ``"lm_forward"``).

        Before calling :meth:`mark`, the caller should call :meth:`sync` to
        ensure the GPU queue is flushed (or use the :meth:`timed` context
        manager which handles this automatically).

        Args:
            label: Name of the component being timed.
        """
        if not self.enabled:
            return
        if self._current_frame is None:
            return
        self.sync()
        now = time.perf_counter()
        elapsed_ms = (now - self._last_mark_time) * 1000.0
        self._current_frame.timings_ms[label] = elapsed_ms
        self._last_mark_time = now

    def frame_end(self, missed: Optional[bool] = None) -> None:
        """Finalize the current frame record.

        Computes total frame time, records GPU memory, and appends the
        frame to the log. The frame is flagged as ``missed`` if total
        time exceeds :attr:`frame_budget_ms` (unless explicitly overridden).

        Args:
            missed: If provided, overrides the automatic missed-frame
                detection. Pass ``False`` for frames where no tokens were
                generated (continue path).
        """
        if not self.enabled:
            return
        if self._current_frame is None:
            return
        self.sync()
        now = time.perf_counter()
        total_ms = (now - self._frame_start_time) * 1000.0
        self._current_frame.timings_ms["total"] = total_ms

        # Missed frame detection
        if missed is not None:
            self._current_frame.missed = missed
        else:
            self._current_frame.missed = total_ms > self.frame_budget_ms

        # GPU memory snapshot
        self._current_frame.memory_mb = _get_memory_mb(self.backend)

        self._frames.append(self._current_frame)
        self._frame_idx += 1
        self._current_frame = None

    # ------------------------------------------------------------------
    # Context manager for clean component timing
    # ------------------------------------------------------------------

    @contextmanager
    def timed(self, label: str) -> Generator[None, None, None]:
        """Context manager that syncs and records timing for a labelled block.

        Usage::

            with perf.timed("lm_forward"):
                tokens = lm_gen.step(codes)

        This is equivalent to calling :meth:`sync` before and after the block,
        then :meth:`mark` with the label. If the logger is disabled, the body
        executes with no overhead.
        """
        if not self.enabled:
            yield
            return
        self.sync()
        self._last_mark_time = time.perf_counter()
        yield
        self.mark(label)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all recorded frames.

        Returns a dict matching the summary JSON schema: frame counts,
        missed frame stats, per-component averages, percentile latencies,
        peak memory, and device metadata.
        """
        if not self._frames:
            return {"total_frames": 0, "backend": self.backend, "device": self.device_name}

        total_times = [f.timings_ms.get("total", 0.0) for f in self._frames]
        missed_count = sum(1 for f in self._frames if f.missed)
        total_frames = len(self._frames)

        result: Dict[str, Any] = {
            "total_frames": total_frames,
            "missed_frames": missed_count,
            "missed_pct": round(100.0 * missed_count / total_frames, 1),
            "avg_frame_ms": round(float(np.mean(total_times)), 1),
            "p50_frame_ms": round(float(np.percentile(total_times, 50)), 1),
            "p95_frame_ms": round(float(np.percentile(total_times, 95)), 1),
            "p99_frame_ms": round(float(np.percentile(total_times, 99)), 1),
            "max_frame_ms": round(float(np.max(total_times)), 1),
            "min_frame_ms": round(float(np.min(total_times)), 1),
        }

        # Per-component averages
        component_labels = ["mimi_encode", "lm_forward", "depformer", "mimi_decode"]
        for label in component_labels:
            values = [f.timings_ms[label] for f in self._frames if label in f.timings_ms]
            if values:
                result[f"avg_{label}_ms"] = round(float(np.mean(values)), 1)

        # Peak memory (driver allocation is the outer bound)
        driver_vals = [f.memory_mb.get("driver", 0.0) for f in self._frames if f.memory_mb]
        if driver_vals:
            result["peak_memory_mb"] = round(float(np.max(driver_vals)), 0)

        result["backend"] = self.backend
        result["device"] = self.device_name
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Union[str, Path]) -> None:
        """Write all frame records and summary to a JSON file.

        The output contains two top-level keys: ``"frames"`` (list of per-frame
        records) and ``"summary"`` (aggregate statistics).

        Args:
            path: Destination file path.
        """
        if not self.enabled:
            return
        data = {
            "frames": [f.to_dict() for f in self._frames],
            "summary": self.summary(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(data, fp, indent=2)

    def export_summary_json(self, path: Union[str, Path]) -> None:
        """Write only the summary statistics to a JSON file.

        Args:
            path: Destination file path.
        """
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.summary(), fp, indent=2)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def frames(self) -> List[FrameRecord]:
        """Return the list of recorded frame records (read-only view)."""
        return list(self._frames)

    @property
    def total_frames(self) -> int:
        """Number of frames recorded so far."""
        return len(self._frames)

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return (
            f"PerfLogger({status}, backend={self.backend!r}, "
            f"frames={len(self._frames)}, device={self.device_name!r})"
        )


# ======================================================================
# Module-level helpers
# ======================================================================


def _detect_device_name() -> str:
    """Best-effort detection of the device name for summary metadata."""
    machine = platform.machine()
    system = platform.system()
    if system == "Darwin":
        # Try to get the chip name on macOS
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return f"Apple Silicon ({machine})"
    return f"{system} {machine}"


def _get_memory_mb(backend: str) -> Dict[str, float]:
    """Snapshot current GPU memory usage in megabytes.

    Returns:
        Dict with ``"allocated"`` and ``"driver"`` keys (MB).
        Returns empty dict if memory tracking is unavailable.
    """
    try:
        import torch
        if backend == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
            driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
            return {
                "allocated": round(allocated, 1),
                "driver": round(driver, 1),
            }
        elif backend == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return {
                "allocated": round(allocated, 1),
                "driver": round(reserved, 1),
            }
    except Exception:
        pass
    return {}


def load_perf_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a previously exported perf JSON file.

    Args:
        path: Path to the JSON file produced by :meth:`PerfLogger.export_json`.

    Returns:
        The parsed dict with ``"frames"`` and ``"summary"`` keys.
    """
    with open(path) as fp:
        return json.load(fp)


def compare_summaries(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """Format a side-by-side comparison table of two perf summaries.

    Args:
        summary_a: Summary dict from the first run.
        summary_b: Summary dict from the second run.
        label_a: Column header for the first run.
        label_b: Column header for the second run.

    Returns:
        A formatted string suitable for printing to stderr / terminal.
    """
    # Collect all numeric keys from both summaries
    all_keys = []
    seen = set()
    for key in list(summary_a.keys()) + list(summary_b.keys()):
        if key not in seen:
            seen.add(key)
            all_keys.append(key)

    lines = []
    lines.append("")
    lines.append(f"{'Metric':<25s}  {label_a:>14s}  {label_b:>14s}  {'Delta':>14s}")
    lines.append("-" * 73)

    for key in all_keys:
        val_a = summary_a.get(key)
        val_b = summary_b.get(key)

        # Skip non-numeric fields for the delta column
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            pct = (delta / val_a * 100.0) if val_a != 0 else float("inf")
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"{key:<25s}  {val_a:>14.1f}  {val_b:>14.1f}  "
                f"{sign}{delta:>.1f} ({sign}{pct:.1f}%)"
            )
        else:
            str_a = str(val_a) if val_a is not None else "-"
            str_b = str(val_b) if val_b is not None else "-"
            lines.append(f"{key:<25s}  {str_a:>14s}  {str_b:>14s}  {'':>14s}")

    lines.append("-" * 73)
    lines.append("")
    return "\n".join(lines)
