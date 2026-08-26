"""Small standard-library runtime observations for reproducibility reports."""

from __future__ import annotations

import os


def peak_process_memory_bytes() -> int | None:
    """Return the process peak working set when the host exposes it."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCountersEx),
                wintypes.DWORD,
            ]
            get_memory_info.restype = wintypes.BOOL
            success = get_memory_info(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.PeakWorkingSetSize) if success else None
        except (AttributeError, OSError):
            return None
    try:
        import resource

        # Linux reports KiB; macOS reports bytes.
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if os.uname().sysname == "Darwin" else peak * 1024
    except (ImportError, AttributeError):
        return None
