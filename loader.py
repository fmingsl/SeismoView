#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loader.py
=========
Background data loading helpers for SeismoView.

This module scans files/directories, filters supported seismic waveform files,
and loads them into an ObsPy ``Stream`` inside a worker thread.

Author
------
M Fang

Created
-------
Legacy file; original creation date unknown.

Last Modified
-------------
2026-03-20
"""

import os
from typing import Iterable, List
from PyQt5.QtCore import QThread, pyqtSignal
from config import SUPPORTED_SEISMIC_EXTENSIONS

try:
    import obspy
    import obspy.io.mseed
    from obspy import read as obspy_read
    from obspy.core import Stream
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    Stream = None  # type: ignore[assignment]

SEISMIC_EXTENSIONS = SUPPORTED_SEISMIC_EXTENSIONS


def collect_seismic_files(paths: Iterable[str]) -> List[str]:
    """Collect supported waveform files from a mixture of files and folders.

    Parameters
    ----------
    paths:
        File or directory paths. Directories are scanned recursively.

    Returns
    -------
    list[str]
        De-duplicated normalized file paths in deterministic order.
    """
    result: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        """Add a file or recursively expand a directory into ``result``."""
        normalized = os.path.normpath(path)
        if normalized in seen:
            return
        seen.add(normalized)

        if os.path.isdir(normalized):
            for root, _, files in os.walk(normalized):
                for filename in sorted(files):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SEISMIC_EXTENSIONS:
                        _add(os.path.join(root, filename))
            return

        if os.path.isfile(normalized):
            result.append(normalized)

    for input_path in paths:
        _add(input_path)

    return result


class DataLoaderThread(QThread):
    """Load one or more waveform files in a worker thread.

    Signals
    -------
    finished(stream, label)
        Emitted when at least one waveform is loaded successfully.
    error(message)
        Emitted on unrecoverable failures or when every file fails.
    progress(percent)
        High-level progress in the range 0-100.
    file_progress(current, total)
        Per-file progress, mainly for status-bar feedback.
    """

    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    file_progress = pyqtSignal(int, int)

    def __init__(self, paths, parent=None):
        """Store source paths for deferred background loading."""
        super().__init__(parent)
        self._paths = [paths] if isinstance(paths, str) else list(paths)
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation for the current load job."""
        self._cancelled = True

    def run(self) -> None:
        """Scan and load files, merging successful traces into one stream."""
        try:
            if not OBSPY_AVAILABLE:
                self.error.emit(
                    "ObsPy 未安装，无法读取地震数据文件。\n"
                    "请运行: pip install obspy"
                )
                return

            # 1. 收集所有目标文件
            self.progress.emit(5)
            files = collect_seismic_files(self._paths)
            if not files:
                self.error.emit(
                    "未找到可识别的地震数据文件。\n"
                    f"支持的扩展名：{', '.join(sorted(SEISMIC_EXTENSIONS))}"
                )
                return

            total = len(files)
            merged = Stream()
            failed = []

            for index, file_path in enumerate(files, start=1):
                if self._cancelled:
                    self.error.emit("加载已取消。")
                    return

                self.file_progress.emit(index, total)
                self.progress.emit(10 + int(80 * index / total))

                try:
                    merged += obspy_read(file_path)
                except Exception as exc:  # pragma: no cover - GUI error path
                    failed.append((os.path.basename(file_path), str(exc)))

            if len(merged) == 0:
                details = '\n'.join(f"  {name}: {err}" for name, err in failed[:10])
                self.error.emit(f"所有文件均读取失败：\n{details}")
                return

            # 3. 合并同名道（拼接时间段）
            try:
                # Merge same-id traces into continuous segments when possible.
                merged.merge(method=1, fill_value='interpolate')
            except Exception:
                pass

            self.progress.emit(100)
            self.finished.emit(merged, self._build_label(files, failed))

        except Exception as exc:  # pragma: no cover - GUI error path
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    @staticmethod
    def _build_label(files: List[str], failed: List[tuple[str, str]]) -> str:
        """Build a concise UI label summarizing a load result."""
        total = len(files)
        if total == 1:
            label = os.path.basename(files[0])
        else:
            label = (
                f"{total} 个文件合并"
                f"（{os.path.basename(files[0])} … {os.path.basename(files[-1])}）"
            )
        if failed:
            label += f"  ⚠ {len(failed)} 个文件读取失败"
        return label
