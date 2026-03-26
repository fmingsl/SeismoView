#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panels.py
=========
Sidebar widgets used by SeismoView.

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

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from config import COLORS


class HeaderPanel(QWidget):
    """Display waveform header metadata and emit selected trace indices."""

    trace_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        """Initialize the panel and create its child widgets."""
        super().__init__(parent)
        self._stream = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the channel list and the bottom detail panel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_bar = QWidget()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet(
            f"background:{COLORS['bg_header']};border-bottom:1px solid {COLORS['border']};"
        )
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(12, 0, 8, 0)

        title_lbl = QLabel("◈  数 据 通 道")
        title_lbl.setStyleSheet(
            f"color:{COLORS['accent_blue']}; font-weight:700;"
            f" font-size:11px; letter-spacing:2px; background:transparent;"
        )
        tb_layout.addWidget(title_lbl)
        tb_layout.addStretch()

        self.count_lbl = QLabel("0 条记录")
        self.count_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px; background:transparent;"
        )
        tb_layout.addWidget(self.count_lbl)
        layout.addWidget(title_bar)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["台网", "台站", "位置", "通道"])
        self.tree.setAlternatingRowColors(False)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(True)

        header = self.tree.header()
        for column, width in enumerate([60, 65, 55, 65]):
            self.tree.setColumnWidth(column, width)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)

        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)

        detail_frame = QWidget()
        detail_frame.setFixedHeight(110)
        detail_frame.setStyleSheet(
            f"background:{COLORS['bg_header']};border-top:1px solid {COLORS['border']};"
        )
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(10, 6, 10, 6)
        detail_layout.setSpacing(4)

        detail_title = QLabel("▸  详细头段信息")
        detail_title.setStyleSheet(
            f"color:{COLORS['text_secondary']}; font-size:10px;"
            f" font-weight:700; letter-spacing:1px; background:transparent;"
        )
        detail_layout.addWidget(detail_title)

        self.detail_lbl = QLabel("请在上方列表中选择一条通道记录")
        self.detail_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:11px;"
            f" font-family:Consolas; background:transparent; line-height:1.6;"
        )
        self.detail_lbl.setWordWrap(True)
        detail_layout.addWidget(self.detail_lbl)
        layout.addWidget(detail_frame)

    def load_stream(self, stream) -> None:
        """Populate the tree using an ObsPy ``Stream``."""
        self._stream = stream
        self.tree.clear()

        for index, trace in enumerate(stream):
            stats = trace.stats
            item = QTreeWidgetItem([
                stats.network or "--",
                stats.station or "--",
                stats.location or "--",
                stats.channel or "--",
            ])
            item.setData(0, Qt.UserRole, index)

            channel = stats.channel or ""
            if channel.startswith(('BH', 'HH')):
                item.setForeground(1, QColor(COLORS['accent_blue']))
            elif channel.startswith('LH'):
                item.setForeground(1, QColor(COLORS['accent_green']))
            elif channel.startswith(('SH', 'EH')):
                item.setForeground(1, QColor(COLORS['accent_amber']))
            else:
                item.setForeground(1, QColor(COLORS['text_primary']))

            self.tree.addTopLevelItem(item)

        self.count_lbl.setText(f"{len(stream)} 条记录")
        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _on_selection_changed(self) -> None:
        """Refresh the detail area and emit selected trace indices."""
        selected = self.tree.selectedItems()
        indices = [item.data(0, Qt.UserRole) for item in selected]

        if not indices or self._stream is None:
            self.detail_lbl.setText("请在上方列表中选择一条通道记录")
            return

        trace = self._stream[indices[0]]
        stats = trace.stats
        duration = stats.endtime - stats.starttime
        text = (
            f"SEED ID: {stats.network}.{stats.station}.{stats.location}.{stats.channel}   "
            f"采样率: {stats.sampling_rate} Hz   采样间隔: {stats.delta:.4f}s   "
            f"数据点数: {stats.npts:,}   时长: {duration:.3f}s\n"
            f"开始: {str(stats.starttime)[:23]} UTC   "
            f"结束: {str(stats.endtime)[:23]} UTC   "
            f"格式: {getattr(stats, '_format', 'Unknown') or 'Unknown'}"
        )
        self.detail_lbl.setText(text)
        self.trace_selected.emit(indices)

    def clear(self) -> None:
        """Reset the widget state after unloading data."""
        self.tree.clear()
        self._stream = None
        self.count_lbl.setText("0 条记录")
        self.detail_lbl.setText("请在上方列表中选择一条通道记录")
