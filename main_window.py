#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_window.py
==============
Main application window for SeismoView.

This module coordinates UI layout, data loading, waveform plotting,
preprocessing, phase picking, and export workflows.

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
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QMenu, QAction, QToolBar,
    QToolButton,
    QMessageBox, QFrame, QSizePolicy, QProgressBar, QScrollArea,
    QDoubleSpinBox, QSpinBox, QComboBox, QStackedWidget, QStyle,
    QDialogButtonBox, QDialog, QFormLayout,
    QTabWidget, QTextEdit, QCheckBox, QGroupBox, QLineEdit,QApplication
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont

from config import COLORS,STYLESHEET
from loader import DataLoaderThread
from canvas_seismic import SeismicCanvas
from panels import HeaderPanel
# from spectrum_windows import SpectrumWindow, PSDWindow, XCorrWindow
from batch_processor  import BatchProcessDialog

class MainWindow(QMainWindow):
    """Top-level Qt window coordinating SeismoView's UI and workflows."""
    def __init__(self):
        """Initialize the main window, state caches, and child widgets."""
        super().__init__()
        self.stream        = None
        self._stream_unit = 'counts'
        self.loader        = None
        self._loaded_paths = []          # 最近一次加载的路径列表（用于 reload）
        self._last_dir     = ""          # 记住上次打开的目录
        self._orig_stream  = None
        self._proc_history = []
        self._spec_wins    = []   # 防 GC：独立频谱窗口引用
        self._psd_wins     = []   # 防 GC：独立 PSD 窗口引用
        self._xcorr_wins   = []
        self._setup_window()
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()
        self.setAcceptDrops(True)

    # ── 窗口基础设置 ──────────────────────────────────────────────────────
    def _setup_window(self):
        self.setWindowTitle("SeismoView — 地震波形查看器")
        self.setMinimumSize(1280, 820)
        self.resize(1600, 960)
        self.setStyleSheet(STYLESHEET)
        # 居中显示
        screen = QApplication.desktop().availableGeometry()
        self.move(
            (screen.width()  - 1600) // 2,
            (screen.height() - 960)  // 2
        )

    # ── 主界面布局 ─────────────────────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 水平分割器：左侧面板 | 右侧画布
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        # ── 左侧面板 ──────────────────────────────────────────────────────
        left_panel = QWidget()
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 文件信息区
        self.file_info_widget = QWidget()
        self.file_info_widget.setFixedHeight(70)
        self.file_info_widget.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        fi_layout = QVBoxLayout(self.file_info_widget)
        fi_layout.setContentsMargins(12, 8, 12, 8)
        fi_layout.setSpacing(2)

        self.file_name_lbl = QLabel("未加载文件")
        self.file_name_lbl.setStyleSheet(
            f"color:{COLORS['text_primary']}; font-size:12px;"
            f" font-weight:600; background:transparent;"
        )
        fi_layout.addWidget(self.file_name_lbl)

        self.file_meta_lbl = QLabel("拖拽文件至窗口，或使用「文件」菜单打开")
        self.file_meta_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px;"
            f" background:transparent;"
        )
        fi_layout.addWidget(self.file_meta_lbl)
        left_layout.addWidget(self.file_info_widget)

        # 头段面板
        self.header_panel = HeaderPanel()
        self.header_panel.trace_selected.connect(self._on_trace_selected)
        left_layout.addWidget(self.header_panel)

        # 控制按钮区
        ctrl_frame = QWidget()
        ctrl_frame.setFixedHeight(52)
        ctrl_frame.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-top:1px solid {COLORS['border']};"
        )
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(8, 5, 8, 5)
        ctrl_layout.setSpacing(6)

        self.plot_all_btn = QPushButton("绘制全部")
        self.plot_all_btn.setObjectName("accent_btn")
        self.plot_all_btn.setFixedHeight(30)
        self.plot_all_btn.setMinimumWidth(60)
        self.plot_all_btn.setToolTip("绘制文件中所有通道的波形")
        self.plot_all_btn.clicked.connect(self._plot_all)

        self.plot_sel_btn = QPushButton("绘制选中")
        self.plot_sel_btn.setFixedHeight(30)
        self.plot_sel_btn.setMinimumWidth(60)
        self.plot_sel_btn.setToolTip("仅绘制在列表中选中的通道（可多选）")
        self.plot_sel_btn.clicked.connect(self._plot_selected)

        self.reset_btn = QPushButton("重置视图")
        self.reset_btn.setFixedHeight(30)
        self.reset_btn.setMinimumWidth(60)
        self.reset_btn.setToolTip("将波形缩放/平移还原至初始状态")
        self.reset_btn.clicked.connect(self._reset_view)

        ctrl_layout.addWidget(self.plot_all_btn)
        ctrl_layout.addWidget(self.plot_sel_btn)
        ctrl_layout.addWidget(self.reset_btn)
        left_layout.addWidget(ctrl_frame)

        splitter.addWidget(left_panel)

        # ── 右侧画布区 ────────────────────────────────────────────────────
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        # ── 画布工具栏（顶部）─────────────────────────────────────────────
        canvas_toolbar = QWidget()
        canvas_toolbar.setFixedHeight(40)
        canvas_toolbar.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        ct_layout = QHBoxLayout(canvas_toolbar)
        ct_layout.setContentsMargins(10, 4, 10, 4)
        ct_layout.setSpacing(6)

        self.canvas_title = QLabel("波形显示区")
        self.canvas_title.setStyleSheet(
            f"color:{COLORS['text_secondary']}; font-size:11px;"
            f" font-weight:700; letter-spacing:1px; background:transparent;"
        )
        ct_layout.addWidget(self.canvas_title)
        ct_layout.addStretch()

        hint_lbl = QLabel("🖱  滚轮缩放 · 拖拽平移")
        hint_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px; background:transparent;"
        )
        ct_layout.addWidget(hint_lbl)

        for text, slot, tip in [
            ("⊕  放大", self._zoom_in,   "放大时间轴（Ctrl++）"),
            ("⊖  缩小", self._zoom_out,  "缩小时间轴（Ctrl+-）"),
            ("⌂  重置", self._reset_view,"重置视图（Ctrl+R）"),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(80, 28)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            ct_layout.addWidget(btn)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background:{COLORS['border_bright']};")
        ct_layout.addWidget(sep)

        dist_btn = QPushButton("📏 震中距排列")
        dist_btn.setFixedHeight(28)
        dist_btn.setToolTip("根据震中距对波形道排序（需要加载事件信息和台站坐标）")
        dist_btn.clicked.connect(self._show_distance_sort_dialog)
        ct_layout.addWidget(dist_btn)

        canvas_layout.addWidget(canvas_toolbar)
        # ──────────────────────────────────────────────────────────────────

        # ── 主波形画布（频谱分析已迁至独立弹出窗口）──────────────────────
        self.canvas = SeismicCanvas()
        self.canvas.status_message.connect(self._set_status)
        self.canvas.pick_added.connect(self._on_pick_added)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.canvas.trim_requested.connect(self._on_trim_requested)
        canvas_layout.addWidget(self.canvas)

        # ── 进度条 ────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{ background:{COLORS['bg_deep']}; border:none; }}"
            f"QProgressBar::chunk {{ background:{COLORS['accent_blue']}; }}"
        )
        self.progress_bar.hide()
        canvas_layout.addWidget(self.progress_bar)

        splitter.addWidget(canvas_container)

        # 分割比例
        splitter.setSizes([300, 1350])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        
        # ── 全宽可折叠功能面板（工具栏正下方）───────────────────────────────
        # 此处在 splitter 前构建，使面板横跨整个窗口宽度
        self.context_panel = QWidget()
        self.context_panel.setFixedHeight(88)
        self.context_panel.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:2px solid {COLORS['border_bright']};"
        )
        cp_layout = QHBoxLayout(self.context_panel)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        cp_layout.setSpacing(0)

        self.context_stack = QStackedWidget()
        self.context_stack.addWidget(self._build_prep_toolbar())   # 0：预处理
        self.context_stack.addWidget(self._build_pick_toolbar())   # 1：拾取
        cp_layout.addWidget(self.context_stack)

        # 右侧收起按钮（"▲ 收起"）
        hide_btn = QPushButton("▲  收起")
        hide_btn.setFixedSize(72, 28)
        hide_btn.setToolTip("收起功能面板")
        hide_btn.setStyleSheet(
            f"QPushButton {{ color:{COLORS['text_muted']}; border:none;"
            f" background:transparent; font-size:10px; }}"
            f"QPushButton:hover {{ color:{COLORS['accent_red']};"
            f" background:{COLORS['border']}; border-radius:4px; }}"
        )
        hide_btn.clicked.connect(self.context_panel.hide)
        hide_wrap = QVBoxLayout()
        hide_wrap.setContentsMargins(4, 9, 10, 9)
        hide_wrap.addWidget(hide_btn)
        cp_layout.addLayout(hide_wrap)

        # 插入到 main_layout 的最前面（splitter 之前）
        main_layout.insertWidget(0, self.context_panel)
        self.context_panel.hide()
        
    def _toggle_context_panel(self, index):
        """控制动态面板的显示、隐藏与切换"""
        if self.context_panel.isVisible() and self.context_stack.currentIndex() == index:
            self.context_panel.hide()
        else:
            self.context_stack.setCurrentIndex(index)
            self.context_panel.show()

    # ── 频谱分析独立窗口 ──────────────────────────────────────────────────────
    def _current_view_args(self):
        """返回 (raw_t, raw_data, raw_meta, active_idx, xmin, xmax)"""
        raw_t    = self.canvas._raw_t
        raw_data = self.canvas._raw_data
        raw_meta = self.canvas._raw_meta
        active_idx = 0
        xmin, xmax = 0.0, 1.0
        if self.canvas._axes:
            xmin, xmax = self.canvas._axes[0].get_xlim()
        return raw_t, raw_data, raw_meta, active_idx, xmin, xmax

    def _open_spectrum_window(self):
        """弹出独立振幅频谱分析窗口"""
        if not self.canvas._raw_t:
            QMessageBox.information(self, "提示", "请先加载数据文件并绘制波形")
            return
        from spectrum_windows import SpectrumWindow
        args = self._current_view_args()
        win = SpectrumWindow(*args, parent=None)
        win.destroyed.connect(lambda: self._spec_wins.remove(win)
                              if win in self._spec_wins else None)
        self._spec_wins.append(win)
        win.show()

    def _open_psd_window(self):
        """弹出独立功率谱密度分析窗口（包含内置预处理管道）"""
        if not self.stream:
            QMessageBox.information(self, "提示", "请先加载数据文件并绘制波形")
            return
        from spectrum_windows import PSDWindow
        _, _, raw_meta, active_idx, xmin, xmax = self._current_view_args()
        inv = getattr(self, '_inv_inventory', None)
        win = PSDWindow(self.stream, inv, raw_meta, active_idx, xmin, xmax,
                        data_unit=getattr(self, '_stream_unit', 'counts'),
                        parent=None)
        win.destroyed.connect(lambda: self._psd_wins.remove(win)
                              if win in self._psd_wins else None)
        self._psd_wins.append(win)
        win.show()

    def _open_xcorr_window(self):
        """弹出多道互相关分析窗口"""
        if not self.stream or len(self.stream) < 2:
            QMessageBox.information(
                self, "提示",
                "多道互相关需要至少 2 道波形数据，请先加载数据文件。")
            return
        from spectrum_windows import XCorrWindow
        _, _, raw_meta, active_idx, xmin, xmax = self._current_view_args()
        # 传入台站坐标字典（若已加载）
        sta_coords = getattr(self, '_sta_coords_cache', {})
        win = XCorrWindow(self.stream, raw_meta, active_idx, xmin, xmax,
                          sta_coords=sta_coords, parent=None)
        win.destroyed.connect(lambda: self._xcorr_wins.remove(win)
                              if win in self._xcorr_wins else None)
        self._xcorr_wins.append(win)
        win.show()

    # ── 批量预处理流水线 ────────────────────────────────────────────────────
    def _open_batch_dialog(self):
        """弹出批量预处理流水线对话框"""
        inv = getattr(self, '_inv_inventory', None)
        dlg = BatchProcessDialog(
            inventory=inv,
            last_dir=self._last_dir,
            parent=self)
        dlg.exec_()
        if dlg._last_dir:
            self._last_dir = dlg._last_dir

    def _on_view_changed(self, xmin: float, xmax: float):
        """波形视图变化 → 推送最新数据到所有已打开的分析窗口"""
        raw_t    = self.canvas._raw_t
        raw_data = self.canvas._raw_data
        raw_meta = self.canvas._raw_meta
        for w in list(self._spec_wins):
            try:
                w.push_update(raw_t, raw_data, raw_meta,
                              w._active_idx, xmin, xmax)
            except RuntimeError:
                if w in self._spec_wins:
                    self._spec_wins.remove(w)
        for w in list(self._psd_wins):
            try:
                w.push_update(self.stream, raw_meta,
                              w._active_idx, xmin, xmax,
                              getattr(self, '_stream_unit', 'counts'))
            except RuntimeError:
                if w in self._psd_wins:
                    self._psd_wins.remove(w)
        for w in list(self._xcorr_wins):
            try:
                w.push_update(self.stream, raw_meta,
                              w._active_idx, xmin, xmax)
            except RuntimeError:
                if w in self._xcorr_wins:
                    self._xcorr_wins.remove(w)

    def _build_prep_toolbar(self):
        """构建：波形预处理面板（双行）
        第一行：操作按钮
        第二行：处理流程历史 chip 标签 + 重置
        """
        outer = QWidget()
        vlay = QVBoxLayout(outer)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        # ══════════════════════════════════════════════════════
        # 第一行：操作按钮区
        # ══════════════════════════════════════════════════════
        row1 = QWidget()
        row1.setFixedHeight(44)
        row1.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        r1 = QHBoxLayout(row1)
        r1.setContentsMargins(12, 6, 12, 6)
        r1.setSpacing(6)

        # 分类标签
        cat_lbl = QLabel("▸ 预处理")
        cat_lbl.setStyleSheet(
            f"color:{COLORS['accent_blue']}; font-size:10px;"
            f" font-weight:700; letter-spacing:1px;"
        )
        r1.addWidget(cat_lbl)
        r1.addWidget(self._make_vsep())

        def _prep_btn(text, tip):
            b = QPushButton(text)
            b.setFixedHeight(28)
            b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_card']};"
                f"color:{COLORS['text_primary']};"
                f"border:1px solid {COLORS['border_bright']};"
                f"border-radius:5px;font-size:11px;padding:0 10px;}}"
                f"QPushButton:hover{{background:{COLORS['selection']};"
                f"border-color:{COLORS['accent_blue']};"
                f"color:{COLORS['accent_blue']};}}"
                f"QPushButton:pressed{{background:#DBEAFE;}}"
            )
            return b

        btn_demean  = _prep_btn("去均值",   "去除直流偏移（Demean）")
        btn_detrend = _prep_btn("去趋势",   "线性去趋势（Detrend）")
        btn_norm    = _prep_btn("归一化 ▾", "选择归一化方式")
        btn_taper   = _prep_btn("Taper ▾",  "选择锥化窗函数类型和参数")
        btn_filter  = _prep_btn("滤波 ▾",   "选择滤波器类型和参数")
        btn_resp    = _prep_btn("去仪器响应 ▾", "去除仪器响应（StationXML / PAZ）")
        btn_trim    = _prep_btn("✂ 截窗 ▾", "截取波形时间窗口（手动点击或输入时间）")
        btn_rotate  = _prep_btn("🔄 分量旋转 ▾", "旋转三分量（ZNE→ZRT / ZNE→LQT / NE→RT）")
        btn_resamp  = _prep_btn("⇅ 重采样 ▾", "升采样 / 降采样，修改数据采样率")

        btn_demean.clicked.connect(self._process_demean)
        btn_detrend.clicked.connect(self._process_detrend)
        btn_norm.clicked.connect(self._show_normalize_menu)
        btn_taper.clicked.connect(self._show_taper_dialog)
        btn_filter.clicked.connect(self._show_filter_dialog)
        btn_resp.clicked.connect(self._show_remove_response_dialog)
        btn_trim.clicked.connect(self._show_trim_menu)
        btn_rotate.clicked.connect(self._show_rotate_dialog)
        btn_resamp.clicked.connect(self._show_resample_dialog)

        for b in [btn_demean, btn_detrend, btn_norm, btn_taper, btn_filter,
                  btn_resp, btn_trim, btn_rotate, btn_resamp]:
            r1.addWidget(b)

        r1.addWidget(self._make_vsep())

        # 重置原始按钮
        btn_reset = QPushButton("↺  重置原始")
        btn_reset.setFixedHeight(28)
        btn_reset.setToolTip("还原到加载时的原始波形，清空所有预处理步骤")
        btn_reset.setStyleSheet(
            f"QPushButton{{background:#FEF3C7;"
            f"color:#92400E;"
            f"border:1px solid #F59E0B;"
            f"border-radius:5px;font-size:11px;font-weight:600;padding:0 10px;}}"
            f"QPushButton:hover{{background:#FDE68A;border-color:#D97706;}}"
            f"QPushButton:pressed{{background:#FCD34D;}}"
        )
        btn_reset.clicked.connect(self._reset_to_original)
        r1.addWidget(btn_reset)

        r1.addStretch()
        vlay.addWidget(row1)

        # ══════════════════════════════════════════════════════
        # 第二行：处理流程历史记录
        # ══════════════════════════════════════════════════════
        row2 = QWidget()
        row2.setFixedHeight(44)
        row2.setStyleSheet(f"background:{COLORS['bg_deep']};")
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(12, 6, 12, 6)
        r2.setSpacing(6)

        hist_lbl = QLabel("流程：")
        hist_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px;"
            f" font-weight:700; letter-spacing:0.5px; white-space:nowrap;"
        )
        hist_lbl.setFixedWidth(42)
        r2.addWidget(hist_lbl)

        # 可横向滚动的 chip 区域
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(32)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:horizontal{height:4px;border-radius:2px;"
            f"background:{COLORS['bg_card']};}}"
            f"QScrollBar::handle:horizontal{{background:{COLORS['border_bright']};"
            "border-radius:2px;min-width:20px;}"
        )

        self._hist_chip_widget = QWidget()
        self._hist_chip_widget.setStyleSheet("background:transparent;")
        self._hist_chip_layout = QHBoxLayout(self._hist_chip_widget)
        self._hist_chip_layout.setContentsMargins(0, 0, 0, 0)
        self._hist_chip_layout.setSpacing(4)
        self._hist_chip_layout.addStretch()
        scroll.setWidget(self._hist_chip_widget)
        r2.addWidget(scroll)

        # 初始占位文字
        self._hist_empty_lbl = QLabel("（尚未进行任何预处理）")
        self._hist_empty_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px; font-style:italic;"
        )
        self._hist_chip_layout.insertWidget(0, self._hist_empty_lbl)

        vlay.addWidget(row2)
        return outer

    # ── 归一化菜单 ────────────────────────────────────────────────────────────
    def _show_normalize_menu(self):
        """点击「归一化▾」按钮时弹出选择菜单"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return
        menu = QMenu(self)
        menu.addAction("最大值归一化  (Max)",       lambda: self._process_normalize('max'))
        menu.addAction("峰峰值归一化  (Peak-Peak)", lambda: self._process_normalize('peak_peak'))
        menu.addAction("RMS 归一化",                 lambda: self._process_normalize('rms'))
        # 弹出位置跟随按钮
        btn = self.sender()
        menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft()))

    # ── 滤波对话框 ────────────────────────────────────────────────────────────
    def _show_filter_dialog(self):
        """弹出滤波参数配置对话框，支持带通 / 低通 / 高通"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        from PyQt5.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox,
                                      QComboBox, QDoubleSpinBox, QLabel as QL,
                                      QStackedWidget)

        dlg = QDialog(self)
        dlg.setWindowTitle("滤波器设置")
        dlg.setMinimumWidth(340)
        dlg.setStyleSheet(self.styleSheet())
        main = QVBoxLayout(dlg)
        main.setSpacing(10)
        main.setContentsMargins(16, 14, 16, 14)

        # 滤波类型选择
        type_row = QHBoxLayout()
        type_row.addWidget(QL("滤波类型："))
        type_combo = QComboBox()
        type_combo.addItems(["带通  (Bandpass)", "低通  (Lowpass)", "高通  (Highpass)"])
        type_combo.setFixedHeight(28)
        type_row.addWidget(type_combo)
        main.addLayout(type_row)

        # 参数区（堆叠，按滤波类型切换）
        stack = QStackedWidget()

        # ── 带通参数 ──
        bp_w = QWidget()
        bp_f = QFormLayout(bp_w)
        bp_fmin = QDoubleSpinBox(); bp_fmin.setRange(0.001, 500); bp_fmin.setValue(1.0); bp_fmin.setSuffix(" Hz")
        bp_fmax = QDoubleSpinBox(); bp_fmax.setRange(0.001, 500); bp_fmax.setValue(10.0); bp_fmax.setSuffix(" Hz")
        bp_f.addRow("最低频率：", bp_fmin)
        bp_f.addRow("最高频率：", bp_fmax)
        stack.addWidget(bp_w)

        # ── 低通参数 ──
        lp_w = QWidget()
        lp_f = QFormLayout(lp_w)
        lp_freq = QDoubleSpinBox(); lp_freq.setRange(0.001, 500); lp_freq.setValue(10.0); lp_freq.setSuffix(" Hz")
        lp_f.addRow("截止频率：", lp_freq)
        stack.addWidget(lp_w)

        # ── 高通参数 ──
        hp_w = QWidget()
        hp_f = QFormLayout(hp_w)
        hp_freq = QDoubleSpinBox(); hp_freq.setRange(0.001, 500); hp_freq.setValue(1.0); hp_freq.setSuffix(" Hz")
        hp_f.addRow("截止频率：", hp_freq)
        stack.addWidget(hp_w)

        type_combo.currentIndexChanged.connect(stack.setCurrentIndex)
        main.addWidget(stack)

        # Corners 选项（阶数）
        corners_row = QHBoxLayout()
        corners_row.addWidget(QL("Butterworth 阶数："))
        corner_spin = QSpinBox(); corner_spin.setRange(1, 8); corner_spin.setValue(4)
        corners_row.addWidget(corner_spin)
        corners_row.addStretch()
        main.addLayout(corners_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        main.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        ftype_idx = type_combo.currentIndex()
        corners   = corner_spin.value()
        idxs = self._selected_stream_indices()
        try:
            if ftype_idx == 0:   # 带通
                fmin, fmax = bp_fmin.value(), bp_fmax.value()
                if fmin >= fmax:
                    QMessageBox.warning(self, "参数错误", "最低频率必须小于最高频率！")
                    return
                for i in idxs:
                    self.stream[i].filter('bandpass', freqmin=fmin, freqmax=fmax,
                                          corners=corners, zerophase=True)
                desc = f"带通 {fmin}-{fmax}Hz"
            elif ftype_idx == 1:  # 低通
                freq = lp_freq.value()
                for i in idxs:
                    self.stream[i].filter('lowpass', freq=freq, corners=corners,
                                          zerophase=True)
                desc = f"低通 {freq}Hz"
            else:                  # 高通
                freq = hp_freq.value()
                for i in idxs:
                    self.stream[i].filter('highpass', freq=freq, corners=corners,
                                          zerophase=True)
                desc = f"高通 {freq}Hz"

            self._add_proc_step(desc, color=COLORS['accent_blue'])
            self._set_status(
                f"滤波完成：{desc}（Butterworth {corners} 阶）  |  作用于 {len(idxs)} 道")
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "滤波失败", f"处理时发生错误：\n{str(e)}")

    # ── Taper 对话框 ──────────────────────────────────────────────────────────
    def _show_taper_dialog(self):
        """弹出 Taper 参数配置对话框"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Taper 设置")
        dlg.setMinimumWidth(360)
        main = QVBoxLayout(dlg)
        main.setSpacing(10)
        main.setContentsMargins(16, 14, 16, 14)

        form = QFormLayout()
        form.setSpacing(8)

        # ── 窗函数类型 ────────────────────────────────────────────────────
        type_combo = QComboBox()
        type_combo.addItems([
            "Cosine（余弦，推荐）",
            "Hanning（汉宁窗）",
            "Hamming（汉明窗）",
            "Blackman（布莱克曼窗）",
            "Bartlett（巴特利特窗）",
        ])
        type_combo.setFixedHeight(28)
        form.addRow("窗函数类型：", type_combo)

        # ── Taper 比例 ────────────────────────────────────────────────────
        pct_spin = QDoubleSpinBox()
        pct_spin.setRange(0.001, 0.5)
        pct_spin.setValue(0.05)
        pct_spin.setSingleStep(0.01)
        pct_spin.setDecimals(3)
        pct_spin.setToolTip("每端锥化占信号总长度的比例（0.001–0.5）")
        # 实时显示对应百分比
        pct_label = QLabel("→ 每端  5.0 %")
        pct_label.setStyleSheet(f"color:{COLORS['text_muted']};font-size:10px;")
        def _on_pct_change(v):
            pct_label.setText(f"→ 每端 {v*100:.1f} %")
        pct_spin.valueChanged.connect(_on_pct_change)
        pct_row = QHBoxLayout()
        pct_row.addWidget(pct_spin)
        pct_row.addWidget(pct_label)
        form.addRow("锥化比例：", pct_row)

        # ── 作用端 ────────────────────────────────────────────────────────
        side_combo = QComboBox()
        side_combo.addItems(["两端（Both ends）", "仅左端（Left）", "仅右端（Right）"])
        side_combo.setFixedHeight(28)
        form.addRow("作用端：", side_combo)

        main.addLayout(form)

        # ── 说明文字 ─────────────────────────────────────────────────────
        note = QLabel(
            "Taper 在信号两端乘以渐变窗函数，消除边缘不连续性，\n"
            "可有效抑制频谱泄漏。建议在滤波前先进行 Taper 处理。"
        )
        note.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:6px 8px;"
        )
        note.setWordWrap(True)
        main.addWidget(note)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        main.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        taper_pct  = pct_spin.value()
        side_idx   = side_combo.currentIndex()   # 0=both 1=left 2=right
        type_idx   = type_combo.currentIndex()
        self._process_taper(taper_pct, side_idx, type_idx)

    def _process_taper(self, max_percentage: float, side_idx: int, type_idx: int):
        """
        对所有道执行 Taper。

        Parameters
        ----------
        max_percentage : float  每端锥化比例 (0–0.5)
        side_idx       : int    0=both 1=left 2=right
        type_idx       : int    0=cosine 1=hanning 2=hamming 3=blackman 4=bartlett
        """
        if not self.stream:
            return

        type_names = ['cosine', 'hanning', 'hamming', 'blackman', 'bartlett']
        type_labels = ['Cosine', 'Hanning', 'Hamming', 'Blackman', 'Bartlett']
        side_labels = ['两端', '左端', '右端']
        wtype = type_names[type_idx]
        wlabel = type_labels[type_idx]
        slabel = side_labels[side_idx]

        idxs = self._selected_stream_indices()
        try:
            for i in idxs:
                tr  = self.stream[i]
                n   = len(tr.data)
                d   = tr.data.astype(float)
                win = self._make_taper_window(wtype, n, max_percentage, side_idx)
                tr.data = d * win

            chip_color = COLORS['accent_amber']
            desc = f"Taper({wlabel},{max_percentage*100:.1f}%,{slabel})"
            self._add_proc_step(desc, color=chip_color)
            self._set_status(
                f"Taper 完成：{wlabel} 窗，{max_percentage*100:.1f}%，{slabel}  |  "
                f"作用于 {len(idxs)} 道"
            )
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "Taper 失败", str(e))

    @staticmethod
    def _make_taper_window(wtype: str, n: int,
                           pct: float, side: int) -> np.ndarray:
        """
        构造长度为 n 的 Taper 窗向量。
        side: 0=both 1=left 2=right
        """
        win = np.ones(n, dtype=float)
        k   = max(2, int(np.floor(n * pct)))   # 每端锥化采样点数

        # 半窗（上升沿 0→1，长度 k）
        if wtype == 'cosine':
            taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / (k - 1)))
        elif wtype == 'hanning':
            full = np.hanning(2 * k)
            taper = full[:k]
        elif wtype == 'hamming':
            full = np.hamming(2 * k)
            taper = full[:k]
        elif wtype == 'blackman':
            full = np.blackman(2 * k)
            taper = full[:k]
        elif wtype == 'bartlett':
            full = np.bartlett(2 * k)
            taper = full[:k]
        else:
            taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(k) / (k - 1)))

        if side in (0, 1):   # 左端
            win[:k] = taper
        if side in (0, 2):   # 右端
            win[-k:] = taper[::-1]

        return win

    # ── 去仪器响应对话框 ────────────────────────────────────────────────────────
    def _show_remove_response_dialog(self):
        """弹出去仪器响应配置对话框（StationXML / PAZ 两种模式）"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("去仪器响应")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # ────────────────────────────────────────────────────────────────
        # 方法选择 Tab
        # ────────────────────────────────────────────────────────────────
        from PyQt5.QtWidgets import QTabWidget, QTextEdit, QCheckBox, QGroupBox
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabBar::tab{{padding:5px 14px;font-size:11px;}}"
            f"QTabBar::tab:selected{{color:{COLORS['accent_blue']};"
            f"border-bottom:2px solid {COLORS['accent_blue']};}}"
        )

        # ════════════════════════════════════════════════════════════════
        # Tab 1：StationXML / Dataless SEED
        # ════════════════════════════════════════════════════════════════
        tab_inv = QWidget()
        tv = QVBoxLayout(tab_inv)
        tv.setSpacing(8)
        tv.setContentsMargins(10, 10, 10, 10)

        inv_row = QHBoxLayout()
        self._inv_path_lbl = QLabel("未选择文件")
        self._inv_path_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;"
            f"border:1px solid {COLORS['border']};border-radius:4px;padding:3px 6px;"
        )
        self._inv_path_lbl.setWordWrap(False)
        self._inv_inventory = None

        browse_btn = QPushButton("浏览…")
        browse_btn.setFixedHeight(26)
        browse_btn.setFixedWidth(68)
        browse_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_header']};color:{COLORS['accent_blue']};"
            f"border:1px solid {COLORS['accent_blue']};border-radius:4px;"
            f"font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{background:{COLORS['accent_blue']}22;}}"
        )

        def _browse_inv():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "选择台站响应文件", self._last_dir,
                "响应文件 (*.xml *.XML *.seed *.SEED *.dataless *.resp *.RESP);;"
                "StationXML (*.xml *.XML);;"
                "Dataless SEED (*.seed *.SEED *.dataless);;"
                "所有文件 (*.*)"
            )
            if not path:
                return
            try:
                from obspy import read_inventory
                self._inv_inventory = read_inventory(path)
                self._inv_path_lbl.setText(os.path.basename(path))
                self._inv_path_lbl.setStyleSheet(
                    f"color:{COLORS['accent_green']};font-size:10px;"
                    f"font-family:Consolas;border:1px solid {COLORS['accent_green']};"
                    f"border-radius:4px;padding:3px 6px;"
                )
                n_ch = sum(len(ch_list)
                           for net in self._inv_inventory
                           for sta in net
                           for ch_list in [list(sta)])
                inv_info_lbl.setText(
                    f"✓  已读取  {len(self._inv_inventory)} 网络 · "
                    f"{sum(len(list(net)) for net in self._inv_inventory)} 台站 · "
                    f"{n_ch} 通道"
                )
                inv_info_lbl.setStyleSheet(
                    f"color:{COLORS['accent_green']};font-size:10px;"
                )
            except Exception as e:
                QMessageBox.warning(dlg, "读取响应文件失败", str(e))

        browse_btn.clicked.connect(_browse_inv)
        inv_row.addWidget(self._inv_path_lbl, 1)
        inv_row.addWidget(browse_btn)
        tv.addLayout(inv_row)

        inv_info_lbl = QLabel("请先选择台站响应文件（StationXML 或 Dataless SEED）")
        inv_info_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
        )
        tv.addWidget(inv_info_lbl)
        tv.addStretch()
        tabs.addTab(tab_inv, "📄  StationXML / Inventory")

        # ════════════════════════════════════════════════════════════════
        # Tab 2：PAZ（极零点）手动输入
        # ════════════════════════════════════════════════════════════════
        tab_paz = QWidget()
        pv = QVBoxLayout(tab_paz)
        pv.setSpacing(6)
        pv.setContentsMargins(10, 10, 10, 10)

        paz_note = QLabel(
            "每行一个复数，格式：实部 虚部（以空格分隔）。\n"
            "例如极点 -4.44+4.44j 填写为：  -4.44  4.44"
        )
        paz_note.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:5px 8px;"
        )
        paz_note.setWordWrap(True)
        pv.addWidget(paz_note)

        paz_form = QFormLayout()
        paz_form.setSpacing(6)

        paz_poles_edit = QTextEdit()
        paz_poles_edit.setFixedHeight(70)
        paz_poles_edit.setPlaceholderText("-4.44  4.44\n-4.44  -4.44")
        paz_poles_edit.setStyleSheet(
            f"background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:4px;"
            f"font-family:Consolas;font-size:11px;padding:4px;"
        )
        paz_form.addRow("极点 Poles：", paz_poles_edit)

        paz_zeros_edit = QTextEdit()
        paz_zeros_edit.setFixedHeight(55)
        paz_zeros_edit.setPlaceholderText("0  0\n0  0")
        paz_zeros_edit.setStyleSheet(
            f"background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:4px;"
            f"font-family:Consolas;font-size:11px;padding:4px;"
        )
        paz_form.addRow("零点 Zeros：", paz_zeros_edit)

        paz_gain_spin = QDoubleSpinBox()
        paz_gain_spin.setRange(1e-20, 1e20)
        paz_gain_spin.setValue(1.0)
        paz_gain_spin.setDecimals(6)
        paz_gain_spin.setToolTip("仪器总增益（sensitivity × gain_scale）")
        paz_form.addRow("仪器增益 Sensitivity：", paz_gain_spin)

        paz_seismometer_gain_spin = QDoubleSpinBox()
        paz_seismometer_gain_spin.setRange(1e-20, 1e20)
        paz_seismometer_gain_spin.setValue(1.0)
        paz_seismometer_gain_spin.setDecimals(6)
        paz_seismometer_gain_spin.setToolTip("地震计本体增益（seismometer gain）")
        paz_form.addRow("地震计增益 Seismometer Gain：", paz_seismometer_gain_spin)

        pv.addLayout(paz_form)
        tabs.addTab(tab_paz, "🔢  极零点 PAZ")

        root.addWidget(tabs)

        # ════════════════════════════════════════════════════════════════
        # 公共参数区
        # ════════════════════════════════════════════════════════════════
        common_box = QGroupBox("公共参数")
        common_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:10px;padding-top:6px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;subcontrol-position:top left;"
            f"padding:0 6px;font-weight:600;letter-spacing:1px;}}"
        )
        cf = QFormLayout(common_box)
        cf.setSpacing(7)

        # 输出物理量
        out_combo = QComboBox()
        out_combo.addItems([
            "速度  (VEL, m/s)",
            "位移  (DISP, m)",
            "加速度  (ACC, m/s²)",
        ])
        out_combo.setFixedHeight(26)
        out_combo.setToolTip("去响应后的输出物理量单位")
        cf.addRow("输出单位：", out_combo)

        # 水位（water level）
        wl_spin = QDoubleSpinBox()
        wl_spin.setRange(0, 200)
        wl_spin.setValue(60)
        wl_spin.setDecimals(0)
        wl_spin.setSuffix("  dB")
        wl_spin.setToolTip(
            "频谱域除法的水位阈值（dB）。防止仪器响应幅值接近零时出现除零。\n"
            "典型值：60 dB（默认）。值越小、去响应越激进，噪声也可能被放大。"
        )
        cf.addRow("水位 Water Level：", wl_spin)

        # 预滤波开关 + 四角频率
        from PyQt5.QtWidgets import QCheckBox
        pf_chk = QCheckBox("启用预滤波（Pre-filter）")
        pf_chk.setChecked(True)
        pf_chk.setToolTip(
            "在去响应前对信号做余弦高通预滤波，避免低频噪声被放大。\n"
            "对应 ObsPy remove_response() 的 pre_filt 参数。"
        )
        cf.addRow("", pf_chk)

        pf_widget = QWidget()
        pf_layout = QHBoxLayout(pf_widget)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setSpacing(6)

        def _pf_spin(val, lo=0.0001, hi=500.0, tip=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(4)
            s.setSuffix(" Hz")
            s.setFixedWidth(100)
            s.setFixedHeight(24)
            s.setToolTip(tip)
            return s

        pf_f1 = _pf_spin(0.001,  tip="f1：低频淡入起点（此频率以下全部衰减）")
        pf_f2 = _pf_spin(0.005,  tip="f2：低频淡入终点（从 f1 到 f2 余弦上升）")
        pf_f3 = _pf_spin(45.0,   tip="f3：高频淡出起点（从 f3 到 f4 余弦下降）")
        pf_f4 = _pf_spin(50.0,   tip="f4：高频淡出终点（此频率以上全部衰减）")

        for lbl, sp in [("f1:", pf_f1), ("f2:", pf_f2), ("f3:", pf_f3), ("f4:", pf_f4)]:
            pf_layout.addWidget(QLabel(lbl))
            pf_layout.addWidget(sp)
        pf_layout.addStretch()

        pf_chk.toggled.connect(pf_widget.setEnabled)
        cf.addRow("预滤波角频率：", pf_widget)

        root.addWidget(common_box)

        # 说明
        tip_lbl = QLabel(
            "⚠  建议在去仪器响应前先进行 Taper（5%）+ 去趋势，以避免边缘效应放大。"
        )
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet(
            f"color:{COLORS['accent_amber']};font-size:10px;"
            f"background:#FFFBEB;border:1px solid #FDE68A;"
            f"border-radius:4px;padding:5px 8px;"
        )
        root.addWidget(tip_lbl)

        # 按钮
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        # ── 收集参数 ─────────────────────────────────────────────────────
        method      = tabs.currentIndex()   # 0=inventory 1=PAZ
        output      = ['VEL', 'DISP', 'ACC'][out_combo.currentIndex()]
        water_level = wl_spin.value()
        pre_filt    = None
        if pf_chk.isChecked():
            pre_filt = (pf_f1.value(), pf_f2.value(), pf_f3.value(), pf_f4.value())

        if method == 0:
            if self._inv_inventory is None:
                QMessageBox.warning(self, "未选择响应文件", "请先加载 StationXML / Dataless SEED 文件。")
                return
            self._process_remove_response_inventory(
                self._inv_inventory, output, water_level, pre_filt)
        else:
            # 解析 PAZ 文本
            def _parse_complex_lines(text):
                result = []
                for line in text.strip().splitlines():
                    parts = line.split()
                    if len(parts) == 0:
                        continue
                    if len(parts) == 1:
                        result.append(complex(float(parts[0]), 0))
                    elif len(parts) >= 2:
                        result.append(complex(float(parts[0]), float(parts[1])))
                return result

            try:
                poles = _parse_complex_lines(paz_poles_edit.toPlainText())
                zeros = _parse_complex_lines(paz_zeros_edit.toPlainText())
                sensitivity = paz_gain_spin.value()
                seismometer_gain = paz_seismometer_gain_spin.value()
            except ValueError as e:
                QMessageBox.warning(self, "PAZ 解析失败",
                                    f"极零点格式错误，请检查输入：\n{e}")
                return
            self._process_remove_response_paz(
                poles, zeros, sensitivity, seismometer_gain,
                output, water_level, pre_filt)

    def _process_remove_response_inventory(self, inventory, output,
                                            water_level, pre_filt):
        """使用 Inventory（StationXML）去仪器响应（仅作用于选中道）"""
        idxs = self._selected_stream_indices()
        try:
            for i in idxs:
                self.stream[i].remove_response(
                    inventory=inventory,
                    output=output,
                    water_level=water_level,
                    pre_filt=pre_filt,
                )
            self._stream_unit = output
            unit_map = {'VEL': 'm/s', 'DISP': 'm', 'ACC': 'm/s²'}
            desc = f"去响应(XML,{output},{unit_map[output]})"
            self._add_proc_step(desc, color='#7C3AED')
            self._set_status(
                f"✓ 去仪器响应完成（StationXML / Inventory）  →  输出：{output} [{unit_map[output]}]"
            )
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "去仪器响应失败",
                                 f"处理时发生错误：\n\n{str(e)}\n\n"
                                 "请确认：\n"
                                 "① StationXML 文件包含当前波形的台站和通道信息\n"
                                 "② 台站 / 通道 / 网络代码与数据头段匹配\n"
                                 "③ 时间范围在仪器响应有效期内")

    def _process_remove_response_paz(self, poles, zeros, sensitivity,
                                      seismometer_gain, output,
                                      water_level, pre_filt):
        """使用 PAZ（极零点）去仪器响应"""
        try:
            from obspy.signal.invsim import simulate_seismometer
            import numpy as np

            # 构建 ObsPy PAZ 字典
            paz_remove = {
                'poles':        poles,
                'zeros':        zeros,
                'gain':         seismometer_gain,
                'sensitivity':  sensitivity,
            }
            # PAZ 模拟目标为平坦响应（位移 → 速度 → 加速度 差分处理不同）
            paz_simulate = None
            idxs = self._selected_stream_indices()
            for i in idxs:
                self.stream[i].simulate(paz_remove=paz_remove,
                            paz_simulate=paz_simulate,
                            water_level=water_level,
                            pre_filt=pre_filt)
                if output == 'DISP':
                    self.stream[i].integrate()
                elif output == 'ACC':
                    self.stream[i].differentiate()
            self._stream_unit = output
            unit_map = {'VEL': 'm/s', 'DISP': 'm', 'ACC': 'm/s²'}
            desc = (f"去响应(PAZ,"
                    f"p={len(poles)},z={len(zeros)},"
                    f"{output})")
            self._add_proc_step(desc, color='#7C3AED')
            self._set_status(
                f"✓ 去仪器响应完成（PAZ）  →  极点 {len(poles)} 个 / 零点 {len(zeros)} 个"
            )
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "PAZ 去仪器响应失败",
                                 f"处理时发生错误：\n\n{str(e)}")

    # ── 历史记录 chip 管理 ────────────────────────────────────────────────────
    def _add_proc_step(self, desc: str, color: str = None):
        """添加一步处理记录，并在 UI 中追加 chip 标签"""
        if color is None:
            color = COLORS['accent_green']
        self._proc_history.append(desc)

        # 首次添加时移除占位文字
        if len(self._proc_history) == 1:
            self._hist_empty_lbl.hide()

        step_no = len(self._proc_history)
        chip = QLabel(f" {step_no}. {desc} ")
        chip.setStyleSheet(
            f"background:{color}22; color:{color};"
            f"border:1px solid {color}88;"
            f"border-radius:10px; font-size:10px; font-weight:600;"
            f"padding:1px 4px;"
        )
        chip.setFixedHeight(22)
        # 插入到 stretch 之前
        idx = self._hist_chip_layout.count() - 1
        self._hist_chip_layout.insertWidget(idx, chip)

    def _clear_proc_history_ui(self):
        """清空所有历史 chip，保留 index 0 的占位标签和末尾的 stretch。
        布局固定结构：[empty_lbl(0), chip1(1), chip2(2), ..., stretch(last)]
        只能从 index 1 开始取，绝不动 index 0 的 empty_lbl。
        """
        while self._hist_chip_layout.count() > 2:
            # 始终取 index 1（chips 紧跟 empty_lbl 之后，stretch 在末尾）
            item = self._hist_chip_layout.takeAt(1)
            if item and item.widget():
                item.widget().deleteLater()
        self._hist_empty_lbl.show()

    def _reset_to_original(self):
        """将 stream 还原到最初加载的原始数据"""
        if not self._orig_stream:
            self._set_status("尚未加载任何文件，无法重置")
            return
        self.stream = self._orig_stream.copy()
        self._stream_unit = 'counts'
        self._proc_history.clear()
        self._clear_proc_history_ui()
        self._set_status("✓ 已重置为原始波形，所有预处理步骤已撤销")
        self._replot_current()

    def _build_pick_toolbar(self):
        """构建：震相拾取面板 """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        lbl = QLabel("拾取模式")
        lbl.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:10px; font-weight:700; letter-spacing:1px;")
        layout.addWidget(lbl)

        self.mode_pan_btn = QPushButton("✥ 平移")
        self.mode_pan_btn.setCheckable(True)
        self.mode_pan_btn.setChecked(True)
        self.mode_pan_btn.clicked.connect(lambda: self._set_pick_mode('pan'))

        self.mode_p_btn = QPushButton("P 拾取")
        self.mode_p_btn.setCheckable(True)
        self.mode_p_btn.clicked.connect(lambda: self._set_pick_mode('P'))

        self.mode_s_btn = QPushButton("S 拾取")
        self.mode_s_btn.setCheckable(True)
        self.mode_s_btn.clicked.connect(lambda: self._set_pick_mode('S'))

        self.mode_pan_btn.setStyleSheet(self._mode_btn_style('#4A90D9', True))
        self.mode_p_btn.setStyleSheet(self._mode_btn_style('#FF3333', False))
        self.mode_s_btn.setStyleSheet(self._mode_btn_style('#33AAFF', False))

        for btn in [self.mode_pan_btn, self.mode_p_btn, self.mode_s_btn]:
            btn.setFixedSize(78, 26)
            layout.addWidget(btn)

        layout.addWidget(self._make_vsep())

        clr_p_btn = QPushButton("清除 P")
        clr_p_btn.clicked.connect(lambda: self._clear_pick('P'))
        clr_s_btn = QPushButton("清除 S")
        clr_s_btn.clicked.connect(lambda: self._clear_pick('S'))
        clr_all_btn = QPushButton("全部清除")
        clr_all_btn.clicked.connect(lambda: self._clear_pick(None))

        for btn in [clr_p_btn, clr_s_btn, clr_all_btn]:
            btn.setFixedHeight(26)
            layout.addWidget(btn)

        layout.addWidget(self._make_vsep())

        export_btn = QPushButton("📋 导出拾取")
        export_btn.setFixedHeight(26)
        export_btn.clicked.connect(self._export_picks)
        layout.addWidget(export_btn)

        layout.addWidget(self._make_vsep())

        import_pick_btn = QPushButton("📂 读取震相")
        import_pick_btn.setFixedHeight(26)
        import_pick_btn.setToolTip(
            "从 CSV / JSON / QuakeML / txt 文件读取震相数据，\n"
            "自动匹配台站并标注到对应波形道上")
        import_pick_btn.clicked.connect(self._show_import_picks_dialog)
        layout.addWidget(import_pick_btn)

        clr_import_btn = QPushButton("✕ 清除导入")
        clr_import_btn.setFixedHeight(26)
        clr_import_btn.setToolTip("清除所有从外部文件读入的震相标注")
        clr_import_btn.clicked.connect(self._clear_imported_picks)
        layout.addWidget(clr_import_btn)

        layout.addStretch()

        self.sp_label = QLabel("S-P = --")
        self.sp_label.setStyleSheet(f"color:{COLORS['accent_amber']}; font-size:11px; font-family:Consolas; font-weight:700;")
        layout.addWidget(self.sp_label)

        return widget   

    def _mode_btn_style(self, color: str, checked: bool) -> str:
        bg = f"{color}40" if checked else COLORS['bg_header']# 40 表示 25% 不透明度
        border = color if checked else COLORS['border_bright']
        text   = color if checked else COLORS['text_secondary']
        border_width = '2px' if checked else '1.5px'
        return (
            f"QPushButton {{"
            f"  background:{bg}; color:{text};"
            f"  border:{border_width} solid {border}; border-radius:5px;"
            f"  font-size:12px; font-weight:{'700' if checked else '500'}; letter-spacing:0.5px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background:{color}66; border-color:{color}; color:{color};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background:{color}80; border-color:{color}; color:{color};"
            f"}}"
        )

    def _make_vsep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background:{COLORS['border_bright']};")
        return sep

    # ── 拾取模式控制 ──────────────────────────────────────────────────────────
    def _set_pick_mode(self, mode: str):
        """切换拾取模式并更新按钮高亮状态"""
        self.canvas.set_pick_mode(mode)
        colors = {'pan': '#4A90D9', 'P': '#FFD700', 'S': '#FF4D6A'}
        btns   = {'pan': self.mode_pan_btn,
                  'P':   self.mode_p_btn,
                  'S':   self.mode_s_btn}
        for m, btn in btns.items():
            btn.setChecked(m == mode)
            btn.setStyleSheet(self._mode_btn_style(colors[m], m == mode))

        hints = {
            'pan': "平移模式  —  左键拖拽，滚轮缩放",
            'P':   "P波拾取模式  —  点击波形任意位置标记P波到时，右键撤销",
            'S':   "S波拾取模式  —  点击波形任意位置标记S波到时，右键撤销",
        }
        self._set_status(hints[mode])

    def keyPressEvent(self, event):
        """快捷键：P/S 切换拾取模式，Esc 返回平移"""
        key = event.key()
        if key == Qt.Key_P:
            self._set_pick_mode('P')
        elif key == Qt.Key_S:
            self._set_pick_mode('S')
        elif key == Qt.Key_Escape:
            self._set_pick_mode('pan')
        else:
            super().keyPressEvent(event)

    def _on_pick_added(self, phase: str, t_sec: float,
                       abs_time: str, amp: float, ch: str):

        # 更新 S-P 时差
        picks = self.canvas.get_picks()
        if 'P' in picks and 'S' in picks:
            tp = picks['P'][0]
            ts = picks['S'][0]
            sp = ts - tp
            if sp >= 0:
                dist_km = sp * 8.0
                self.sp_label.setText(f"S-P = {sp:.3f}s")
            else:
                self.sp_label.setText("S-P = 异常")
        else:
            self.sp_label.setText("S-P = --")

    def _clear_pick(self, phase):
        """清除拾取并重置面板"""
        self.canvas.clear_picks(phase)
        picks = self.canvas.get_picks()
        if 'P' in picks and 'S' in picks:
            tp = picks['P'][0]
            ts = picks['S'][0]
            sp = ts - tp
            if sp >= 0:
                self.sp_label.setText(f"S-P = {sp:.3f}s")
            else:
                self.sp_label.setText("S-P = 异常")
        else:
            self.sp_label.setText("S-P = --")        

    def _export_picks(self):
        """导出拾取结果到 CSV 或 TXT 文件"""
        picks = self.canvas.get_picks()
        if not picks:
            QMessageBox.information(self, "无拾取结果", "尚未进行任何拾取操作，无法导出。")
            return

        # 提取台网和台站信息 (从当前渲染的第一个数据流中获取)
        network = "Unknown"
        station = "Unknown"
        if self.stream and len(self.stream) > 0:
            network = self.stream[0].stats.network
            station = self.stream[0].stats.station

        # 弹出保存文件对话框，增加对 CSV 的支持
        path, _ = QFileDialog.getSaveFileName(
            self, "导出拾取结果", f"{network}_{station}_picks.csv",
            "CSV 数据文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if not path:
            return  # 用户取消了保存

        is_csv = path.lower().endswith('.csv')
        lines = []

        if is_csv:
            # CSV 格式：带表头，适合直接导入 Excel 或 Python/Pandas
            lines.append("Network,Station,Phase,UTC_Time,Amplitude")
            for phase, (t_sec, abs_t, amp) in sorted(picks.items()):
                lines.append(f"{network},{station},{phase},{abs_t},{amp:.6g}")
        else:
            # TXT 格式：适合人类阅读
            lines.append(f"台网 (Network): {network}")
            lines.append(f"台站 (Station): {station}")
            lines.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("-" * 50)
            lines.append(f"{'震相':<8} | {'绝对到时 (UTC)':<25} | {'振幅':<15}")
            lines.append("-" * 50)
            for phase, (t_sec, abs_t, amp) in sorted(picks.items()):
                lines.append(f"{phase:<10} | {abs_t:<25} | {amp:.6g}")
            
            # 如果同时拾取了P和S，附加计算的时差信息
            if 'P' in picks and 'S' in picks:
                sp_diff = picks['S'][0] - picks['P'][0]
                lines.append("-" * 50)
                lines.append(f"S-P 时差: {sp_diff:.4f} 秒")
                lines.append(f"估算震中距: 约 {sp_diff * 8.0:.1f} km (基于 8km/s 经验波速)")

        # 写入文件
        text_to_save = '\n'.join(lines)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text_to_save)
            self._set_status(f"拾取结果已成功导出：{os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入文件时发生错误：\n{str(e)}")

    # ── 导入震相（从外部文件读入并标注）────────────────────────────────────────
    def _clear_imported_picks(self):
        self.canvas.clear_imported_picks()
        self._set_status("已清除所有导入的震相标注")

    def _show_import_picks_dialog(self):
        """选择文件 → 解析 → 预览 → 匹配台站 → 标注到画布"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        # ── 第一步：选择文件 ──────────────────────────────────────────────
        path, _ = QFileDialog.getOpenFileName(
            self, "读取震相数据文件",
            self._last_dir or "",
            "所有支持格式 (*.csv *.txt *.json *.xml *.quakeml *.hyp);;"
            "CSV 文件 (*.csv);;"
            "文本文件 (*.txt *.hyp);;"
            "JSON 文件 (*.json);;"
            "QuakeML 文件 (*.xml *.quakeml);;"
            "所有文件 (*.*)"
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)

        # ── 第二步：解析 ──────────────────────────────────────────────────
        try:
            parsed = self._parse_picks_file(path)
        except Exception as e:
            QMessageBox.critical(self, "文件解析失败",
                                 f"无法解析文件：\n{path}\n\n{e}")
            return

        if not parsed:
            QMessageBox.warning(self, "无有效数据",
                                "文件中未找到可用的震相记录，请检查文件格式。")
            return

        # ── 第三步：预览 + 匹配选项对话框 ────────────────────────────────
        from PyQt5.QtWidgets import (QTableWidget, QTableWidgetItem,
                                     QHeaderView, QAbstractItemView,
                                     QListWidget, QListWidgetItem,
                                     QSplitter as _Splitter)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"震相导入预览  —  {os.path.basename(path)}")
        dlg.setMinimumSize(680, 520)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(8)
        root.setContentsMargins(14, 12, 14, 12)

        # 统计信息
        phases_found = sorted({r['phase'] for r in parsed})
        stas_found   = sorted({r['station'] for r in parsed})
        stat_lbl = QLabel(
            f"共解析到 {len(parsed)} 条震相记录  ·  "
            f"{len(stas_found)} 个台站  ·  "
            f"震相类型：{', '.join(phases_found)}"
        )
        stat_lbl.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:10px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:5px 8px;"
        )
        root.addWidget(stat_lbl)

        # 预览表格
        tbl = QTableWidget(len(parsed), 5)
        tbl.setHorizontalHeaderLabels(["台网", "台站", "通道", "震相", "到时 (UTC)"])
        tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.setFixedHeight(200)
        tbl.setStyleSheet(
            f"QTableWidget{{background:{COLORS['bg_card']};"
            f"alternate-background-color:{COLORS['bg_deep']};"
            f"font-family:Consolas;font-size:10px;"
            f"border:1px solid {COLORS['border_bright']};border-radius:4px;}}"
            f"QHeaderView::section{{background:{COLORS['bg_header']};"
            f"color:{COLORS['text_secondary']};padding:4px;font-size:10px;"
            f"border:none;border-bottom:1px solid {COLORS['border_bright']};}}"
        )
        for row, r in enumerate(parsed):
            tbl.setItem(row, 0, QTableWidgetItem(r.get('network', '')))
            tbl.setItem(row, 1, QTableWidgetItem(r.get('station', '')))
            tbl.setItem(row, 2, QTableWidgetItem(r.get('channel', '')))
            ph_item = QTableWidgetItem(r.get('phase', ''))
            from PyQt5.QtGui import QColor as _QColor
            from canvas_seismic import SeismicCanvas as _SC
            col = _SC.IMPORTED_PHASE_COLORS.get(
                r.get('phase', ''), _SC.IMPORTED_PHASE_DEFAULT_COLOR)
            ph_item.setForeground(_QColor(col))
            tbl.setItem(row, 3, ph_item)
            tbl.setItem(row, 4, QTableWidgetItem(str(r.get('time', ''))))
        root.addWidget(tbl)

        # 匹配选项
        match_box = QGroupBox("台站匹配选项")
        match_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:8px;padding-top:4px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 5px;"
            f"font-weight:600;}}"
        )
        mf = QFormLayout(match_box)
        mf.setSpacing(7)
        mf.setContentsMargins(10, 10, 10, 10)

        match_combo = QComboBox()
        match_combo.setFixedHeight(24)
        match_combo.addItem("仅按台站名匹配（STA）",            "sta")
        match_combo.addItem("按台网 + 台站匹配（NET.STA）",     "net_sta")
        match_combo.addItem("按台网 + 台站 + 通道匹配（NET.STA.CHA）", "net_sta_cha")
        mf.addRow("匹配方式：", match_combo)

        # 当前数据流中的台站列表
        stream_stas = sorted({tr.stats.station for tr in self.stream})
        overlap = sorted(set(stas_found) & set(stream_stas))
        no_match = sorted(set(stas_found) - set(stream_stas))

        match_info = QLabel(
            f"数据流台站：{', '.join(stream_stas) or '—'}\n"
            f"文件台站中可匹配：{', '.join(overlap) or '（无）'}  "
            f"{'  ⚠ 无法匹配：' + ', '.join(no_match) if no_match else ''}"
        )
        match_info.setWordWrap(True)
        match_info.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"font-family:Consolas;"
        )
        mf.addRow("匹配预览：", match_info)

        # 如果文件中的震相有时区/时间格式，允许用户指定时间基准
        time_base_combo = QComboBox()
        time_base_combo.setFixedHeight(24)
        time_base_combo.addItem("绝对 UTC 时间（文件中包含完整时间字符串）", "utc")
        time_base_combo.addItem("相对秒（从数据流第一道起始时刻偏移）",       "rel")
        mf.addRow("时间格式：", time_base_combo)

        root.addWidget(match_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        match_mode = match_combo.currentData()
        time_mode  = time_base_combo.currentData()

        # ── 第四步：匹配到 stream trace，计算相对时间，调用 canvas ──────
        picks_for_canvas = self._match_picks_to_stream(
            parsed, match_mode=match_mode, time_mode=time_mode)

        if not picks_for_canvas:
            QMessageBox.warning(
                self, "匹配失败",
                "文件中的台站与当前数据流中的台站没有匹配项。\n"
                "请检查：\n① 台站名拼写是否一致\n② 是否选择了正确的匹配方式")
            return

        self.canvas.add_imported_picks(picks_for_canvas)
        n = len(picks_for_canvas)
        self._set_status(
            f"✓ 已导入 {n} 条震相标注  |  "
            f"来源：{os.path.basename(path)}")

    # ── 震相文件解析器 ────────────────────────────────────────────────────────
    def _parse_picks_file(self, path: str) -> list:
        """
        自动识别文件格式并解析，返回：
          [ {network, station, channel, phase, time (UTCDateTime or float)}, ... ]
        支持：CSV / JSON / QuakeML / NonLinLoc .hyp / 纯文本
        """
        import os
        ext = os.path.splitext(path)[1].lower()

        # ── QuakeML / ObsPy event format ──────────────────────────────────
        if ext in ('.xml', '.quakeml'):
            return self._parse_quakeml_picks(path)

        # ── JSON ──────────────────────────────────────────────────────────
        if ext == '.json':
            return self._parse_json_picks(path)

        # ── NonLinLoc .hyp ────────────────────────────────────────────────
        if ext == '.hyp':
            return self._parse_nll_picks(path)

        # ── CSV / TXT：自动探测列名 ───────────────────────────────────────
        return self._parse_csv_picks(path)

    def _parse_csv_picks(self, path: str) -> list:
        """解析 CSV 或以空白/制表符分隔的文本文件"""
        import csv, re
        from obspy import UTCDateTime

        results = []
        # 尝试多种分隔符
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            sample = f.read(4096)
        sep = ',' if sample.count(',') > sample.count('\t') else '\t'
        if sample.count(sep) < 3:
            sep = None   # 让 csv.Sniffer 自动判断

        with open(path, encoding='utf-8-sig', errors='replace') as f:
            try:
                dialect = csv.Sniffer().sniff(sample) if sep is None else None
                reader  = (csv.DictReader(f, dialect=dialect)
                           if dialect else csv.DictReader(f, delimiter=sep))
            except Exception:
                reader = csv.DictReader(f, delimiter=',')

            # 列名规范化（大小写不敏感）
            COL_MAP = {
                'network': ['network', 'net', 'nw'],
                'station': ['station', 'sta', 'stat', 'staid'],
                'channel': ['channel', 'cha', 'chan', 'comp'],
                'phase':   ['phase', 'phase_type', 'phase_hint', 'pha',
                            'onset', 'wave'],
                'time':    ['time', 'arrival_time', 'pick_time', 'onset_time',
                            'utctime', 'datetime', 'abs_time', 't'],
            }

            def find_col(row_keys, aliases):
                lk = {k.lower().strip(): k for k in row_keys}
                for alias in aliases:
                    if alias in lk:
                        return lk[alias]
                return None

            col_cache = {}
            for row in reader:
                if not col_cache:
                    col_cache = {k: find_col(row.keys(), v)
                                 for k, v in COL_MAP.items()}

                def get(k):
                    c = col_cache.get(k)
                    return row.get(c, '').strip() if c else ''

                sta   = get('station')
                phase = get('phase')
                tstr  = get('time')
                if not sta or not phase or not tstr:
                    continue

                # 解析时间：尝试 UTCDateTime，失败则尝试 float
                try:
                    t = UTCDateTime(tstr)
                except Exception:
                    try:
                        t = float(tstr)
                    except Exception:
                        continue

                results.append({
                    'network': get('network'),
                    'station': sta,
                    'channel': get('channel'),
                    'phase':   phase,
                    'time':    t,
                })
        return results

    def _parse_json_picks(self, path: str) -> list:
        """解析 JSON 格式震相文件（列表或含 picks 键的字典）"""
        import json
        from obspy import UTCDateTime
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            records = data.get('picks', data.get('arrivals', data.get('data', [])))
        else:
            records = data

        results = []
        for r in records:
            if not isinstance(r, dict):
                continue
            sta   = (r.get('station') or r.get('sta') or '').strip()
            phase = (r.get('phase') or r.get('phase_type') or
                     r.get('phase_hint') or '').strip()
            traw  = r.get('time') or r.get('arrival_time') or r.get('pick_time') or ''
            if not sta or not phase or not traw:
                continue
            try:
                t = UTCDateTime(str(traw))
            except Exception:
                try:
                    t = float(traw)
                except Exception:
                    continue
            results.append({
                'network': (r.get('network') or r.get('net') or '').strip(),
                'station': sta,
                'channel': (r.get('channel') or r.get('cha') or '').strip(),
                'phase':   phase,
                'time':    t,
            })
        return results

    def _parse_quakeml_picks(self, path: str) -> list:
        """利用 ObsPy 解析 QuakeML / SC3ML 格式"""
        try:
            from obspy import read_events
        except ImportError:
            raise RuntimeError("需要 ObsPy 的 read_events 功能来解析 QuakeML 文件")

        cat = read_events(path)
        results = []
        for ev in cat:
            for pick in ev.picks:
                wid   = pick.waveform_id
                phase = ''
                # 尝试从 arrivals 找对应的 phase_hint
                for orig in ev.origins:
                    for arr in orig.arrivals:
                        if arr.pick_id == pick.resource_id:
                            phase = arr.phase or ''
                            break
                    if phase:
                        break
                if not phase:
                    phase = pick.phase_hint or '?'
                results.append({
                    'network': wid.network_code or '',
                    'station': wid.station_code or '',
                    'channel': wid.channel_code or '',
                    'phase':   phase,
                    'time':    pick.time,
                })
        return results

    def _parse_nll_picks(self, path: str) -> list:
        """解析 NonLinLoc .hyp 格式中的 PHASE 数据块"""
        from obspy import UTCDateTime
        results = []
        in_phase = False
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('PHASE'):
                    in_phase = True
                    continue
                if stripped.startswith('END_PHASE') or (in_phase and stripped == ''):
                    in_phase = False
                    continue
                if not in_phase:
                    continue
                parts = stripped.split()
                if len(parts) < 7:
                    continue
                try:
                    sta   = parts[0]
                    phase = parts[4]
                    # NLL 格式：日期 YYYYMMDD，时间 HHMMSS.mmm
                    date_s  = parts[6]          # YYYYMMDD
                    time_s  = parts[7]          # HHMMSS.mmm
                    yr, mo, dy = date_s[:4], date_s[4:6], date_s[6:8]
                    hh, mm     = time_s[:2], time_s[2:4]
                    ss         = time_s[4:]
                    tstr = f"{yr}-{mo}-{dy}T{hh}:{mm}:{ss}"
                    t = UTCDateTime(tstr)
                    results.append({
                        'network': '',
                        'station': sta,
                        'channel': '',
                        'phase':   phase,
                        'time':    t,
                    })
                except Exception:
                    continue
        return results

    # ── 震相匹配 ──────────────────────────────────────────────────────────────
    def _match_picks_to_stream(self, parsed: list, match_mode: str = 'sta',
                               time_mode: str = 'utc') -> list:
        """
        将解析到的震相记录匹配到当前 stream 的 trace 索引，
        计算相对到时（秒），返回 [(trace_idx, phase, t_rel), ...]。
        """
        from obspy import UTCDateTime

        results = []
        for r in parsed:
            phase = r.get('phase', '')
            t_raw = r.get('time')
            if not phase or t_raw is None:
                continue

            # 找到匹配的 trace 索引（可能多道，如多分量）
            matched_indices = []
            for i, tr in enumerate(self.stream):
                s = tr.stats
                if match_mode == 'sta':
                    ok = (s.station.upper() == r.get('station', '').upper())
                elif match_mode == 'net_sta':
                    ok = (s.station.upper() == r.get('station', '').upper() and
                          (not r.get('network') or
                           s.network.upper() == r['network'].upper()))
                else:   # net_sta_cha
                    ok = (s.station.upper() == r.get('station', '').upper() and
                          (not r.get('network') or
                           s.network.upper() == r['network'].upper()) and
                          (not r.get('channel') or
                           s.channel.upper() == r['channel'].upper()))
                if ok:
                    matched_indices.append(i)

            if not matched_indices:
                continue

            # 计算相对到时
            for idx in matched_indices:
                base_t = self.stream[idx].stats.starttime
                if time_mode == 'rel' or isinstance(t_raw, float):
                    t_rel = float(t_raw)
                else:
                    try:
                        t_abs = UTCDateTime(t_raw) if not isinstance(
                            t_raw, UTCDateTime) else t_raw
                        t_rel = float(t_abs - base_t)
                    except Exception:
                        continue

                # 过滤超出数据范围的到时
                dur = float(self.stream[idx].stats.endtime - base_t)
                if t_rel < -1.0 or t_rel > dur + 1.0:
                    continue

                results.append((idx, phase, t_rel))

        return results

    def _show_prep_quick_menu(self, anchor_widget=None):
        """Show a quick preprocessing menu anchored to the toolbar button."""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return
        menu = QMenu(self)
        menu.addAction("去均值 (Demean)", self._process_demean)
        menu.addAction("去趋势 (Detrend)", self._process_detrend)
        menu.addSeparator()
        menu.addAction("归一化 → Max", lambda: self._process_normalize('max'))
        menu.addAction("归一化 → Peak-Peak", lambda: self._process_normalize('peak_peak'))
        menu.addAction("归一化 → RMS", lambda: self._process_normalize('rms'))
        menu.addSeparator()
        menu.addAction("滤波设置…", self._show_filter_dialog)
        menu.addAction("Taper 设置…", self._show_taper_dialog)
        menu.addAction("去仪器响应…", self._show_remove_response_dialog)
        menu.addAction("截窗…", self._show_trim_menu)
        menu.addAction("分量旋转…", self._show_rotate_dialog)
        menu.addAction("重采样…", self._show_resample_dialog)
        menu.addSeparator()
        menu.addAction("重置为原始波形", self._reset_to_original)

        btn = anchor_widget or self.sender()
        pos = btn.mapToGlobal(btn.rect().bottomLeft()) if btn else self.cursor().pos()
        menu.exec_(pos)

    def _show_pick_quick_menu(self, anchor_widget=None):
        """Show a quick phase-picking menu anchored to the toolbar button."""
        menu = QMenu(self)
        menu.addAction("P 波拾取模式", lambda: self._set_pick_mode('P'))
        menu.addAction("S 波拾取模式", lambda: self._set_pick_mode('S'))
        menu.addAction("平移模式", lambda: self._set_pick_mode('pan'))
        menu.addSeparator()
        menu.addAction("清除 P 波拾取", lambda: self._clear_pick('P'))
        menu.addAction("清除 S 波拾取", lambda: self._clear_pick('S'))
        menu.addAction("清除全部拾取", lambda: self._clear_pick(None))
        menu.addSeparator()
        menu.addAction("导出拾取结果…", self._export_picks)
        menu.addAction("读取震相文件…", self._show_import_picks_dialog)
        menu.addAction("清除导入震相", self._clear_imported_picks)

        btn = anchor_widget or self.sender()
        pos = btn.mapToGlobal(btn.rect().bottomLeft()) if btn else self.cursor().pos()
        menu.exec_(pos)

    def _show_toolbar_help(self):
        """Open a compact help dialog from the toolbar."""
        self._show_help()

    # ── 菜单栏 ─────────────────────────────────────────────────────────────
    def _setup_menubar(self):
        mb = self.menuBar()

        # 文件菜单
        file_menu = mb.addMenu("文件(&F)")
        self._add_action(file_menu, "打开文件(&O)...", self.open_file,
                         shortcut="Ctrl+O", tip="打开一个或多个地震数据文件")
        self._add_action(file_menu, "打开文件夹(&D)...", self.open_folder,
                         shortcut="Ctrl+Shift+O", tip="打开文件夹，自动扫描其中所有地震文件")
        self._add_action(file_menu, "重新加载(&R)", self.reload_file,
                         shortcut="Ctrl+Shift+R", tip="重新加载当前文件")
        file_menu.addSeparator()
        self._add_action(file_menu, "保存波形文件(&S)...", self.save_waveform,
                         shortcut="Ctrl+S", tip="将选中通道保存为地震波形文件")
        self._add_action(file_menu, "导出波形图(&E)...", self.export_figure,
                         shortcut="Ctrl+E", tip="将当前波形图导出为图片")
        file_menu.addSeparator()
        self._add_action(file_menu, "退出(&X)", self.close, shortcut="Alt+F4")

        # 视图菜单
        view_menu = mb.addMenu("视图(&V)")
        self._add_action(view_menu, "放大时间轴", self._zoom_in,  shortcut="Ctrl++")
        self._add_action(view_menu, "缩小时间轴", self._zoom_out, shortcut="Ctrl+-")
        self._add_action(view_menu, "重置视图",   self._reset_view, shortcut="Ctrl+R")
        view_menu.addSeparator()
        self._add_action(view_menu, "绘制全部通道", self._plot_all)
        self._add_action(view_menu, "绘制选中通道", self._plot_selected)
        view_menu.addSeparator()
        self._add_action(view_menu, "按震中距排列…", self._show_distance_sort_dialog)

        # 预处理菜单
        proc_menu = mb.addMenu("预处理(&P)")
        self._add_action(proc_menu, "去均值", self._process_demean)
        self._add_action(proc_menu, "去趋势", self._process_detrend)
        norm_menu = proc_menu.addMenu("归一化")
        self._add_action(norm_menu, "最大值归一化", lambda: self._process_normalize('max'))
        self._add_action(norm_menu, "峰峰值归一化", lambda: self._process_normalize('peak_peak'))
        self._add_action(norm_menu, "RMS 归一化", lambda: self._process_normalize('rms'))
        proc_menu.addSeparator()
        self._add_action(proc_menu, "滤波设置…", self._show_filter_dialog)
        self._add_action(proc_menu, "Taper 设置…", self._show_taper_dialog)
        self._add_action(proc_menu, "去仪器响应…", self._show_remove_response_dialog)
        self._add_action(proc_menu, "截窗…", self._show_trim_menu)
        self._add_action(proc_menu, "分量旋转…", self._show_rotate_dialog)
        self._add_action(proc_menu, "重采样…", self._show_resample_dialog)
        proc_menu.addSeparator()
        self._add_action(proc_menu, "恢复原始波形", self._reset_to_original)

        # 分析菜单
        analysis_menu = mb.addMenu("分析(&A)")
        self._add_action(analysis_menu, "振幅频谱分析…", self._open_spectrum_window)
        self._add_action(analysis_menu, "功率谱密度分析 (PSD)…", self._open_psd_window)
        self._add_action(analysis_menu, "多道互相关分析…", self._open_xcorr_window)

        # 拾取菜单
        pick_menu = mb.addMenu("拾取(&K)")
        self._add_action(pick_menu, "P波拾取模式(&P)",
                         lambda: self._set_pick_mode('P'), shortcut="P",
                         tip="切换到P波拾取模式，点击波形放置P波标注")
        self._add_action(pick_menu, "S波拾取模式(&S)",
                         lambda: self._set_pick_mode('S'), shortcut="S",
                         tip="切换到S波拾取模式，点击波形放置S波标注")
        self._add_action(pick_menu, "平移模式(&N)",
                         lambda: self._set_pick_mode('pan'), shortcut="Escape",
                         tip="返回平移/缩放模式")
        pick_menu.addSeparator()
        self._add_action(pick_menu, "清除P波拾取", lambda: self._clear_pick('P'))
        self._add_action(pick_menu, "清除S波拾取", lambda: self._clear_pick('S'))
        self._add_action(pick_menu, "清除全部拾取(&C)",
                         lambda: self._clear_pick(None), shortcut="Ctrl+K")
        pick_menu.addSeparator()
        self._add_action(pick_menu, "导出拾取结果(&X)...",
                         self._export_picks, shortcut="Ctrl+Shift+E")

        # 帮助菜单
        help_menu = mb.addMenu("帮助(&H)")
        self._add_action(help_menu, "使用说明", self._show_help)
        self._add_action(help_menu, "预处理说明", self._show_preprocess_help)
        self._add_action(help_menu, "拾取说明", self._show_pick_help)
        self._add_action(help_menu, "频谱分析说明", self._show_spectrum_help)
        help_menu.addSeparator()
        self._add_action(help_menu, "关于 SeismoView", self._show_about)

    def _add_action(self, menu, text, slot, shortcut=None, tip=None):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ── 工具栏 ─────────────────────────────────────────────────────────────
    def _setup_toolbar(self):
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        def make_btn(text, slot, tip):
            btn = QPushButton(text)
            btn.setFixedHeight(30)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            return btn

        # ── 打开文件 / 文件夹 下拉按钮 ────────────────────────────────────
        open_btn = QPushButton("📂  打开  ▾")
        open_btn.setFixedHeight(30)
        open_btn.setObjectName("accent_btn")
        open_btn.setToolTip("打开地震数据文件或文件夹")
        open_menu = QMenu(open_btn)
        open_menu.setStyleSheet(
            f"QMenu{{background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;padding:4px 0;}}"
            f"QMenu::item{{padding:7px 20px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}"
            f"QMenu::separator{{height:1px;background:{COLORS['border']};margin:3px 8px;}}"
        )
        act_open_files = QAction("📄  打开文件（可多选）", self)
        act_open_files.setShortcut("Ctrl+O")
        act_open_files.triggered.connect(self.open_file)
        open_menu.addAction(act_open_files)

        act_open_folder = QAction("📁  打开文件夹（自动扫描）", self)
        act_open_folder.setShortcut("Ctrl+Shift+O")
        act_open_folder.triggered.connect(self.open_folder)
        open_menu.addAction(act_open_folder)

        open_btn.setMenu(open_menu)
        tb.addWidget(open_btn)
        tb.addSeparator()
        
        # ── 动态功能切换按钮 + 快捷菜单 ──
        prep_btn = QToolButton()
        prep_btn.setText("🛠  波形预处理")
        prep_btn.setFixedHeight(30)
        prep_btn.setToolTip("点击展开预处理面板；箭头打开预处理快捷菜单")
        prep_btn.setPopupMode(QToolButton.MenuButtonPopup)
        prep_btn.clicked.connect(lambda: self._toggle_context_panel(0))
        prep_menu = QMenu(prep_btn)
        prep_menu.addAction("展开预处理面板", lambda: self._toggle_context_panel(0))
        prep_menu.addSeparator()
        prep_menu.addAction("去均值", self._process_demean)
        prep_menu.addAction("去趋势", self._process_detrend)
        prep_menu.addAction("滤波设置…", self._show_filter_dialog)
        prep_menu.addAction("重采样…", self._show_resample_dialog)
        prep_menu.addAction("恢复原始波形", self._reset_to_original)
        prep_btn.setMenu(prep_menu)
        tb.addWidget(prep_btn)

        pick_btn = QToolButton()
        pick_btn.setText("🎯  震相拾取")
        pick_btn.setFixedHeight(30)
        pick_btn.setToolTip("点击展开拾取面板；箭头打开拾取快捷菜单")
        pick_btn.setPopupMode(QToolButton.MenuButtonPopup)
        pick_btn.clicked.connect(lambda: self._toggle_context_panel(1))
        pick_menu = QMenu(pick_btn)
        pick_menu.addAction("展开拾取面板", lambda: self._toggle_context_panel(1))
        pick_menu.addSeparator()
        pick_menu.addAction("P 波拾取模式", lambda: self._set_pick_mode('P'))
        pick_menu.addAction("S 波拾取模式", lambda: self._set_pick_mode('S'))
        pick_menu.addAction("平移模式", lambda: self._set_pick_mode('pan'))
        pick_menu.addSeparator()
        pick_menu.addAction("导出拾取结果…", self._export_picks)
        pick_menu.addAction("读取震相文件…", self._show_import_picks_dialog)
        pick_btn.setMenu(pick_menu)
        tb.addWidget(pick_btn)

        # ── 频谱分析下拉按钮 ──────────────────────────────────────────────
        spec_btn = QPushButton("📊  频谱分析  ▾")
        spec_btn.setFixedHeight(30)
        spec_btn.setToolTip("打开频谱分析或功率谱密度分析独立窗口")
        spec_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};"
            f"color:{COLORS['accent_blue']};"
            f"border:1px solid {COLORS['accent_blue']};"
            f"border-radius:5px;font-size:11px;font-weight:600;padding:0 10px;}}"
            f"QPushButton:hover{{background:{COLORS['accent_blue']}22;}}"
            f"QPushButton:pressed{{background:{COLORS['accent_blue']}44;}}"
        )
        spec_menu = QMenu(spec_btn)
        spec_menu.setStyleSheet(
            f"QMenu{{background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;"
            f"padding:4px 0;}}"
            f"QMenu::item{{padding:7px 20px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}"
            f"QMenu::separator{{height:1px;background:{COLORS['border']};margin:3px 8px;}}"
        )
        act_spec = QAction("📈  振幅频谱分析", self)
        act_spec.setToolTip("弹出独立振幅频谱分析窗口（Amplitude Spectrum）")
        act_spec.triggered.connect(self._open_spectrum_window)
        spec_menu.addAction(act_spec)

        act_psd = QAction("〰  功率谱密度分析  (PSD)", self)
        act_psd.setToolTip("弹出独立 PSD 分析窗口，含 NLNM/NHNM 参考噪声模型")
        act_psd.triggered.connect(self._open_psd_window)
        spec_menu.addAction(act_psd)

        spec_menu.addSeparator()

        act_xcorr = QAction("⇌  多道互相关分析", self)
        act_xcorr.setToolTip(
            "计算参考道与目标道之间的互相关函数，\n"
            "显示峰值时延、相关系数及视速度估算")
        act_xcorr.triggered.connect(self._open_xcorr_window)
        spec_menu.addAction(act_xcorr)

        spec_btn.setMenu(spec_menu)
        tb.addWidget(spec_btn)
        # ─────────────────────────────────────────────────────────────────

        tb.addSeparator()

        # ── 导出下拉按钮（保存波形 + 导出图片）──────────────────────────
        export_btn = QPushButton("💾  导出  ▾")
        export_btn.setFixedHeight(30)
        export_btn.setToolTip("保存波形文件或导出波形图片")
        export_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};"
            f"color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:5px;font-size:11px;font-weight:600;padding:0 10px;}}"
            f"QPushButton:hover{{background:{COLORS['selection']};"
            f"border-color:{COLORS['accent_blue']};color:{COLORS['accent_blue']};}}"
        )
        export_menu = QMenu(export_btn)
        export_menu.setStyleSheet(
            f"QMenu{{background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;padding:4px 0;}}"
            f"QMenu::item{{padding:7px 20px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}"
            f"QMenu::separator{{height:1px;background:{COLORS['border']};margin:3px 8px;}}"
        )
        act_save_wave = QAction("💽  保存波形文件…", self)
        act_save_wave.setShortcut("Ctrl+S")
        act_save_wave.setToolTip("将选中通道保存为地震波形文件（MiniSEED / SAC / GSE2 等）")
        act_save_wave.triggered.connect(self.save_waveform)
        export_menu.addAction(act_save_wave)

        export_menu.addSeparator()
        act_export_fig = QAction("🖼  导出波形图片…", self)
        act_export_fig.setShortcut("Ctrl+E")
        act_export_fig.triggered.connect(self.export_figure)
        export_menu.addAction(act_export_fig)

        export_btn.setMenu(export_menu)
        tb.addWidget(export_btn)

        tb.addSeparator()

        # ── 批量处理按钮 ──────────────────────────────────────────────────
        batch_btn = QPushButton("⚙  批量处理")
        batch_btn.setFixedHeight(30)
        batch_btn.setToolTip(
            "打开批量预处理流水线对话框：\n"
            "选择文件目录 → 编排处理步骤 → 批量导出")
        batch_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};"
            f"color:{COLORS['accent_amber']};"
            f"border:1px solid {COLORS['accent_amber']};"
            f"border-radius:5px;font-size:11px;font-weight:600;padding:0 10px;}}"
            f"QPushButton:hover{{background:{COLORS['accent_amber']}22;}}"
            f"QPushButton:pressed{{background:{COLORS['accent_amber']}44;}}"
        )
        batch_btn.clicked.connect(self._open_batch_dialog)
        tb.addWidget(batch_btn)

        help_btn = QPushButton("❓  帮助")
        help_btn.setFixedHeight(30)
        help_btn.setToolTip("打开使用说明与功能帮助")
        help_btn.clicked.connect(self._show_toolbar_help)
        tb.addWidget(help_btn)

        
        # # ── 新增：波形预处理下拉菜单按钮 ──
        # prep_btn = QPushButton("🛠  波形预处理 ▾")
        # prep_btn.setFixedHeight(30)
        # prep_btn.setToolTip("包含去均值、去趋势、归一化、滤波等功能")
        # prep_menu = QMenu(prep_btn)
        # prep_menu.addAction("去均值 (Demean)", self._process_demean)
        # prep_menu.addAction("去趋势 (Detrend)", self._process_detrend)
        # prep_menu.addAction("归一化 (Normalize)", self._process_normalize)
        # prep_menu.addAction("带通滤波 (Bandpass)...", self._process_filter)
        
        # # 将菜单绑定到按钮
        # prep_btn.setMenu(prep_menu)
        # tb.addWidget(prep_btn)
        
        # # ── 2. 新增：拾取功能下拉菜单 ──
        # pick_btn = QPushButton("🎯  震相拾取 ▾")
        # pick_btn.setFixedHeight(30)
        # pick_btn.setToolTip("P波/S波拾取及结果导出")
        
        # pick_menu = QMenu(pick_btn)
        
        # act_p = QAction("P波拾取 (快捷键: P)", self)
        # act_p.triggered.connect(lambda: self._set_pick_mode('P'))
        # pick_menu.addAction(act_p)
        
        # act_s = QAction("S波拾取 (快捷键: S)", self)
        # act_s.triggered.connect(lambda: self._set_pick_mode('S'))
        # pick_menu.addAction(act_s)
        
        # act_pan = QAction("平移模式 (快捷键: Esc)", self)
        # act_pan.triggered.connect(lambda: self._set_pick_mode('pan'))
        # pick_menu.addAction(act_pan)
        
        # pick_menu.addSeparator()
        
        # act_export = QAction("导出拾取结果...", self)
        # act_export.triggered.connect(self._export_picks)
        # pick_menu.addAction(act_export)
        
        # pick_btn.setMenu(pick_menu)
        # tb.addWidget(pick_btn)        
        
        # ────────────────────────────────

    # ── 状态栏 ─────────────────────────────────────────────────────────────
    def _setup_statusbar(self):
        sb = self.statusBar()
        self.status_lbl = QLabel("就绪  |  请打开地震数据文件开始分析")
        self.status_lbl.setStyleSheet(
            f"color:{COLORS['text_secondary']}; padding: 0 8px;"
        )
        sb.addWidget(self.status_lbl, 1)

        self.format_lbl = QLabel("")
        self.format_lbl.setStyleSheet(
            f"color:{COLORS['accent_green']}; padding: 0 8px;"
            f" font-family:Consolas; font-size:11px;"
        )
        sb.addPermanentWidget(self.format_lbl)

    def _set_status(self, msg):
        """Set the status-bar text using a single helper entry point."""
        self.status_lbl.setText(msg)

    # ── 文件操作 ───────────────────────────────────────────────────────────
    def open_file(self):
        """打开一个或多个地震数据文件"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "打开地震数据文件（可多选）", self._last_dir,
            "地震数据文件 (*.mseed *.seed *.sac *.SAC *.miniseed "
            "*.gse2 *.gse *.dat *.seisan *.msd *.segy *.su);;"
            "MiniSEED 文件 (*.mseed *.miniseed *.msd);;"
            "SAC 文件 (*.sac *.SAC);;"
            "SEED 文件 (*.seed);;"
            "GSE2 文件 (*.gse2 *.gse);;"
            "所有文件 (*.*)"
        )
        if paths:
            self._last_dir = os.path.dirname(paths[0])
            self._load_paths(paths)

    def open_folder(self):
        """打开文件夹，递归扫描其中所有地震数据文件"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择包含地震数据的文件夹", self._last_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self._last_dir = folder
            self._load_paths([folder])

    def reload_file(self):
        """Reload the most recently opened files or folders."""
        if self._loaded_paths:
            self._load_paths(self._loaded_paths)
        else:
            self._set_status("没有已加载的文件，请先打开文件或文件夹")

    def _load_paths(self, paths: list):
        """统一入口：启动后台线程加载一组路径（文件/文件夹均可）"""
        self._loaded_paths = list(paths)   # 保存用于 reload
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.header_panel.clear()
        self.canvas._draw_welcome()

        label = (os.path.basename(paths[0]) if len(paths) == 1
                 else f"{len(paths)} 个路径")
        self._set_status(f"正在加载：{label} ...")

        self.loader = DataLoaderThread(paths)
        self.loader.finished.connect(self._on_load_finished)
        self.loader.error.connect(self._on_load_error)
        self.loader.progress.connect(self.progress_bar.setValue)
        self.loader.file_progress.connect(self._on_file_progress)
        self.loader.start()

    def _on_file_progress(self, current: int, total: int):
        """多文件加载时更新状态栏提示"""
        self._set_status(f"正在加载第 {current} / {total} 个文件 ...")

    def _on_load_finished(self, stream, label):
        """Update UI state after the background loader returns successfully."""
        self.stream = stream
        # 保存原始数据备份（永不修改）
        self._orig_stream = stream.copy()
        self._stream_unit = 'counts'
        # 清空处理历史
        self._proc_history.clear()
        if hasattr(self, '_hist_chip_layout'):
            self._clear_proc_history_ui()
        self.progress_bar.hide()

        # ── 文件信息汇总 ─────────────────────────────────────────────────
        # 计算所有源文件总大小
        total_bytes = 0
        for p in (self._loaded_paths or []):
            if os.path.isfile(p):
                total_bytes += os.path.getsize(p)
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            total_bytes += os.path.getsize(fp)
                        except OSError:
                            pass
        size_str = (f"{total_bytes/1024/1024:.2f} MB"
                    if total_bytes > 1024*1024
                    else f"{total_bytes/1024:.1f} KB")

        fmt = stream[0].stats._format or '未知格式'
        n_files = len(self._loaded_paths) if self._loaded_paths else 1

        self.file_name_lbl.setText(label)
        self.file_meta_lbl.setText(
            f"{len(stream)} 条通道   |   "
            f"{n_files} 个文件   |   "
            f"{size_str}   |   {fmt}"
        )
        self.format_lbl.setText(f"  {fmt}  ")
        self.setWindowTitle(f"SeismoView — {label}")
        self.header_panel.load_stream(stream)

        # 绘制波形：默认只绘制第一道（已在左侧列表中自动选中），
        # 其他道通过左侧列表点选激活。
        self.canvas.plot_stream(stream, [0])
        self._set_status(
            f"已加载 {len(stream)} 道（来自 {n_files} 个文件）  |  "
            f"当前显示第 1 道  |  在左侧列表中选择要查看/处理的通道"
            )
        self.canvas_title.setText(
            f"波形显示  —  {stream[0].stats.network}.{stream[0].stats.station}"
        )

    def _on_load_error(self, msg):
        """Handle load failures and restore the UI to a safe idle state."""
        self.progress_bar.hide()
        self._set_status("文件加载失败")
        QMessageBox.critical(self, "文件读取错误", msg)

    # ── 波形绘制控制 ───────────────────────────────────────────────────────
    def _on_trace_selected(self, indices):
        """Plot the traces selected in the left header panel."""
        if self.stream and indices:
            self.canvas.plot_stream(self.stream, indices)
            tr = self.stream[indices[0]]
            s  = tr.stats
            self.canvas_title.setText(
                f"{s.network}.{s.station}.{s.channel}  "
                f"— 共选 {len(indices)} 道"
            )
            self._set_status(
                f"已选中 {len(indices)} 道  |  预处理 / 拾取 / 频谱分析将仅作用于选中道  "
                f"|  {s.network}.{s.station}.{s.location}.{s.channel}"
                + (f" 等" if len(indices) > 1 else "")
            )

    def _plot_all(self):
        """Plot all traces currently loaded in memory."""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return
        n = min(len(self.stream), 16)
        self.canvas.plot_stream(self.stream, list(range(n)))
        self._set_status(f"绘制全部通道（共 {n} 道）")

    def _plot_selected(self):
        """Plot only traces selected in the header panel."""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return
        sel = self.header_panel.tree.selectedItems()
        if not sel:
            self._set_status("请先在通道列表中选择要绘制的通道")
            return
        indices = [item.data(0, Qt.UserRole) for item in sel]
        self.canvas.plot_stream(self.stream, indices)
        self._set_status(f"绘制选中 {len(indices)} 道")

    def _reset_view(self):
        """Reset the seismic canvas to its original x-axis extent."""
        self.canvas.reset_view()

    def _zoom_in(self):
        self.canvas.zoom_in()

    def _zoom_out(self):
        self.canvas.zoom_out()

    # ── 数据预处理逻辑 ──────────────────────────────────────────────────────
    def _process_demean(self):
        if not self.stream: return
        idxs = self._selected_stream_indices()
        for i in idxs:
            self.stream[i].detrend('demean')
        self._add_proc_step("去均值", color='#6B7280')
        self._set_status(f"去均值完成  |  作用于 {len(idxs)} 道")
        self._replot_current()

    def _process_detrend(self):
        if not self.stream: return
        idxs = self._selected_stream_indices()
        for i in idxs:
            self.stream[i].detrend('linear')
        self._add_proc_step("去趋势", color='#6B7280')
        self._set_status(f"去趋势完成  |  作用于 {len(idxs)} 道")
        self._replot_current()

    # ── 截窗 ──────────────────────────────────────────────────────────────────
    def _show_trim_menu(self):
        """「✂ 截窗 ▾」按钮的下拉菜单"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return
        btn = self.sender()
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{COLORS['bg_card']};color:{COLORS['text_primary']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;padding:4px 0;}}"
            f"QMenu::item{{padding:7px 20px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}"
            f"QMenu::separator{{height:1px;background:{COLORS['border']};margin:3px 8px;}}"
        )
        act_manual = menu.addAction("🖱  手动截窗（在波形上拖拽选区）")
        act_manual.setToolTip("切换到截窗模式，在波形上左键拖拽选择保留的时间段")
        menu.addSeparator()
        act_dialog = menu.addAction("⌨  输入截窗（手动输入时间）")
        act_dialog.setToolTip("打开对话框，精确输入截窗的开始 / 结束时间或持续时长")

        act_manual.triggered.connect(self._start_manual_trim)
        act_dialog.triggered.connect(self._show_trim_dialog)

        pos = btn.mapToGlobal(btn.rect().bottomLeft()) if btn else self.cursor().pos()
        menu.exec_(pos)

    def _start_manual_trim(self):
        """启动手动截窗模式（两次点击：第一次定起点，第二次定终点）"""
        if not self.stream:
            return
        self.canvas.set_pick_mode('pan')   # 关闭拾取模式，避免冲突
        self.canvas.set_trim_mode(True)

    def _on_trim_requested(self, t_start_rel: float, t_end_rel: float):
        """canvas 拖拽完成后回调，执行实际截窗"""
        if not self.stream or not self._raw_meta_starttime():
            return
        base = self._raw_meta_starttime()
        from obspy import UTCDateTime
        t0 = base + t_start_rel
        t1 = base + t_end_rel
        self._process_trim(t0, t1)

    def _raw_meta_starttime(self):
        """从当前 stream 取第一道的 starttime（ObsPy UTCDateTime）"""
        if self.stream:
            return self.stream[0].stats.starttime
        return None

    def _show_trim_dialog(self):
        """弹出精确输入截窗对话框"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        from obspy import UTCDateTime

        # 取第一道时间范围作为默认值
        st0   = self.stream[0].stats.starttime
        et0   = max(tr.stats.endtime for tr in self.stream)
        total = float(et0 - st0)

        dlg = QDialog(self)
        dlg.setWindowTitle("截窗参数设置")
        dlg.setMinimumWidth(440)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # ── 当前数据时间范围提示 ─────────────────────────────────────────
        info_lbl = QLabel(
            f"数据时间范围：\n"
            f"  起始  {str(st0)[:23]}  UTC\n"
            f"  终止  {str(et0)[:23]}  UTC\n"
            f"  总时长  {total:.3f} 秒"
        )
        info_lbl.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:10px;font-family:Consolas;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:7px 10px;"
        )
        root.addWidget(info_lbl)

        # ── 截窗模式 Tab ─────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabBar::tab{{padding:5px 14px;font-size:11px;}}"
            f"QTabBar::tab:selected{{color:{COLORS['accent_blue']};"
            f"border-bottom:2px solid {COLORS['accent_blue']};}}"
        )

        # ════ Tab A：相对秒（偏移量）────────────────────────────────────
        tab_rel = QWidget()
        rel_form = QFormLayout(tab_rel)
        rel_form.setContentsMargins(10, 10, 10, 10)
        rel_form.setSpacing(8)

        rel_start = QDoubleSpinBox()
        rel_start.setRange(0, total)
        rel_start.setValue(0.0)
        rel_start.setDecimals(3)
        rel_start.setSuffix("  秒")
        rel_start.setToolTip("相对于数据起始时刻的偏移量（秒）")
        rel_form.addRow("起始偏移：", rel_start)

        # 结束：直接输入结束偏移 or 输入持续时长（两种方式联动）
        end_mode_combo = QComboBox()
        end_mode_combo.addItems(["指定结束偏移（秒）", "指定持续时长（秒）"])
        end_mode_combo.setFixedHeight(24)
        rel_form.addRow("结束方式：", end_mode_combo)

        rel_end = QDoubleSpinBox()
        rel_end.setRange(0.001, total)
        rel_end.setValue(total)
        rel_end.setDecimals(3)
        rel_end.setSuffix("  秒")

        rel_dur = QDoubleSpinBox()
        rel_dur.setRange(0.001, total)
        rel_dur.setValue(total)
        rel_dur.setDecimals(3)
        rel_dur.setSuffix("  秒")
        rel_dur.setVisible(False)

        end_stack_widget = QWidget()
        end_stack_lay = QVBoxLayout(end_stack_widget)
        end_stack_lay.setContentsMargins(0, 0, 0, 0)
        end_stack_lay.addWidget(rel_end)
        end_stack_lay.addWidget(rel_dur)
        rel_form.addRow("结束/时长：", end_stack_widget)

        # 实时预览标签
        preview_lbl = QLabel()
        preview_lbl.setStyleSheet(
            f"color:{COLORS['accent_blue']};font-size:10px;font-family:Consolas;"
            f"background:{COLORS['bg_header']};border-radius:3px;padding:4px 6px;"
        )
        rel_form.addRow("截窗预览：", preview_lbl)

        def _update_rel_preview():
            s  = rel_start.value()
            if end_mode_combo.currentIndex() == 0:
                e = rel_end.value()
            else:
                e = s + rel_dur.value()
            t0 = st0 + s
            t1 = st0 + e
            dur = e - s
            if dur <= 0:
                preview_lbl.setText("⚠ 结束时间必须晚于开始时间")
                preview_lbl.setStyleSheet(
                    f"color:#DC2626;font-size:10px;font-family:Consolas;"
                    f"background:#FEF2F2;border-radius:3px;padding:4px 6px;"
                )
            else:
                preview_lbl.setText(
                    f"{str(t0)[:23]}  →  {str(t1)[:23]}\n"
                    f"保留 {dur:.3f} 秒"
                )
                preview_lbl.setStyleSheet(
                    f"color:{COLORS['accent_blue']};font-size:10px;font-family:Consolas;"
                    f"background:{COLORS['bg_header']};border-radius:3px;padding:4px 6px;"
                )

        def _on_end_mode_change(idx):
            rel_end.setVisible(idx == 0)
            rel_dur.setVisible(idx == 1)
            _update_rel_preview()

        end_mode_combo.currentIndexChanged.connect(_on_end_mode_change)
        rel_start.valueChanged.connect(_update_rel_preview)
        rel_end.valueChanged.connect(_update_rel_preview)
        rel_dur.valueChanged.connect(_update_rel_preview)
        _update_rel_preview()

        tabs.addTab(tab_rel, "⏱  相对时间（秒偏移）")

        # ════ Tab B：绝对 UTC 时间字符串 ────────────────────────────────
        tab_abs = QWidget()
        abs_form = QFormLayout(tab_abs)
        abs_form.setContentsMargins(10, 10, 10, 10)
        abs_form.setSpacing(8)

        from PyQt5.QtWidgets import QLineEdit
        abs_start_edit = QLineEdit(str(st0)[:23])
        abs_start_edit.setFont(QFont("Consolas", 10))
        abs_start_edit.setPlaceholderText("例：2024-01-01T12:00:00.000")
        abs_form.addRow("开始时间 (UTC)：", abs_start_edit)

        abs_end_edit = QLineEdit(str(et0)[:23])
        abs_end_edit.setFont(QFont("Consolas", 10))
        abs_end_edit.setPlaceholderText("例：2024-01-01T12:01:00.000")
        abs_form.addRow("结束时间 (UTC)：", abs_end_edit)

        abs_preview_lbl = QLabel("请输入有效的 UTC 时间字符串")
        abs_preview_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;"
            f"background:{COLORS['bg_header']};border-radius:3px;padding:4px 6px;"
        )
        abs_form.addRow("截窗预览：", abs_preview_lbl)

        def _update_abs_preview():
            try:
                t0 = UTCDateTime(abs_start_edit.text().strip())
                t1 = UTCDateTime(abs_end_edit.text().strip())
                dur = float(t1 - t0)
                if dur <= 0:
                    raise ValueError("结束时间必须晚于开始时间")
                abs_preview_lbl.setText(
                    f"{str(t0)[:23]}  →  {str(t1)[:23]}\n保留 {dur:.3f} 秒"
                )
                abs_preview_lbl.setStyleSheet(
                    f"color:{COLORS['accent_blue']};font-size:10px;font-family:Consolas;"
                    f"background:{COLORS['bg_header']};border-radius:3px;padding:4px 6px;"
                )
            except Exception as e:
                abs_preview_lbl.setText(f"⚠  {e}")
                abs_preview_lbl.setStyleSheet(
                    f"color:#DC2626;font-size:10px;font-family:Consolas;"
                    f"background:#FEF2F2;border-radius:3px;padding:4px 6px;"
                )

        abs_start_edit.textChanged.connect(_update_abs_preview)
        abs_end_edit.textChanged.connect(_update_abs_preview)
        _update_abs_preview()

        tabs.addTab(tab_abs, "📅  绝对时间（UTC）")

        root.addWidget(tabs)

        # ── 填充值选项 ───────────────────────────────────────────────────
        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("超出范围时："))
        pad_combo = QComboBox()
        pad_combo.addItems(["不填充（截断）", "用零值填充（pad=0）", "用掩码填充（masked）"])
        pad_combo.setFixedHeight(24)
        pad_combo.setToolTip(
            "当截窗范围超出数据范围时：\n"
            "不填充：自动对齐到数据边界\n"
            "零值填充：超出部分补零（pad=True, fill_value=0）\n"
            "掩码填充：超出部分用 numpy masked array 填充"
        )
        pad_row.addWidget(pad_combo)
        pad_row.addStretch()
        root.addLayout(pad_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        pad_idx = pad_combo.currentIndex()
        pad     = pad_idx > 0
        fill    = 0 if pad_idx == 1 else None

        try:
            if tabs.currentIndex() == 0:
                # 相对时间
                s = rel_start.value()
                e = (rel_end.value() if end_mode_combo.currentIndex() == 0
                     else s + rel_dur.value())
                t0_abs = st0 + s
                t1_abs = st0 + e
            else:
                # 绝对时间
                t0_abs = UTCDateTime(abs_start_edit.text().strip())
                t1_abs = UTCDateTime(abs_end_edit.text().strip())

            if float(t1_abs - t0_abs) <= 0:
                QMessageBox.warning(self, "参数错误", "结束时间必须晚于开始时间！")
                return

            self._process_trim(t0_abs, t1_abs, pad=pad, fill_value=fill)
        except Exception as e:
            QMessageBox.critical(self, "时间解析失败", f"无法解析时间参数：\n{e}")

    def _process_trim(self, t_start, t_end, pad: bool = False, fill_value=None):
        """执行 stream.trim，更新视图"""
        if not self.stream:
            return
        try:
            from obspy import UTCDateTime
            t0 = UTCDateTime(t_start)
            t1 = UTCDateTime(t_end)
            if pad:
                self.stream.trim(starttime=t0, endtime=t1,
                                 pad=True, fill_value=fill_value or 0)
            else:
                self.stream.trim(starttime=t0, endtime=t1)

            dur = float(t1 - t0)
            desc = (f"截窗 {str(t0)[11:23]}—{str(t1)[11:23]}"
                    f" ({dur:.2f}s)")
            self._add_proc_step(desc, color='#EA580C')
            self._set_status(
                f"✓ 截窗完成  |  "
                f"{str(t0)[:19].replace('T',' ')}  →  "
                f"{str(t1)[:19].replace('T',' ')}  "
                f"（保留 {dur:.3f} 秒）"
            )
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "截窗失败", f"处理时发生错误：\n{e}")

    # ── 分量旋转 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _group_by_station(stream):
        """
        将 Stream 按 NET.STA.LOC.频带 分组，返回：
          { key: [(idx, tr), ...] }  其中 key = "NET.STA.LOC.band"
        """
        groups = {}
        for i, tr in enumerate(stream):
            s   = tr.stats
            band = s.channel[:2] if len(s.channel) >= 2 else s.channel
            key  = f"{s.network}.{s.station}.{s.location or ''}.{band}"
            groups.setdefault(key, []).append((i, tr))
        return groups

    @staticmethod
    def _read_baz_from_header(traces):
        """
        尝试从多个头段字段读取反方位角（back-azimuth）。
        返回 float 或 None。
        """
        for _, tr in traces:
            s = tr.stats
            # SAC 文件专有头段
            if hasattr(s, 'sac'):
                baz = getattr(s.sac, 'baz', None)
                if baz is not None and not np.isnan(float(baz)):
                    return float(baz)
            # 通用 extra 字段
            for field in ('back_azimuth', 'backazimuth', 'baz'):
                v = getattr(s, field, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
        return None

    @staticmethod
    def _read_inc_from_header(traces):
        """
        尝试从头段读取入射角（inclination，相对垂直方向）。
        返回 float 或 None。
        """
        for _, tr in traces:
            s = tr.stats
            if hasattr(s, 'sac'):
                inc = getattr(s.sac, 'user0', None)   # 常见存放位置
                if inc is not None and not np.isnan(float(inc)):
                    return float(inc)
            for field in ('inclination', 'incidence', 'inc'):
                v = getattr(s, field, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
        return None

    def _show_rotate_dialog(self):
        """分量旋转主对话框"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        groups = self._group_by_station(self.stream)
        if not groups:
            self._set_status("当前数据无可用通道")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("分量旋转")
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # ── 台站 / 分量组选择 ─────────────────────────────────────────────
        grp_lbl = QLabel("选择台站 / 频带（显示检测到的水平分量）：")
        grp_lbl.setStyleSheet(
            f"font-weight:600;color:{COLORS['text_primary']};font-size:11px;")
        root.addWidget(grp_lbl)

        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView

        grp_list = QListWidget()
        grp_list.setSelectionMode(QAbstractItemView.SingleSelection)
        grp_list.setFixedHeight(min(180, 30 + len(groups) * 30))
        grp_list.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_card']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;}}"
            f"QListWidget::item{{padding:4px 10px;font-family:Consolas;font-size:11px;}}"
            f"QListWidget::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}"
        )

        # 方向分量检测（Z/N/E/1/2/R/T/L/Q）
        DIR_SET = set('ZNERLT12QU')
        valid_keys = []
        for key, trs in groups.items():
            channels = sorted(tr.stats.channel for _, tr in trs)
            comp_chars = [c[-1].upper() for c in channels if c]
            horiz = [c for c in comp_chars if c in DIR_SET and c != 'Z']
            vert  = [c for c in comp_chars if c == 'Z']
            tag   = '  '.join(channels)
            note  = ''
            if 'N' in comp_chars and 'E' in comp_chars:
                note = '  ✓ ZNE可旋转'
            elif '1' in comp_chars and '2' in comp_chars:
                note = '  ✓ Z12可旋转'
            elif len(horiz) >= 2:
                note = f'  ⚠ 含 {"/".join(comp_chars)} 请手动确认'
            else:
                note = '  ✗ 分量不足'
            item = QListWidgetItem(f"{key}   [{tag}]{note}")
            item.setData(Qt.UserRole, key)
            grp_list.addItem(item)
            valid_keys.append(key)

        if grp_list.count() > 0:
            grp_list.setCurrentRow(0)
        root.addWidget(grp_list)

        # ── 旋转方法 ──────────────────────────────────────────────────────
        method_box = QGroupBox("旋转方法")
        method_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:10px;padding-top:6px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 6px;"
            f"font-weight:600;letter-spacing:1px;}}"
        )
        mf = QFormLayout(method_box)
        mf.setSpacing(8)

        METHOD_OPTS = [
            ("NE → RT    （水平旋转到径向/切向，需要反方位角；Z 分量不变）",    "NE->RT"),
            ("RT → NE    （径向/切向旋转回 N/E，需要反方位角）",               "RT->NE"),
            ("ZNE → LQT  （旋转到 P/SV/SH 波坐标，需要反方位角 + 入射角）",   "ZNE->LQT"),
            ("LQT → ZNE  （P/SV/SH 波坐标旋转回 ZNE，需要反方位角 + 入射角）","LQT->ZNE"),
            ("→ ZNE       （利用头段方位角将任意水平分量转回 ZNE）",           "->ZNE"),
        ]
        method_combo = QComboBox()
        method_combo.setFixedHeight(26)
        for label, code in METHOD_OPTS:
            method_combo.addItem(label, code)
        mf.addRow("旋转方向：", method_combo)
        root.addWidget(method_box)

        # ── 反方位角 / 入射角 ─────────────────────────────────────────────
        param_box = QGroupBox("旋转参数")
        param_box.setStyleSheet(method_box.styleSheet())
        pf = QFormLayout(param_box)
        pf.setSpacing(8)

        baz_spin = QDoubleSpinBox()
        baz_spin.setRange(0.0, 360.0)
        baz_spin.setDecimals(4)
        baz_spin.setSuffix("  °")
        baz_spin.setToolTip(
            "反方位角（Back-azimuth）：从台站指向震源的方位角\n"
            "取值范围 0–360°。若头段中含有该信息，已自动填入。")

        inc_spin = QDoubleSpinBox()
        inc_spin.setRange(0.0, 180.0)
        inc_spin.setDecimals(4)
        inc_spin.setSuffix("  °")
        inc_spin.setToolTip(
            "入射角（Inclination from vertical）：P 波与竖直方向的夹角\n"
            "仅 ZNE→LQT 方法需要。典型范围 0–30°。")

        baz_src_lbl = QLabel("来源：手动输入")
        baz_src_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;")
        inc_src_lbl = QLabel("来源：手动输入")
        inc_src_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;")

        baz_row = QHBoxLayout()
        baz_row.addWidget(baz_spin)
        baz_row.addWidget(baz_src_lbl)
        pf.addRow("反方位角 BAZ：", baz_row)

        inc_row = QHBoxLayout()
        inc_row.addWidget(inc_spin)
        inc_row.addWidget(inc_src_lbl)
        pf.addRow("入射角 INC：", inc_row)

        root.addWidget(param_box)

        # 自动填充头段信息（随选中组动态更新）
        def _autofill_params():
            sel = grp_list.currentItem()
            if sel is None:
                return
            key  = sel.data(Qt.UserRole)
            trs  = groups.get(key, [])
            baz  = self._read_baz_from_header(trs)
            inc  = self._read_inc_from_header(trs)
            if baz is not None:
                baz_spin.setValue(baz)
                baz_src_lbl.setText(f"来源：头段自动读取  ({baz:.4f}°)")
                baz_src_lbl.setStyleSheet(
                    f"color:{COLORS['accent_green']};font-size:10px;")
            else:
                baz_src_lbl.setText("来源：手动输入（头段中未找到反方位角）")
                baz_src_lbl.setStyleSheet(
                    f"color:{COLORS['accent_amber']};font-size:10px;")
            if inc is not None:
                inc_spin.setValue(inc)
                inc_src_lbl.setText(f"来源：头段自动读取  ({inc:.4f}°)")
                inc_src_lbl.setStyleSheet(
                    f"color:{COLORS['accent_green']};font-size:10px;")
            else:
                inc_src_lbl.setText("来源：手动输入（头段中未找到入射角）")
                inc_src_lbl.setStyleSheet(
                    f"color:{COLORS['accent_amber']};font-size:10px;")

        grp_list.currentItemChanged.connect(lambda *_: _autofill_params())
        _autofill_params()   # 初次填充

        # 入射角随方法显示/隐藏
        def _on_method_change(idx):
            code = METHOD_OPTS[idx][1]
            need_inc = code in ('ZNE->LQT', 'LQT->ZNE')
            inc_spin.setEnabled(need_inc)
            inc_src_lbl.setEnabled(need_inc)
            need_baz = code != '->ZNE'
            baz_spin.setEnabled(need_baz)
            baz_src_lbl.setEnabled(need_baz)
        method_combo.currentIndexChanged.connect(_on_method_change)
        _on_method_change(0)

        # ── 说明 ─────────────────────────────────────────────────────────
        note_lbl = QLabel(
            "旋转完成后，原分量将被旋转后的新分量替换（原始备份可通过「↺ 重置原始」恢复）。\n"
            "NE→RT：R=径向（朝震源方向）/ T=切向；ZNE→LQT：L=P波方向 / Q=SV波方向 / T=SH波方向"
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:6px 8px;"
        )
        root.addWidget(note_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        sel = grp_list.currentItem()
        if sel is None:
            return
        key        = sel.data(Qt.UserRole)
        trs        = groups.get(key, [])
        idx        = method_combo.currentIndex()
        method     = METHOD_OPTS[idx][1]
        baz        = baz_spin.value()
        inc        = inc_spin.value()

        self._process_rotate(key, trs, method, baz, inc)

    def _process_rotate(self, group_key, trs, method, back_azimuth, inclination):
        """
        执行分量旋转。

        Parameters
        ----------
        group_key     : str   台站频带组 key，用于日志
        trs           : list  [(stream_idx, Trace), ...]
        method        : str   ObsPy rotate 方法字符串
        back_azimuth  : float 反方位角 (°)
        inclination   : float 入射角 (°)，仅 ZNE->LQT 需要
        """
        if not self.stream:
            return
        try:
            from obspy.core import Stream as ObspyStream

            # 用选中道构建临时 Stream
            indices  = [i for i, _ in trs]
            tmp_st   = ObspyStream([self.stream[i].copy() for i in indices])

            # 执行旋转
            kwargs = {'method': method, 'back_azimuth': back_azimuth}
            if method in ('ZNE->LQT', 'LQT->ZNE'):
                kwargs['inclination'] = inclination
            if method == '->ZNE':
                # ->ZNE 利用头段方位角，不传 back_azimuth
                kwargs.pop('back_azimuth', None)

            tmp_st.rotate(**kwargs)

            # 将旋转后的结果写回 self.stream
            # 删除原有道（倒序删除，防止索引偏移）
            for i in sorted(indices, reverse=True):
                self.stream.remove(self.stream[i])
            # 追加旋转后的道
            for tr in tmp_st:
                self.stream.append(tr)

            # 更新左侧面板
            self.header_panel.load_stream(self.stream)

            new_chs = '  /  '.join(tr.stats.channel for tr in tmp_st)
            desc    = f"旋转({method},{back_azimuth:.1f}°)"
            self._add_proc_step(desc, color='#0891B2')
            self._set_status(
                f"✓ 分量旋转完成  |  方法：{method}  |  BAZ：{back_azimuth:.4f}°  |  "
                f"新分量：{new_chs}"
            )
            # 绘制旋转后的新分量
            n_new = len(tmp_st)
            total = len(self.stream)
            start_idx = total - n_new
            self.canvas.plot_stream(self.stream, list(range(start_idx, total)))

        except Exception as e:
            QMessageBox.critical(
                self, "分量旋转失败",
                f"旋转时发生错误：\n\n{str(e)}\n\n"
                "常见原因：\n"
                "① 分量不完整（NE→RT 需要 N/E 两道；ZNE→LQT 需要 Z/N/E 三道）\n"
                "② 反方位角超出范围（应为 0–360°）\n"
                "③ 各分量采样率或时间范围不一致\n"
                "④ →ZNE 方法需要头段中包含 azimuth / dip 信息"
            )

    # ── 重采样 ────────────────────────────────────────────────────────────────
    def _show_resample_dialog(self):
        """重采样对话框：显示各道当前采样率，输入目标采样率，选择方法"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        # 收集各道采样率信息
        rates = sorted({tr.stats.sampling_rate for tr in self.stream})
        rate_info = "  /  ".join(f"{r:g} Hz" for r in rates)

        dlg = QDialog(self)
        dlg.setWindowTitle("重采样（Resample）")
        dlg.setMinimumWidth(440)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # ── 当前采样率提示 ────────────────────────────────────────────────
        info_lbl = QLabel(
            f"当前数据共 {len(self.stream)} 道\n"
            f"采样率：{rate_info}"
        )
        info_lbl.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:10px;font-family:Consolas;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:7px 10px;"
        )
        root.addWidget(info_lbl)

        # ── 目标采样率 ────────────────────────────────────────────────────
        param_box = QGroupBox("重采样参数")
        param_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:10px;padding-top:6px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 6px;"
            f"font-weight:600;letter-spacing:1px;}}"
        )
        pf = QFormLayout(param_box)
        pf.setSpacing(10)
        pf.setContentsMargins(12, 12, 12, 12)

        # 目标采样率输入
        target_spin = QDoubleSpinBox()
        target_spin.setRange(0.001, 100000.0)
        target_spin.setDecimals(3)
        target_spin.setSuffix("  Hz")
        target_spin.setFixedHeight(26)
        target_spin.setToolTip("目标采样率（Hz）。低于当前采样率为降采样，高于为升采样。")
        # 默认填入当前最高采样率的一半（常见降采样场景）
        default_rate = rates[-1] / 2.0 if rates else 50.0
        target_spin.setValue(default_rate)
        pf.addRow("目标采样率：", target_spin)

        # 快捷预设按钮
        preset_row = QHBoxLayout()
        preset_row.setSpacing(5)
        for hz in [1, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
            pb = QPushButton(f"{hz}")
            pb.setFixedSize(42, 22)
            pb.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_header']};"
                f"color:{COLORS['text_secondary']};"
                f"border:1px solid {COLORS['border']};border-radius:3px;"
                f"font-size:10px;}}"
                f"QPushButton:hover{{background:{COLORS['accent_blue']}22;"
                f"color:{COLORS['accent_blue']};"
                f"border-color:{COLORS['accent_blue']};}}"
            )
            pb.clicked.connect(lambda _, v=hz: target_spin.setValue(float(v)))
            preset_row.addWidget(pb)
        preset_row.addStretch()
        preset_wrap = QWidget()
        preset_wrap.setLayout(preset_row)
        pf.addRow("快捷预设 (Hz)：", preset_wrap)

        # 重采样方法
        method_combo = QComboBox()
        method_combo.setFixedHeight(26)
        method_combo.addItem("resample  — FFT 频域重采样（推荐，保频谱精度）",  "resample")
        method_combo.addItem("decimate  — 整数倍降采样 + 抗混叠低通滤波",      "decimate")
        method_combo.addItem("interpolate — 时域多项式插值（适合小比例变化）",  "interpolate")
        method_combo.setToolTip(
            "resample：ObsPy resample()，FFT 域变换，精度最高，支持任意目标采样率\n"
            "decimate：ObsPy decimate()，仅支持整数倍降采样，自动应用抗混叠滤波\n"
            "interpolate：ObsPy interpolate()，时域插值，适合采样率微调"
        )
        pf.addRow("重采样方法：", method_combo)

        # 抗混叠（仅 resample/interpolate 有效）
        aa_check = QCheckBox("降采样前自动应用抗混叠低通滤波（推荐）")
        aa_check.setChecked(True)
        aa_check.setToolTip(
            "勾选后，当目标采样率 < 当前采样率时，\n"
            "会在重采样前应用截止频率 = 目标奈奎斯特频率的低通滤波，防止频谱混叠。\n"
            "decimate 方法内置抗混叠，此选项对其无效。"
        )
        pf.addRow("抗混叠：", aa_check)

        # 应用范围
        scope_combo = QComboBox()
        scope_combo.setFixedHeight(26)
        scope_combo.addItem("所有通道", "all")
        scope_combo.addItem("仅当前选中通道", "selected")
        pf.addRow("应用范围：", scope_combo)

        root.addWidget(param_box)

        # ── 实时预览标签 ──────────────────────────────────────────────────
        preview_lbl = QLabel()
        preview_lbl.setWordWrap(True)
        preview_lbl.setStyleSheet(
            f"color:{COLORS['accent_blue']};font-size:10px;font-family:Consolas;"
            f"background:{COLORS['bg_header']};border-radius:3px;padding:6px 8px;"
        )

        def _update_preview():
            tgt = target_spin.value()
            method = method_combo.currentData()
            lines = []
            for r in rates:
                if abs(r - tgt) < 1e-6:
                    lines.append(f"  {r:g} Hz  →  {tgt:g} Hz  （无需处理）")
                    continue
                direction = "↓ 降采样" if tgt < r else "↑ 升采样"
                ratio = r / tgt if tgt < r else tgt / r
                if method == 'decimate':
                    factor = int(round(r / tgt))
                    actual = r / factor if factor > 0 else tgt
                    warn = (f"  ⚠ 非整数倍（实际将采样至 {actual:g} Hz）"
                            if abs(actual - tgt) > 0.01 else "")
                    lines.append(f"  {r:g} Hz  →  {tgt:g} Hz  {direction}  ×{factor}{warn}")
                else:
                    lines.append(f"  {r:g} Hz  →  {tgt:g} Hz  {direction}  ×{ratio:.3f}")
            preview_lbl.setText("\n".join(lines) if lines else "—")

        target_spin.valueChanged.connect(_update_preview)
        method_combo.currentIndexChanged.connect(_update_preview)
        _update_preview()
        root.addWidget(preview_lbl)

        # ── 说明 ──────────────────────────────────────────────────────────
        note_lbl = QLabel(
            "建议在重采样前先进行去均值 + 去趋势 + Taper，以减少边缘效应。\n"
            "decimate 仅支持整数倍降采样；若目标非整数比，请改用 resample 或 interpolate。"
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:6px 8px;"
        )
        root.addWidget(note_lbl)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        target_rate = target_spin.value()
        method      = method_combo.currentData()
        anti_alias  = aa_check.isChecked()
        scope       = scope_combo.currentData()

        # 确定要处理的 trace 索引
        if scope == 'selected':
            indices = self._selected_stream_indices()
        else:
            indices = self._selected_stream_indices()  # 两者都用选中的道
        # 若用户明确选择"所有通道"，则处理全部
        if scope == 'all':
            indices = list(range(len(self.stream)))

        self._process_resample(target_rate, method, anti_alias, indices)

    def _process_resample(self, target_rate: float, method: str,
                          anti_alias: bool, indices: list):
        """执行重采样"""
        if not self.stream:
            return
        try:
            changed = 0
            for i in indices:
                tr = self.stream[i]
                src_rate = tr.stats.sampling_rate
                if abs(src_rate - target_rate) < 1e-6:
                    continue   # 采样率已一致，跳过

                # 抗混叠低通滤波（仅降采样时，且非 decimate）
                if anti_alias and target_rate < src_rate and method != 'decimate':
                    nyq = target_rate / 2.0 * 0.9   # 留 10% 余量
                    tr.filter('lowpass', freq=nyq, corners=4, zerophase=True)

                if method == 'resample':
                    tr.resample(sampling_rate=target_rate)
                elif method == 'decimate':
                    factor = int(round(src_rate / target_rate))
                    if factor < 2:
                        factor = 2
                    tr.decimate(factor=factor, no_filter=True)
                elif method == 'interpolate':
                    tr.interpolate(sampling_rate=target_rate)

                changed += 1

            if changed == 0:
                self._set_status("所有选中通道的采样率已与目标一致，无需处理")
                return

            method_name = {'resample': 'FFT重采样',
                           'decimate': 'Decimate',
                           'interpolate': '插值重采样'}.get(method, method)
            desc = f"重采样→{target_rate:g}Hz ({method_name})"
            self._add_proc_step(desc, color='#0D9488')   # 青绿色
            self._set_status(
                f"✓ 重采样完成  |  方法：{method_name}  |  "
                f"目标采样率：{target_rate:g} Hz  |  处理 {changed} 道"
            )
            # 刷新左侧面板（采样率已变）
            self.header_panel.load_stream(self.stream)
            self._replot_current()

        except Exception as e:
            QMessageBox.critical(
                self, "重采样失败",
                f"处理时发生错误：\n\n{str(e)}\n\n"
                "常见原因：\n"
                "① decimate 目标因子必须为整数（≥2）\n"
                "② 数据长度太短，无法支持所选滤波阶数\n"
                "③ 目标采样率超出合理范围"
            )

    def _selected_stream_indices(self) -> list:
        """
        获取当前选中的 trace 索引列表（用于预处理操作）。
        若无选中则返回 [0]（默认第一道）。
        """
        try:
            selected = self.header_panel.tree.selectedItems()
            idxs = [item.data(0, Qt.UserRole) for item in selected
                    if item.data(0, Qt.UserRole) is not None]
            if not idxs and self.stream:
                return [0]
            return idxs
        except Exception:
            return [0] if self.stream else []

    def _current_selected_indices(self):
        """兼容旧调用，委托给 _selected_stream_indices"""
        return self._selected_stream_indices()

    def _process_normalize(self, mode: str = 'max'):
        """归一化：支持最大值 / 峰峰值 / RMS 三种方式（仅选中道）"""
        if not self.stream: return
        idxs = self._selected_stream_indices()
        try:
            for tr in [self.stream[i] for i in idxs]:
                d = tr.data.astype(float)
                if mode == 'max':
                    peak = np.max(np.abs(d))
                    if peak > 0:
                        tr.data = d / peak
                    label = "归一化(Max)"
                elif mode == 'peak_peak':
                    pp = d.max() - d.min()
                    if pp > 0:
                        tr.data = (d - d.min()) / pp * 2 - 1
                    label = "归一化(P-P)"
                else:  # rms
                    rms = np.sqrt(np.mean(d ** 2))
                    if rms > 0:
                        tr.data = d / rms
                    label = "归一化(RMS)"

            self._add_proc_step(label, color=COLORS['accent_green'])
            self._set_status(f"归一化完成：{label}  |  作用于 {len(idxs)} 道")
            self._replot_current()
        except Exception as e:
            QMessageBox.critical(self, "归一化失败", str(e))

    def _replot_current(self):
        """处理完毕后，自动重新绘制当前视图中的波形"""
        sel = self.header_panel.tree.selectedItems()
        # 如果列表中有选中的通道，则只重绘选中的；否则重绘默认的全部通道
        if sel:
            self._plot_selected()
        else:
            self._plot_all()

    # ── 按震中距排列 ───────────────────────────────────────────────────────────
    def _show_distance_sort_dialog(self):
        """弹出震中距排列对话框：支持手动输入事件坐标 + CSV 加载事件/台站信息"""
        if not self.stream:
            self._set_status("请先打开数据文件")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("按震中距排列波形")
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabBar::tab{{padding:5px 16px;font-size:11px;}}"
            f"QTabBar::tab:selected{{color:{COLORS['accent_blue']};"
            f"border-bottom:2px solid {COLORS['accent_blue']};}}"
        )

        # ══ Tab 1：手动输入事件坐标 ═══════════════════════════════════════
        tab_manual = QWidget()
        mf = QFormLayout(tab_manual)
        mf.setContentsMargins(12, 12, 12, 12)
        mf.setSpacing(10)

        ev_lat = QDoubleSpinBox(); ev_lat.setRange(-90, 90);   ev_lat.setDecimals(6)
        ev_lon = QDoubleSpinBox(); ev_lon.setRange(-180, 180); ev_lon.setDecimals(6)
        ev_dep = QDoubleSpinBox(); ev_dep.setRange(0, 1000);   ev_dep.setDecimals(3)
        ev_dep.setSuffix("  km")
        for sp in [ev_lat, ev_lon, ev_dep]: sp.setFixedHeight(26)
        mf.addRow("震源纬度 (°N)：", ev_lat)
        mf.addRow("震源经度 (°E)：", ev_lon)
        mf.addRow("震源深度 (km)：", ev_dep)

        # 台站坐标来源（手动模式）
        sta_src_lbl = QLabel("台站坐标来源（手动模式）：")
        sta_src_lbl.setStyleSheet(
            f"font-weight:600;color:{COLORS['text_secondary']};font-size:10px;")
        mf.addRow(sta_src_lbl)

        sta_csv_row = QHBoxLayout()
        self._sta_csv_path_man = QLineEdit()
        self._sta_csv_path_man.setPlaceholderText(
            "可选：加载台站坐标 CSV（network,station,longitude,latitude,elevation）")
        self._sta_csv_path_man.setFixedHeight(24)
        sta_browse_btn = QPushButton("浏览…")
        sta_browse_btn.setFixedSize(60, 24)
        sta_browse_btn.clicked.connect(
            lambda: self._browse_csv(self._sta_csv_path_man, "台站坐标"))
        sta_csv_row.addWidget(self._sta_csv_path_man)
        sta_csv_row.addWidget(sta_browse_btn)
        mf.addRow("台站 CSV：", sta_csv_row)

        sta_note = QLabel(
            "如不加载台站 CSV，将尝试从波形头段 stats.sac.stla / stlo 或\n"
            "stats.coordinates 读取台站坐标。"
        )
        sta_note.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:9px;")
        mf.addRow(sta_note)
        tabs.addTab(tab_manual, "📍  手动输入震源")

        # ══ Tab 2：从 CSV 加载事件 + 台站信息 ════════════════════════════
        tab_csv = QWidget()
        cf = QFormLayout(tab_csv)
        cf.setContentsMargins(12, 12, 12, 12)
        cf.setSpacing(10)

        ev_csv_row = QHBoxLayout()
        self._ev_csv_path = QLineEdit()
        self._ev_csv_path.setPlaceholderText(
            "事件 CSV（id,time,latitude,longitude,depth）")
        self._ev_csv_path.setFixedHeight(24)
        ev_browse_btn = QPushButton("浏览…")
        ev_browse_btn.setFixedSize(60, 24)
        ev_browse_btn.clicked.connect(
            lambda: self._browse_csv(self._ev_csv_path, "事件信息"))
        ev_csv_row.addWidget(self._ev_csv_path)
        ev_csv_row.addWidget(ev_browse_btn)
        cf.addRow("事件 CSV：", ev_csv_row)

        sta_csv_row2 = QHBoxLayout()
        self._sta_csv_path = QLineEdit()
        self._sta_csv_path.setPlaceholderText(
            "台站 CSV（network,station,longitude,latitude,elevation）")
        self._sta_csv_path.setFixedHeight(24)
        sta_browse_btn2 = QPushButton("浏览…")
        sta_browse_btn2.setFixedSize(60, 24)
        sta_browse_btn2.clicked.connect(
            lambda: self._browse_csv(self._sta_csv_path, "台站坐标"))
        sta_csv_row2.addWidget(self._sta_csv_path)
        sta_csv_row2.addWidget(sta_browse_btn2)
        cf.addRow("台站 CSV：", sta_csv_row2)

        # 事件选择（多事件时）
        self._ev_select_combo = QComboBox()
        self._ev_select_combo.setFixedHeight(24)
        self._ev_select_combo.addItem("（请先加载事件 CSV）")
        cf.addRow("选择事件：", self._ev_select_combo)
        self._ev_csv_events = []

        def _load_ev_csv():
            p = self._ev_csv_path.text().strip()
            if not p or not os.path.isfile(p):
                return
            try:
                evs = self._parse_event_csv(p)
                self._ev_csv_events = evs
                self._ev_select_combo.clear()
                for ev in evs:
                    self._ev_select_combo.addItem(
                        f"{ev.get('id','?')}  {str(ev.get('time',''))[:19]}  "
                        f"({ev['lat']:.4f}, {ev['lon']:.4f})  "
                        f"dep={ev.get('depth',0):.1f}km",
                        ev
                    )
                self._set_status(f"已加载 {len(evs)} 条事件记录")
            except Exception as e:
                QMessageBox.warning(dlg, "加载失败", f"事件 CSV 解析失败：\n{e}")

        ev_browse_btn.clicked.connect(lambda: (
            _load_ev_csv() if self._ev_csv_path.text() else None))
        load_ev_btn = QPushButton("加载事件列表")
        load_ev_btn.setFixedHeight(24)
        load_ev_btn.clicked.connect(_load_ev_csv)
        cf.addRow("", load_ev_btn)

        tabs.addTab(tab_csv, "📂  从 CSV 加载")

        root.addWidget(tabs)

        # ── 排序选项 ──────────────────────────────────────────────────────
        sort_box = QGroupBox("排列选项")
        sort_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:8px;padding-top:4px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 5px;font-weight:600;}}"
        )
        sf = QFormLayout(sort_box)
        sf.setSpacing(8); sf.setContentsMargins(12, 10, 12, 10)

        order_combo = QComboBox(); order_combo.setFixedHeight(24)
        order_combo.addItem("由近到远（升序）", "asc")
        order_combo.addItem("由远到近（降序）", "desc")
        sf.addRow("排列方向：", order_combo)

        dist_unit = QComboBox(); dist_unit.setFixedHeight(24)
        dist_unit.addItem("角距离 (°)", "deg")
        dist_unit.addItem("千米 (km)",  "km")
        sf.addRow("距离单位：", dist_unit)

        # ── 分量选择（直接筛选已有分量，R/T 需预先旋转好）──────────────
        comp_combo = QComboBox(); comp_combo.setFixedHeight(24)
        comp_combo.addItem("全部通道（不筛选）", "all")
        comp_combo.addItem("Z 分量（垂直）",    "Z")
        comp_combo.addItem("N 分量（南北）",    "N")
        comp_combo.addItem("E 分量（东西）",    "E")
        comp_combo.addItem("R 分量（径向）",    "R")
        comp_combo.addItem("T 分量（切向）",    "T")
        sf.addRow("显示分量：", comp_combo)

        # 实时统计当前 stream 中各分量数量，帮助用户确认
        comp_counts = {}
        for tr in self.stream:
            c = tr.stats.channel[-1].upper() if tr.stats.channel else '?'
            comp_counts[c] = comp_counts.get(c, 0) + 1
        comp_hint = QLabel(
            "当前数据：" + "  ".join(
                f"{k}×{v}" for k, v in sorted(comp_counts.items())))
        comp_hint.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:9px;font-family:Consolas;")
        sf.addRow("", comp_hint)

        # ── 时间对齐方式 ──────────────────────────────────────────────────
        align_combo = QComboBox(); align_combo.setFixedHeight(24)
        align_combo.addItem("无对齐（以各道数据起始时刻对齐）",  "none")
        align_combo.addItem("发震时刻对齐（需事件 time 字段）",  "origin")
        align_combo.addItem("P波到时对齐",                       "p_pick")
        sf.addRow("时间对齐：", align_combo)

        # 震相文件（P波对齐时使用）
        picks_file_row = QHBoxLayout()
        picks_file_edit = QLineEdit()
        picks_file_edit.setPlaceholderText(
            "震相 CSV（station,phase,time）— 留空则用已导入震相或 SAC.a")
        picks_file_edit.setFixedHeight(24)
        picks_browse_btn = QPushButton("浏览…")
        picks_browse_btn.setFixedSize(60, 24)
        picks_browse_btn.clicked.connect(
            lambda: self._browse_csv(picks_file_edit, "震相"))
        picks_file_row.addWidget(picks_file_edit)
        picks_file_row.addWidget(picks_browse_btn)
        picks_file_widget = QWidget(); picks_file_widget.setLayout(picks_file_row)
        picks_file_widget.setVisible(False)
        sf.addRow("震相文件：", picks_file_widget)

        mark_s_check = QCheckBox("同时将 S 波到时标注到波形上")
        mark_s_check.setChecked(True)
        mark_s_check.setVisible(False)
        sf.addRow("", mark_s_check)

        # 对齐说明标签
        align_note = QLabel()
        align_note.setWordWrap(True)
        align_note.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:9px;font-family:Consolas;")
        sf.addRow("", align_note)

        def _on_align_change():
            mode = align_combo.currentData()
            is_p = mode == 'p_pick'
            picks_file_widget.setVisible(is_p)
            mark_s_check.setVisible(is_p)
            if mode == 'none':
                align_note.setText("各道从自身起始时刻 t=0 开始绘制。")
            elif mode == 'origin':
                align_note.setText(
                    "以事件发震时刻为 t=0。\n"
                    "手动模式：需在此对话框下方输入发震时刻（UTC）。\n"
                    "CSV 模式：自动使用 time 字段。")
            else:
                align_note.setText(
                    "P波到时为 t=0。优先级：\n"
                    "① 上方指定的震相文件 → ② 已用「读取震相」导入的震相\n"
                    "③ SAC 头段 a 标记 → ④ 当前手动拾取")
        align_combo.currentIndexChanged.connect(_on_align_change)
        _on_align_change()

        show_dist_check = QCheckBox("在波形标签中显示震中距")
        show_dist_check.setChecked(True)
        sf.addRow("", show_dist_check)

        root.addWidget(sort_box)

        # ── 发震时刻（手动模式 + 发震时刻对齐时使用）────────────────────
        origin_box = QGroupBox('发震时刻（手动输入模式 + "发震时刻对齐" 时生效）')
        origin_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:4px;padding-top:4px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 5px;font-weight:600;}}"
        )
        of = QFormLayout(origin_box)
        of.setSpacing(6); of.setContentsMargins(12, 8, 12, 8)
        origin_edit = QLineEdit()
        origin_edit.setPlaceholderText("例：2023-02-06T01:17:35.0  （UTC，留空则跳过对齐）")
        origin_edit.setFixedHeight(24)
        of.addRow("发震时刻 (UTC)：", origin_edit)
        root.addWidget(origin_box)

        # ── 结果预览 ─────────────────────────────────────────────────────
        preview_lbl = QLabel("确定后将计算各台站震中距并重新排列波形")
        preview_lbl.setWordWrap(True)
        preview_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;"
            f"background:{COLORS['bg_header']};border-radius:3px;padding:5px 8px;")
        root.addWidget(preview_lbl)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        # ── 取参数 ────────────────────────────────────────────────────────
        ascending    = order_combo.currentData() == 'asc'
        use_km       = dist_unit.currentData() == 'km'
        show_dist    = show_dist_check.isChecked()
        component    = comp_combo.currentData()
        align_mode   = align_combo.currentData()
        origin_str   = origin_edit.text().strip()
        picks_csv    = picks_file_edit.text().strip()
        do_mark_s    = mark_s_check.isChecked()
        sta_csv_path = ''

        if tabs.currentIndex() == 0:
            # 手动模式
            ev_info = {'lat': ev_lat.value(), 'lon': ev_lon.value(),
                       'depth': ev_dep.value(), 'id': 'manual',
                       'time': origin_str}
            sta_csv_path = self._sta_csv_path_man.text().strip()
        else:
            # CSV 模式
            idx = self._ev_select_combo.currentIndex()
            if not self._ev_csv_events or idx < 0:
                QMessageBox.warning(self, "未选择事件", "请先加载事件 CSV 并选择事件")
                return
            ev_info = self._ev_csv_events[idx]
            sta_csv_path = self._sta_csv_path.text().strip()

        # 加载台站坐标表
        sta_coords = {}
        if sta_csv_path and os.path.isfile(sta_csv_path):
            try:
                sta_coords = self._parse_station_csv(sta_csv_path)
                self._sta_coords_cache = sta_coords   # 缓存供互相关窗口使用
            except Exception as e:
                QMessageBox.warning(self, "台站 CSV 错误", f"无法解析台站坐标：\n{e}")

        self._apply_distance_sort(ev_info, sta_coords, ascending, use_km,
                                  show_dist, component=component,
                                  align_mode=align_mode,
                                  picks_csv=picks_csv,
                                  mark_s=do_mark_s)

    def _browse_csv(self, line_edit, label: str):
        p, _ = QFileDialog.getOpenFileName(
            self, f"选择{label} CSV 文件",
            self._last_dir or "",
            "CSV 文件 (*.csv);;文本文件 (*.txt);;所有文件 (*.*)"
        )
        if p:
            line_edit.setText(p)
            self._last_dir = os.path.dirname(p)

    # ── CSV 解析 ────────────────────────────────────────────────────────────
    def _parse_event_csv(self, path: str) -> list:
        """
        解析事件信息 CSV，返回：
          [{'id', 'time', 'lat', 'lon', 'depth'}, ...]
        列名大小写不敏感，支持多种别名。
        """
        import csv
        COL = {
            'id':    ['id', 'event_id', 'evid', 'eventid'],
            'time':  ['time', 'origin_time', 'datetime', 'utctime', 'ot'],
            'lat':   ['latitude', 'lat', 'evlat', 'y'],
            'lon':   ['longitude', 'lon', 'evlon', 'x'],
            'depth': ['depth', 'dep', 'evdep', 'depth_km'],
        }
        results = []
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            col_map = {}
            for row in reader:
                if not col_map:
                    lk = {k.lower().strip(): k for k in row.keys()}
                    for field, aliases in COL.items():
                        for a in aliases:
                            if a in lk:
                                col_map[field] = lk[a]; break
                def g(k): return row.get(col_map.get(k, ''), '').strip()
                try:
                    results.append({
                        'id':    g('id') or f"ev{len(results)+1}",
                        'time':  g('time'),
                        'lat':   float(g('lat')),
                        'lon':   float(g('lon')),
                        'depth': float(g('depth') or 0),
                    })
                except (ValueError, TypeError):
                    continue
        return results

    def _parse_station_csv(self, path: str) -> dict:
        """
        解析台站坐标 CSV，返回：
          {'STA': {'lat': ..., 'lon': ..., 'elev': ...}, ...}
        键为台站代码（大写），支持 network.station 格式。
        """
        import csv
        COL = {
            'net':  ['network', 'net', 'nw'],
            'sta':  ['station', 'sta', 'stat'],
            'lat':  ['latitude', 'lat', 'stla', 'y'],
            'lon':  ['longitude', 'lon', 'stlo', 'x'],
            'elev': ['elevation', 'elev', 'alt', 'stel', 'z'],
        }
        results = {}
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            col_map = {}
            for row in reader:
                if not col_map:
                    lk = {k.lower().strip(): k for k in row.keys()}
                    for field, aliases in COL.items():
                        for a in aliases:
                            if a in lk:
                                col_map[field] = lk[a]; break
                def g(k): return row.get(col_map.get(k, ''), '').strip()
                try:
                    sta  = g('sta').upper()
                    net  = g('net').upper()
                    lat  = float(g('lat'))
                    lon  = float(g('lon'))
                    elev = float(g('elev') or 0)
                    if not sta:
                        continue
                    results[sta] = {'lat': lat, 'lon': lon, 'elev': elev, 'net': net}
                    if net:   # 也存 NET.STA 键
                        results[f"{net}.{sta}"] = results[sta]
                except (ValueError, TypeError):
                    continue
        return results

    # ── 距离计算 + 重排 ─────────────────────────────────────────────────────
    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        """Haversine 公式，返回千米"""
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _km_to_deg(km: float) -> float:
        import math
        return math.degrees(km / 6371.0)

    def _get_sta_coords(self, tr, sta_coords: dict):
        """
        按优先级获取台站坐标：
        1. sta_coords CSV 表（STA / NET.STA 键）
        2. SAC 头段 stats.sac.stla / stlo
        3. stats.coordinates['latitude'] / ['longitude']
        返回 (lat, lon) 或 None
        """
        s   = tr.stats
        sta = s.station.upper()
        net = s.network.upper()

        for key in [f"{net}.{sta}", sta]:
            if key in sta_coords:
                c = sta_coords[key]
                return c['lat'], c['lon']

        # SAC 头段
        if hasattr(s, 'sac'):
            stla = getattr(s.sac, 'stla', None)
            stlo = getattr(s.sac, 'stlo', None)
            if stla is not None and stlo is not None:
                if not (np.isnan(float(stla)) or np.isnan(float(stlo))):
                    return float(stla), float(stlo)

        # ObsPy Inventory 附加坐标
        coords = getattr(s, 'coordinates', None)
        if coords:
            return coords.get('latitude'), coords.get('longitude')

        return None

    def _apply_distance_sort(self, ev_info: dict, sta_coords: dict,
                             ascending: bool, use_km: bool, show_dist: bool,
                             component: str = 'all',
                             align_mode: str = 'none',
                             picks_csv: str = '',
                             mark_s: bool = True):
        """
        计算各道震中距，按分量筛选，按时间对齐，然后重绘。
        若提供 picks_csv，解析后用于 P 波对齐并（可选）标注 S 波。
        """
        if not self.stream:
            return

        from obspy import UTCDateTime
        from obspy import Stream as _Stream

        ev_lat = ev_info['lat']
        ev_lon = ev_info['lon']

        # ── 1. 解析震相文件（优先于 canvas 已导入的震相）─────────────────
        # file_picks: {station_upper: {phase_upper: UTCDateTime}}
        file_picks = {}   # e.g. {'EHS23': {'P': UTCDateTime(...), 'S': UTCDateTime(...)}}
        if picks_csv and os.path.isfile(picks_csv):
            try:
                raw = self._parse_picks_file(picks_csv)
                for r in raw:
                    sta   = r.get('station', '').upper()
                    phase = r.get('phase',   '').upper()
                    t     = r.get('time')
                    if not sta or not phase or t is None:
                        continue
                    try:
                        t_utc = UTCDateTime(t) if not isinstance(t, UTCDateTime) else t
                    except Exception:
                        continue
                    file_picks.setdefault(sta, {})[phase] = t_utc
            except Exception as e:
                QMessageBox.warning(self, "震相文件读取失败",
                                    f"无法解析震相文件：\n{picks_csv}\n\n{e}")

        # ── 2. 分量筛选 ───────────────────────────────────────────────────
        work_traces = []
        orig_map    = []
        for i, tr in enumerate(self.stream):
            last = tr.stats.channel[-1].upper() if tr.stats.channel else ''
            if component == 'all' or last == component:
                work_traces.append(tr)
                orig_map.append(i)

        if not work_traces:
            QMessageBox.warning(self, "无匹配通道",
                f"当前数据中未找到 {component} 分量的通道。\n"
                '请检查通道名或改为"全部通道"。')
            return

        # ── 3. 计算震中距 ────────────────────────────────────────────────
        trace_dists = []
        missing     = []
        for wi, tr in enumerate(work_traces):
            coords = self._get_sta_coords(tr, sta_coords)
            if coords is None or coords[0] is None:
                missing.append(tr.stats.station)
                trace_dists.append((wi, float('inf'), '???', tr))
                continue
            sta_lat, sta_lon = coords
            km  = self._haversine_km(ev_lat, ev_lon, sta_lat, sta_lon)
            deg = self._km_to_deg(km)
            val, dist_str = (km, f"{km:.1f} km") if use_km else (deg, f"{deg:.3f}°")
            trace_dists.append((wi, val, dist_str, tr))

        trace_dists.sort(key=lambda x: x[1], reverse=not ascending)

        # ── 4. 时间对齐：计算每道偏移量（秒）────────────────────────────
        origin_utc = None
        if align_mode == 'origin':
            t_raw = ev_info.get('time', '')
            if t_raw:
                try:
                    origin_utc = UTCDateTime(str(t_raw))
                except Exception:
                    align_mode = 'none'

        # canvas 中已导入的 P 到时（station → t_rel_sec）备用
        canvas_p = {}
        if align_mode == 'p_pick':
            for (tidx, phase, t_rel) in self.canvas._imported_picks_data:
                if phase.upper().startswith('P') and tidx < len(self.stream):
                    sta = self.stream[tidx].stats.station.upper()
                    if sta not in canvas_p:
                        canvas_p[sta] = float(t_rel)

        def _get_offset(tr) -> float:
            """返回该道时间轴的偏移秒数：t_display = tr.times() - offset"""
            if align_mode == 'origin' and origin_utc is not None:
                return float(tr.stats.starttime - origin_utc)
            if align_mode == 'p_pick':
                sta = tr.stats.station.upper()
                # 优先：本次指定的震相文件
                if sta in file_picks and 'P' in file_picks[sta]:
                    return float(file_picks[sta]['P'] - tr.stats.starttime)
                # 次选：canvas 已导入震相
                if sta in canvas_p:
                    return canvas_p[sta]
                # 再次：SAC 头段 a 标记
                if hasattr(tr.stats, 'sac'):
                    a = getattr(tr.stats.sac, 'a', -12345)
                    if a not in (-12345, None):
                        try: return float(a)
                        except Exception: pass
                # 最后：当前手动 P 拾取（全局）
                picks = self.canvas.get_picks()
                if 'P' in picks:
                    return picks['P'][0]
            return 0.0

        # ── 5. 组装绘图参数 ──────────────────────────────────────────────
        tmp_stream          = _Stream(work_traces)
        sorted_work_indices = [td[0] for td in trace_dists]

        dist_labels_map  = {}
        time_offsets_map = {}
        for wi, val, dist_str, tr in trace_dists:
            dist_labels_map[wi]  = dist_str
            time_offsets_map[wi] = _get_offset(tr)

        if missing:
            self._set_status(
                f"⚠ 未找到坐标：{', '.join(sorted(set(missing)))} — 排在末尾")

        self.canvas.plot_stream(
            tmp_stream,
            sorted_work_indices,
            dist_labels  = dist_labels_map  if show_dist else None,
            time_offsets = time_offsets_map,
        )

        # ── 6. 将震相文件中的到时标注到画布上 ───────────────────────────
        # 同时标注 P（作为参考线）和 S（以及文件中所有其他震相）
        if file_picks:
            picks_for_canvas = []
            for render_i, (wi, val, dist_str, tr) in enumerate(trace_dists):
                sta = tr.stats.station.upper()
                offset = time_offsets_map[wi]
                if sta not in file_picks:
                    continue
                for phase, t_utc in file_picks[sta].items():
                    # 跳过 S 波标注（若用户未勾选）
                    if phase.startswith('S') and not mark_s:
                        continue
                    t_rel = float(t_utc - tr.stats.starttime) - offset
                    picks_for_canvas.append((wi, phase, t_rel))
            if picks_for_canvas:
                self.canvas.add_imported_picks(picks_for_canvas)
        elif canvas_p and align_mode == 'p_pick':
            # 没有新文件，但 canvas 已有导入震相——按偏移后坐标重新标注
            updated = []
            for (tidx, phase, t_rel) in self.canvas._imported_picks_data:
                if tidx >= len(self.stream):
                    continue
                sta = self.stream[tidx].stats.station.upper()
                # 找到对应的 work trace 索引
                for wi, tr in enumerate(work_traces):
                    if tr.stats.station.upper() == sta:
                        offset = time_offsets_map.get(wi, 0.0)
                        updated.append((wi, phase, t_rel - offset))
                        break
            if updated:
                self.canvas.add_imported_picks(updated)

        # ── 7. 更新标题与状态栏 ──────────────────────────────────────────
        comp_str  = component if component != 'all' else '全部'
        dir_str   = '由近到远' if ascending else '由远到近'
        align_str = {'none': '起始对齐', 'origin': '发震时刻对齐',
                     'p_pick': 'P波对齐'}[align_mode]
        n_ok = sum(1 for _, d, _, _ in trace_dists if d != float('inf'))
        near = trace_dists[0][2] if trace_dists else '—'
        far  = trace_dists[n_ok-1][2] if n_ok else '—'
        picks_note = (f"  |  已标注 {len(file_picks)} 台站震相"
                      if file_picks else '')
        self.canvas_title.setText(
            f"震中距排列  ·  {comp_str}分量  ·  {dir_str}  ·  {align_str}"
            f"  — 震源 ({ev_lat:.4f}°N, {ev_lon:.4f}°E)"
        )
        self._set_status(
            f"✓ 震中距排列  |  {len(work_traces)} 道 {comp_str} 分量  |  "
            f"{align_str}  |  最近 {near}  最远 {far}{picks_note}"
        )

    # ── 导出图片 ───────────────────────────────────────────────────────────
    def export_figure(self):
        if not self.stream:
            QMessageBox.information(self, "提示", "请先加载数据文件并绘制波形")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出波形图", "",
            "PNG 图像 (*.png);;SVG 矢量图 (*.svg);;PDF 文档 (*.pdf)"
        )
        if path:
            try:
                self.canvas.fig.savefig(
                    path, dpi=200, bbox_inches='tight',
                    facecolor=COLORS['bg_card']
                )
                self._set_status(f"波形图已导出：{os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    # ── 波形文件保存 ───────────────────────────────────────────────────────────
    def save_waveform(self):
        """弹出保存波形对话框：选择通道、格式、编码，然后写文件"""
        if not self.stream:
            QMessageBox.information(self, "提示", "请先加载数据文件")
            return

        from PyQt5.QtWidgets import (QListWidget, QListWidgetItem,
                                     QAbstractItemView, QCheckBox)

        dlg = QDialog(self)
        dlg.setWindowTitle("保存波形文件")
        dlg.setMinimumWidth(520)
        dlg.setMinimumHeight(540)
        dlg.setStyleSheet(self.styleSheet())
        root = QVBoxLayout(dlg)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 14)

        # ── 标题提示 ─────────────────────────────────────────────────────
        hint = QLabel(
            "选择要保存的通道（可多选），所有选中通道合并为一个 Stream 写入同一文件。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color:{COLORS['text_secondary']};font-size:11px;"
            f"background:{COLORS['bg_deep']};border-radius:4px;padding:6px 8px;"
        )
        root.addWidget(hint)

        # ── 通道列表 ─────────────────────────────────────────────────────
        ch_label = QLabel("通道列表（勾选要保存的通道）：")
        ch_label.setStyleSheet(f"font-weight:600;color:{COLORS['text_primary']};font-size:11px;")
        root.addWidget(ch_label)

        ch_list = QListWidget()
        ch_list.setSelectionMode(QAbstractItemView.NoSelection)
        ch_list.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_card']};"
            f"border:1px solid {COLORS['border_bright']};border-radius:6px;}}"
            f"QListWidget::item{{padding:5px 8px;border-bottom:1px solid {COLORS['border']};}}"
            f"QListWidget::item:hover{{background:{COLORS['bg_header']};}}"
        )
        ch_list.setFixedHeight(min(200, 30 + len(self.stream) * 32))

        # 预选：当前左侧面板选中的道
        panel_sel_indices = set()
        for item in self.header_panel.tree.selectedItems():
            idx = item.data(0, Qt.UserRole)
            if idx is not None:
                panel_sel_indices.add(idx)

        for i, tr in enumerate(self.stream):
            s = tr.stats
            label = (f"{s.network}.{s.station}.{s.location or '--'}.{s.channel}"
                     f"   {s.sampling_rate:.0f} Hz   {s.npts:,} pts"
                     f"   {str(s.starttime)[:19].replace('T',' ')}")
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # 若面板中有预选则默认勾上，否则全部勾上（无预选时）
            checked = (i in panel_sel_indices) if panel_sel_indices else True
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            ch_list.addItem(item)

        root.addWidget(ch_list)

        # 全选 / 反选 快捷按钮
        sel_row = QHBoxLayout()
        def _set_all(state):
            for r in range(ch_list.count()):
                ch_list.item(r).setCheckState(state)
        btn_all  = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_inv  = QPushButton("反选")
        for b in (btn_all, btn_none, btn_inv):
            b.setFixedHeight(24)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_header']};"
                f"color:{COLORS['text_secondary']};border:1px solid {COLORS['border_bright']};"
                f"border-radius:4px;font-size:10px;padding:0 8px;}}"
                f"QPushButton:hover{{border-color:{COLORS['accent_blue']};"
                f"color:{COLORS['accent_blue']};}}"
            )
        btn_all.clicked.connect(lambda: _set_all(Qt.Checked))
        btn_none.clicked.connect(lambda: _set_all(Qt.Unchecked))
        btn_inv.clicked.connect(lambda: [
            ch_list.item(r).setCheckState(
                Qt.Unchecked if ch_list.item(r).checkState() == Qt.Checked
                else Qt.Checked)
            for r in range(ch_list.count())
        ])
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addWidget(btn_inv)
        sel_row.addStretch()
        root.addLayout(sel_row)

        # ── 格式选择 ─────────────────────────────────────────────────────
        fmt_box = QGroupBox("输出格式与编码")
        fmt_box.setStyleSheet(
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:10px;padding-top:6px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;subcontrol-position:top left;"
            f"padding:0 6px;font-weight:600;letter-spacing:1px;}}"
        )
        ff = QFormLayout(fmt_box)
        ff.setSpacing(8)

        fmt_combo = QComboBox()
        # (显示名, obspy format string, 默认扩展名, 支持的编码列表)
        FORMATS = [
            ("MiniSEED (.mseed)",          "MSEED",    ".mseed",  ["STEIM2", "STEIM1", "INT32", "INT24", "INT16", "FLOAT32", "FLOAT64"]),
            ("SAC (.sac)",                 "SAC",      ".sac",    ["—（SAC 固定单精度浮点）"]),
            ("GSE2 (.gse2)",               "GSE2",     ".gse2",   ["CM6", "INT"]),
            ("SEGY (.segy)",               "SEGY",     ".segy",   ["—（SEGY 默认）"]),
            ("ASCII / TSPAIR (.ascii)",    "TSPAIR",   ".ascii",  ["—（文本格式）"]),
        ]
        fmt_combo.addItems([f[0] for f in FORMATS])
        fmt_combo.setFixedHeight(26)
        ff.addRow("文件格式：", fmt_combo)

        enc_combo = QComboBox()
        enc_combo.setFixedHeight(26)
        enc_combo.setToolTip("数据编码方式（仅 MiniSEED / GSE2 有效）")

        def _update_enc(idx):
            enc_combo.clear()
            enc_list = FORMATS[idx][3]
            enc_combo.addItems(enc_list)
            enc_combo.setEnabled(not enc_list[0].startswith("—"))

        fmt_combo.currentIndexChanged.connect(_update_enc)
        _update_enc(0)   # 初始化
        ff.addRow("数据编码：", enc_combo)

        root.addWidget(fmt_box)

        # ── 预处理状态提示 ────────────────────────────────────────────────
        if self._proc_history:
            proc_lbl = QLabel(
                "⚡ 当前波形已经过预处理：" + "  →  ".join(self._proc_history)
            )
            proc_lbl.setWordWrap(True)
            proc_lbl.setStyleSheet(
                f"color:{COLORS['accent_green']};font-size:10px;"
                f"background:#F0FDF4;border:1px solid #86EFAC;"
                f"border-radius:4px;padding:5px 8px;"
            )
            root.addWidget(proc_lbl)

        # ── 按钮 ─────────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel,
                                Qt.Horizontal, dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        root.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        # 收集勾选的道索引
        selected_indices = [r for r in range(ch_list.count())
                            if ch_list.item(r).checkState() == Qt.Checked]
        if not selected_indices:
            QMessageBox.warning(self, "未选择通道", "请至少勾选一条通道再保存。")
            return

        fmt_idx  = fmt_combo.currentIndex()
        fmt_name = FORMATS[fmt_idx][0].split()[0]   # 简短名称用于建议文件名
        fmt_code = FORMATS[fmt_idx][1]
        ext      = FORMATS[fmt_idx][2]
        enc      = enc_combo.currentText() if not enc_combo.currentText().startswith("—") else None

        # 建议文件名
        tr0    = self.stream[selected_indices[0]]
        s0     = tr0.stats
        t_str  = str(s0.starttime)[:10].replace('-', '')
        suggest = f"{s0.network}.{s0.station}_{t_str}{ext}"

        path, _ = QFileDialog.getSaveFileName(
            self, "保存波形文件", os.path.join(self._last_dir, suggest),
            f"{fmt_name} (*{ext});;所有文件 (*.*)"
        )
        if not path:
            return

        self._do_save_waveform(path, selected_indices, fmt_code, enc)

    def _do_save_waveform(self, path: str, indices: list, fmt_code: str, encoding):
        """执行波形文件写入，自动处理数据类型与编码的兼容性"""
        try:
            from obspy.core import Stream as ObspyStream
            import numpy as np

            out_stream = ObspyStream()
            for i in indices:
                out_stream += self.stream[i].copy()

            kwargs = {'format': fmt_code}

            if fmt_code == 'MSEED':
                # 根据用户选择的编码，自动转换 dtype
                INT_ENCODINGS   = {'STEIM2', 'STEIM1', 'INT32', 'INT24', 'INT16'}
                FLOAT_ENCODINGS = {'FLOAT32', 'FLOAT64'}

                # 如果用户没有指定编码，根据数据类型自动选择
                if not encoding:
                    sample = out_stream[0].data
                    if np.issubdtype(sample.dtype, np.integer):
                        encoding = 'STEIM2'
                    else:
                        encoding = 'FLOAT32'

                if encoding in INT_ENCODINGS:
                    # 整数编码：需要 int32
                    for tr in out_stream:
                        if not np.issubdtype(tr.data.dtype, np.integer):
                            # 浮点→整数：按比例缩放到 int32 范围，保留相对幅值
                            d = tr.data.astype(np.float64)
                            peak = np.max(np.abs(d))
                            if peak > 0:
                                # 缩放到 int32 的 80% 动态范围，保留 headroom
                                scale = (2**31 - 1) * 0.8 / peak
                                d = d * scale
                            tr.data = np.round(d).astype(np.int32)
                elif encoding in FLOAT_ENCODINGS:
                    # 浮点编码：转换为对应精度
                    target = np.float32 if encoding == 'FLOAT32' else np.float64
                    for tr in out_stream:
                        tr.data = tr.data.astype(target)

                kwargs['encoding'] = encoding

            elif fmt_code == 'SAC':
                # SAC 固定为 float32
                for tr in out_stream:
                    tr.data = tr.data.astype(np.float32)

            elif fmt_code == 'GSE2':
                enc = encoding or 'CM6'
                if enc == 'CM6':
                    for tr in out_stream:
                        if not np.issubdtype(tr.data.dtype, np.integer):
                            d = tr.data.astype(np.float64)
                            peak = np.max(np.abs(d))
                            if peak > 0:
                                d = d * ((2**23 - 1) * 0.8 / peak)
                            tr.data = np.round(d).astype(np.int32)

            out_stream.write(path, **kwargs)

            n = len(out_stream)
            enc_info = f", 编码: {encoding}" if encoding else ""
            self._last_dir = os.path.dirname(path)
            self._set_status(
                f"✓ 已保存 {n} 条通道 → {os.path.basename(path)}  [{fmt_code}{enc_info}]"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "保存失败",
                f"写入文件时发生错误：\n\n{str(e)}\n\n"
                "建议：\n"
                "① 如数据经过浮点预处理，可尝试选择 FLOAT32 或 FLOAT64 编码\n"
                "② SAC 格式不支持多道混合采样率\n"
                "③ 目标路径无写入权限"
            )

    # ── 拖拽支持（多文件 + 文件夹）─────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
        if paths:
            self._last_dir = os.path.dirname(paths[0])
            self._load_paths(paths)

    # ── 帮助 / 关于 ────────────────────────────────────────────────────────
    def _show_preprocess_help(self):
        """Show preprocessing workflow guidance."""
        QMessageBox.information(
            self,
            "预处理说明",
            "预处理建议流程：\n\n"
            "1. 去均值 / 去趋势：先移除基线漂移。\n"
            "2. Taper：在滤波前抑制边缘效应。\n"
            "3. 滤波：按研究目标选择带通、低通或高通。\n"
            "4. 去仪器响应：将 counts 转换到速度/位移/加速度。\n"
            "5. 截窗 / 重采样 / 旋转：用于后续分析或批处理。\n\n"
            "提示：所有预处理默认作用于左侧列表当前选中的通道；若无选中，则默认作用于第一道。"
        )

    def _show_pick_help(self):
        """Show phase-picking guidance."""
        QMessageBox.information(
            self,
            "拾取说明",
            "震相拾取用法：\n\n"
            "• 点击工具栏“震相拾取”可展开拾取面板。\n"
            "• 切换到 P 或 S 模式后，在波形上单击即可添加到时标记。\n"
            "• Esc 返回平移模式，右键可撤销当前模式下的最近一次拾取。\n"
            "• 可导出 CSV/TXT，也可导入外部震相文件叠加显示。\n\n"
            "建议：先缩放到清晰的初至附近，再进行精细拾取。"
        )

    def _show_spectrum_help(self):
        """Show spectrum-analysis guidance."""
        QMessageBox.information(
            self,
            "频谱分析说明",
            "频谱分析入口：\n\n"
            "• 振幅频谱：查看当前显示时间窗内的频率成分。\n"
            "• PSD：用于噪声特征、仪器状态和背景能量分析。\n"
            "• 多道互相关：比较参考道与其他道之间的时延和相似度。\n\n"
            "建议：先在主波形窗口缩放到目标时间段，再打开分析窗口，这样分析会继承当前时间窗。"
        )

    def _show_help(self):
        msg = """
<h3 style="color:#00D4FF">SeismoView 使用说明</h3>
<br>
<b>一、文件与视图</b>
<ul>
  <li>工具栏「打开」支持多文件加载，也支持递归扫描整个文件夹。</li>
  <li>左侧通道列表可单选或多选；未选中时，部分处理默认作用于第一道。</li>
  <li>滚轮缩放时间轴，左键拖拽平移，Ctrl+R 可快速重置视图。</li>
</ul>
<br>
<b>二、波形预处理</b>
<ul>
  <li>点击工具栏「🛠 波形预处理」可展开快捷面板，也可通过菜单栏「预处理」访问全部功能。</li>
  <li>支持：去均值、去趋势、归一化、Taper、带通/低通/高通滤波、去仪器响应、截窗、分量旋转、重采样。</li>
  <li>推荐顺序：去均值 → 去趋势 → Taper → 滤波 → 去仪器响应 → 截窗/重采样。</li>
</ul>
<br>
<b>三、波形交互</b>
<ul>
  <li>点击工具栏「🎯 震相拾取」展开拾取面板，切换 P/S 模式后在波形上单击即可添加拾取。</li>
  <li>Esc 返回平移模式；可清除 P、S 或全部拾取。</li>
  <li>支持导出拾取结果，也支持从 CSV / JSON / QuakeML 等文件导入震相。</li>
</ul>
<br>
<b>四、频谱分析</b>
<ul>
  <li>工具栏「📊 频谱分析」提供振幅频谱、PSD 和多道互相关分析。</li>
  <li>分析窗口会继承主波形当前的显示时间窗，便于对局部波段做针对性分析。</li>
</ul>
<br>
<b>五、导出与批处理</b>
<ul>
  <li>「导出」支持保存处理后的波形文件，以及导出当前波形图。</li>
  <li>「批量处理」可对多个文件按统一流程执行预处理并批量输出。</li>
</ul>
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("使用说明")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(msg)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

    def _show_about(self):
        msg = """
<h3 style="color:#00D4FF">SeismoView v1.0</h3>
<p>专业地震波形数据查看器</p>
<p>
  <b>支持格式：</b>MiniSEED · SAC · SEED · GSE2 · SEISAN 及更多<br>
  <b>依赖库：</b>ObsPy · PyQt5 · Matplotlib · NumPy<br>
  <b>目标平台：</b>Windows 10/11 x64
</p>
<p style="color:#7A8BA8; font-size:11px">
  本软件基于 ObsPy 地震数据处理框架构建。<br>
  ObsPy 遵循 LGPL-3.0 许可证。
</p>
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("关于 SeismoView")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(msg)
        dlg.exec_()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
