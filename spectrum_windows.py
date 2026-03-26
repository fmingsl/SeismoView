#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectrum_windows.py
===================
Standalone analysis windows for amplitude spectrum, PSD, and cross-correlation.

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

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFrame
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from config import COLORS
from canvas_spectrum import SpectrumCanvas, PSDCanvas

class SpectrumWindow(QMainWindow):
    """独立振幅频谱分析窗口，含 NavigationToolbar + 控制选项 + 多道选择。"""

    def __init__(self, raw_t, raw_data, raw_meta, active_idx,
                 xmin, xmax, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SeismoView — 振幅频谱分析")
        self.resize(960, 640)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._raw_t = raw_t
        self._raw_data = raw_data
        self._raw_meta = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta) - 1)) if raw_meta else 0
        self._xmin = xmin
        self._xmax = xmax
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(42)
        hdr.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("Spectrum (Amplitude)")
        lbl.setStyleSheet(
            f"color:{COLORS['accent_green']};font-weight:700;"
            f"font-size:12px;letter-spacing:1.5px;"
        )
        hl.addWidget(lbl)
        hl.addStretch()
        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;"
        )
        hl.addWidget(self._info_lbl)
        root.addWidget(hdr)

        ctrl = QWidget()
        ctrl.setFixedHeight(38)
        ctrl.setStyleSheet(
            f"background:{COLORS['bg_deep']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(10, 4, 10, 4)
        cl.setSpacing(6)

        accent = COLORS['accent_green']

        def tog(txt, tip, on=False):
            b = QPushButton(txt)
            b.setCheckable(True)
            b.setChecked(on)
            b.setFixedHeight(26)
            b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};"
                f"border:1px solid {COLORS['border_bright']};border-radius:5px;"
                f"font-size:10px;padding:0 9px;}}"
                f"QPushButton:checked{{background:{accent}22;color:{accent};border-color:{accent};}}"
                f"QPushButton:hover{{border-color:{accent};color:{accent};}}"
            )
            b.toggled.connect(self._refresh)
            return b

        self._welch = tog("Welch", "Welch 平均（降低方差）", True)
        self._logx  = tog("Log-F", "频率轴对数",              True)
        self._db    = tog("dB",    "纵轴 dB re 1 count",      True)
        self._logy  = tog("Log-A", "振幅轴对数（线性模式有效）", False)
        for b in (self._welch, self._logx, self._db, self._logy):
            cl.addWidget(b)

        cl.addWidget(self._vsep())

        if self._raw_meta:
            cl.addWidget(QLabel("Trace"))
            self._ch_combo = QComboBox()
            self._ch_combo.setFixedHeight(26)
            for i, (st, *_) in enumerate(self._raw_meta):
                self._ch_combo.addItem(f"{st.network}.{st.station}.{st.channel}", i)
            self._ch_combo.setCurrentIndex(self._active_idx)
            self._ch_combo.currentIndexChanged.connect(
                lambda ci: self._set_trace(self._ch_combo.itemData(ci))
            )
            cl.addWidget(self._ch_combo)
            cl.addWidget(self._vsep())

        ref_btn = QPushButton("↻  刷新")
        ref_btn.setFixedHeight(26)
        ref_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};color:{accent};"
            f"border:1px solid {accent};border-radius:5px;font-size:10px;"
            f"padding:0 9px;font-weight:600;}}"
            f"QPushButton:hover{{background:{accent}22;}}"
        )
        ref_btn.clicked.connect(self._refresh)
        cl.addWidget(ref_btn)
        cl.addStretch()
        root.addWidget(ctrl)

        self._sc = SpectrumCanvas()
        nav = NavigationToolbar(self._sc, cw)
        nav.setStyleSheet(
            f"QToolBar{{background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};spacing:4px;}}"
        )
        root.addWidget(nav)
        root.addWidget(self._sc)

    @staticmethod
    def _vsep():
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f"background:{COLORS['border_bright']};")
        return f

    def _set_trace(self, idx):
        self._active_idx = idx
        self._refresh()

    def _refresh(self):
        if not self._raw_t:
            return
        self._sc.load_data(self._raw_t, self._raw_data, self._raw_meta)
        self._sc.set_options(
            welch=self._welch.isChecked(),
            log_x=self._logx.isChecked(),
            db=self._db.isChecked(),
            log_y=self._logy.isChecked(),
        )
        self._sc._active_idx = self._active_idx
        self._sc.update_spectra(self._xmin, self._xmax)
        if self._raw_meta and self._active_idx < len(self._raw_meta):
            st = self._raw_meta[self._active_idx][0]
            ch = f"{st.network}.{st.station}.{st.location}.{st.channel}"
        else:
            ch = "?"
        self._info_lbl.setText(
            f"Trace: {ch}   |   Window: {self._xmin:.3f}s - {self._xmax:.3f}s"
        )

    def push_update(self, raw_t, raw_data, raw_meta, active_idx, xmin, xmax):
        self._raw_t = raw_t
        self._raw_data = raw_data
        self._raw_meta = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta) - 1)) if raw_meta else 0
        self._xmin = xmin
        self._xmax = xmax
        if hasattr(self, '_ch_combo'):
            self._ch_combo.blockSignals(True)
            self._ch_combo.setCurrentIndex(self._active_idx)
            self._ch_combo.blockSignals(False)
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# 独立功率谱密度窗口
# ─────────────────────────────────────────────────────────────────────────────
class PSDWindow(QMainWindow):
    """
    独立 PSD 分析窗口。

    与 SpectrumWindow 不同，PSD 窗口内置预处理管道：
      去均值 → 去趋势 → Taper → 去仪器响应（→ 加速度 m/s²）

    仅当完成去仪器响应后，PSD 才转换为 dB rel. 1 (m/s²)²/Hz
    并允许与 Peterson NLNM/NHNM 直接比较。
    """

    def __init__(self, stream, inventory, raw_meta, active_idx,
                 xmin, xmax, data_unit='counts', parent=None):
        super().__init__(parent)
        self.setWindowTitle("SeismoView — PSD 分析")
        self.resize(980, 680)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._stream = stream
        self._inventory = inventory
        self._raw_meta = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta) - 1)) if raw_meta else 0
        self._xmin = xmin
        self._xmax = xmax
        self._data_unit = (data_unit or 'counts').upper()
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(42)
        hdr.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("PSD - 功率谱密度")
        lbl.setStyleSheet(
            f"color:{COLORS['accent_blue']};font-weight:700;"
            f"font-size:12px;letter-spacing:1.5px;"
        )
        hl.addWidget(lbl)
        hl.addStretch()
        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;"
        )
        hl.addWidget(self._info_lbl)
        root.addWidget(hdr)

        self._pc = PSDCanvas()

        prep = QWidget()
        prep.setFixedHeight(44)
        prep.setStyleSheet(
            f"background:{COLORS['bg_deep']};"
            f"border-bottom:1px solid {COLORS['border']};"
        )
        pl = QHBoxLayout(prep)
        pl.setContentsMargins(12, 6, 12, 6)
        pl.setSpacing(10)

        def ck(txt, tip, on=True):
            b = QPushButton(txt)
            b.setCheckable(True)
            b.setChecked(on)
            b.setToolTip(tip)
            b.setFixedHeight(26)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};"
                f"border:1px solid {COLORS['border_bright']};border-radius:5px;font-size:10px;padding:0 9px;}}"
                f"QPushButton:checked{{background:{COLORS['accent_blue']}22;color:{COLORS['accent_blue']};"
                f"border-color:{COLORS['accent_blue']};}}"
            )
            b.toggled.connect(self._refresh)
            return b

        self._ck_demean = ck("Demean", "Remove mean", True)
        self._ck_detrend = ck("Detrend", "Remove linear trend", True)
        self._ck_taper = ck("Taper", "Kept only for FFT fallback", True)
        self._ck_resp = ck("Resp", "Apply response only for counts input", True)
        if self._inventory is None:
            self._ck_resp.setChecked(False)
            self._ck_resp.setEnabled(False)

        for b in (self._ck_demean, self._ck_detrend, self._ck_taper, self._ck_resp):
            pl.addWidget(b)

        pl.addWidget(self._vsep())

        accent = COLORS['accent_blue']

        def tog(txt, tip, on=False):
            b = QPushButton(txt)
            b.setCheckable(True)
            b.setChecked(on)
            b.setFixedHeight(26)
            b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_card']};color:{COLORS['text_secondary']};"
                f"border:1px solid {COLORS['border_bright']};border-radius:5px;font-size:10px;padding:0 9px;}}"
                f"QPushButton:checked{{background:{accent}22;color:{accent};border-color:{accent};}}"
                f"QPushButton:hover{{border-color:{accent};color:{accent};}}"
            )
            b.toggled.connect(self._refresh)
            return b

        self._logx = tog("Log-F", "Log frequency axis", True)
        self._db = tog("dB", "Show PSD in dB", True)
        self._nm = tog("NM", "Overlay Peterson NLNM/NHNM", True)
        self._axis = tog("Period", "Show period axis", False)
        for b in (self._logx, self._db, self._nm, self._axis):
            pl.addWidget(b)

        pl.addWidget(self._vsep())

        if self._raw_meta:
            pl.addWidget(QLabel("Trace"))
            self._ch_combo = QComboBox()
            self._ch_combo.setFixedHeight(26)
            for i, (st, *_) in enumerate(self._raw_meta):
                self._ch_combo.addItem(f"{st.network}.{st.station}.{st.channel}", i)
            self._ch_combo.setCurrentIndex(self._active_idx)
            self._ch_combo.currentIndexChanged.connect(
                lambda ci: self._set_trace(self._ch_combo.itemData(ci))
            )
            pl.addWidget(self._ch_combo)
            pl.addWidget(self._vsep())

        ref_btn = QPushButton("刷新")
        ref_btn.setFixedHeight(26)
        ref_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};color:{accent};"
            f"border:1px solid {accent};border-radius:5px;font-size:10px;"
            f"padding:0 9px;font-weight:600;}}"
            f"QPushButton:hover{{background:{accent}22;}}"
        )
        ref_btn.clicked.connect(self._refresh)
        pl.addWidget(ref_btn)
        pl.addStretch()
        root.addWidget(prep)

        nav = NavigationToolbar(self._pc, cw)
        nav.setStyleSheet(
            f"QToolBar{{background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};spacing:4px;}}"
        )
        root.addWidget(nav)
        root.addWidget(self._pc)

    @staticmethod
    def _vsep():
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f"background:{COLORS['border_bright']};")
        return f

    def _set_trace(self, idx):
        self._active_idx = idx
        self._refresh()

    @staticmethod
    def _noise_window_hint(data):
        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0
        if rms <= 0:
            return "Quiet/flat window"
        crest = float(np.max(np.abs(data)) / rms)
        return ("Likely event-contaminated window"
                if crest > 8.0 else "Likely noise window")

    def _refresh(self):
        required_attrs = (
            '_pc', '_ck_demean', '_ck_detrend', '_ck_taper', '_ck_resp',
            '_logx', '_db', '_nm', '_axis'
        )
        if not all(hasattr(self, name) for name in required_attrs):
            return
        if not self._stream or not self._raw_meta:
            return
        idx = self._active_idx
        if idx >= len(self._stream) or idx >= len(self._raw_meta):
            return

        _, color = self._raw_meta[idx]
        sr = float(self._stream[idx].stats.sampling_rate)

        try:
            from copy import deepcopy
            tr = deepcopy(self._stream[idx])
            t0 = tr.stats.starttime + self._xmin
            t1 = tr.stats.starttime + self._xmax
            tr.trim(t0, t1)
        except Exception as e:
            self._show_err(f"trim failed: {e}")
            return

        if len(tr.data) < 32:
            self._show_err("window too short (< 32 samples)")
            return

        steps = []
        try:
            if self._ck_demean.isChecked():
                tr.detrend('demean')
                steps.append('Demean')
            if self._ck_detrend.isChecked():
                tr.detrend('linear')
                steps.append('Detrend')
            if self._ck_taper.isChecked():
                steps.append('Taper skipped (Welch window)')
        except Exception as e:
            self._show_err(f"preprocess failed: {e}")
            return

        data = tr.data.astype(np.float64, copy=False)
        if not np.isfinite(data).all():
            self._show_err("window contains NaN/Inf samples")
            return

        apply_resp = (
            self._ck_resp.isChecked() and
            self._inventory is not None and
            self._data_unit == 'COUNTS'
        )
        if self._ck_resp.isChecked() and self._data_unit != 'COUNTS':
            steps.append(f"Input={self._data_unit}; skip resp")
        if apply_resp:
            steps.append('resp(freq-domain)')

        steps.append(self._noise_window_hint(data))

        self._pc.set_options(
            log_x=self._logx.isChecked(),
            db=self._db.isChecked(),
            nhnm=self._nm.isChecked(),
            period_axis=self._axis.isChecked(),
        )
        self._pc.plot_psd(
            data, sr, tr.stats, color,
            inventory=self._inventory if apply_resp else None,
            apply_resp=apply_resp,
            preproc_info=' | '.join(steps) if steps else 'Raw',
            data_unit=self._data_unit,
        )

        ch = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
        self._info_lbl.setText(
            f"Trace: {ch}   |   Window: {self._xmin:.2f}s - {self._xmax:.2f}s   |   Input: {self._data_unit}"
        )

    def _show_err(self, msg: str):
        self._pc.fig.clear()
        ax = self._pc.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self._pc.fig.patch.set_facecolor(COLORS['bg_card'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5, 0.5, msg,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9,
            color=COLORS['accent_amber'],
            style='italic',
            wrap=True,
        )
        self._pc.draw()

    def push_update(self, stream, raw_meta, active_idx, xmin, xmax, data_unit=None):
        self._stream = stream
        self._raw_meta = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta) - 1)) if raw_meta else 0
        self._xmin = xmin
        self._xmax = xmax
        if data_unit is not None:
            self._data_unit = (data_unit or 'counts').upper()
        if hasattr(self, '_ch_combo'):
            self._ch_combo.blockSignals(True)
            self._ch_combo.setCurrentIndex(self._active_idx)
            self._ch_combo.blockSignals(False)
        self._refresh()

# ─────────────────────────────────────────────────────────────────────────────
# 头段信息面板
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 多道互相关分析窗口
# ─────────────────────────────────────────────────────────────────────────────

class XCorrWindow(QMainWindow):
    """
    多道互相关分析窗口。

    功能：
    - 选择参考道与目标道（支持多目标道）
    - 对当前主窗口可见时间窗内的数据计算归一化互相关函数
    - 显示互相关波形，标注主峰对应的时延（lag）
    - 若两道台站坐标已知（SAC 头段或台站 CSV），自动计算视速度
    - 支持叠加显示多道互相关曲线
    - 结果可导出为 CSV
    """

    def __init__(self, stream, raw_meta, active_idx, xmin, xmax,
                 sta_coords=None, parent=None):
        """
        Parameters
        ----------
        stream     : ObsPy Stream（完整数据，用于截取时间段）
        raw_meta   : list of (stats, color)，与 canvas._raw_meta 一致
        active_idx : 当前主画布活跃道索引（默认作为参考道）
        xmin/xmax  : 当前可见时间窗（相对秒）
        sta_coords : dict {STA: {'lat': ..., 'lon': ...}} 或 None
        """
        super().__init__(parent)
        self.setWindowTitle("SeismoView — 多道互相关分析")
        self.resize(1020, 680)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._stream     = stream
        self._raw_meta   = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta) - 1)) if raw_meta else 0
        self._xmin       = xmin
        self._xmax       = xmax
        self._sta_coords = sta_coords or {}

        self._build_ui()
        self._refresh()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QListWidget, QListWidgetItem, QAbstractItemView,
            QSplitter, QGroupBox, QCheckBox, QDoubleSpinBox,
            QScrollArea, QGridLayout
        )

        cw = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 标题栏 ────────────────────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(42)
        hdr.setStyleSheet(
            f"background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(14, 0, 14, 0)
        title_lbl = QLabel("⇌  多道互相关分析  (Cross-correlation)")
        title_lbl.setStyleSheet(
            f"color:{COLORS['accent_green']};font-weight:700;"
            f"font-size:12px;letter-spacing:1.5px;")
        hl.addWidget(title_lbl); hl.addStretch()
        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;")
        hl.addWidget(self._info_lbl)
        root.addWidget(hdr)

        # ── 控制栏 ────────────────────────────────────────────────────────
        ctrl = QWidget(); ctrl.setFixedHeight(44)
        ctrl.setStyleSheet(
            f"background:{COLORS['bg_deep']};"
            f"border-bottom:1px solid {COLORS['border']};")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(10, 4, 10, 4)
        cl.setSpacing(8)

        # 参考道选择
        cl.addWidget(QLabel("参考道："))
        self._ref_combo = QComboBox(); self._ref_combo.setFixedHeight(26)
        for i, (st, _) in enumerate(self._raw_meta):
            self._ref_combo.addItem(
                f"{st.network}.{st.station}.{st.channel}", i)
        self._ref_combo.setCurrentIndex(self._active_idx)
        self._ref_combo.currentIndexChanged.connect(self._refresh)
        cl.addWidget(self._ref_combo)

        cl.addWidget(self._vsep())

        # 显示选项
        C = COLORS['accent_green']
        def tog(txt, tip, on=True):
            b = QPushButton(txt); b.setCheckable(True); b.setChecked(on)
            b.setFixedHeight(26); b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_card']};"
                f"color:{COLORS['text_secondary']};"
                f"border:1px solid {COLORS['border_bright']};"
                f"border-radius:5px;font-size:10px;padding:0 9px;}}"
                f"QPushButton:checked{{background:{C}22;color:{C};"
                f"border-color:{C};}}"
                f"QPushButton:hover{{border-color:{C};color:{C};}}")
            b.toggled.connect(self._refresh); return b

        self._norm_btn  = tog("归一化", "将互相关系数归一化到 [-1, 1]", True)
        self._stack_btn = tog("叠加显示", "将所有目标道的互相关曲线叠加到同一子图", False)
        self._vel_btn   = tog("显示视速度", "利用台站间距和峰值时延估算视速度", True)
        for b in (self._norm_btn, self._stack_btn, self._vel_btn):
            cl.addWidget(b)

        cl.addWidget(self._vsep())

        ref_btn = QPushButton("↻ 刷新"); ref_btn.setFixedHeight(26)
        ref_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['bg_card']};color:{C};"
            f"border:1px solid {C};border-radius:5px;"
            f"font-size:10px;padding:0 9px;font-weight:600;}}"
            f"QPushButton:hover{{background:{C}22;}}")
        ref_btn.clicked.connect(self._refresh)
        cl.addWidget(ref_btn)

        exp_btn = QPushButton("📋 导出 CSV"); exp_btn.setFixedHeight(26)
        exp_btn.setStyleSheet(ref_btn.styleSheet())
        exp_btn.clicked.connect(self._export_csv)
        cl.addWidget(exp_btn)

        cl.addStretch()
        root.addWidget(ctrl)

        # ── 主体：左侧道选择 + 右侧图形 ──────────────────────────────────
        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(4)

        # 左侧：目标道列表
        left = QWidget(); left.setMaximumWidth(240); left.setMinimumWidth(160)
        lv = QVBoxLayout(left); lv.setContentsMargins(8, 8, 4, 8); lv.setSpacing(4)
        lv.addWidget(QLabel("目标道（可多选）："))
        self._target_list = QListWidget()
        self._target_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._target_list.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_card']};"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:4px;font-size:10px;font-family:Consolas;}}"
            f"QListWidget::item:selected{{background:{COLORS['accent_green']}22;"
            f"color:{COLORS['accent_green']};}}")
        for i, (st, _) in enumerate(self._raw_meta):
            item = QListWidgetItem(
                f"{st.network}.{st.station}.{st.channel}")
            item.setData(Qt.UserRole, i)
            self._target_list.addItem(item)
        # 默认选中除参考道外的第一道
        if self._target_list.count() > 1:
            for r in range(self._target_list.count()):
                if r != self._active_idx:
                    self._target_list.item(r).setSelected(True)
                    break
        self._target_list.itemSelectionChanged.connect(self._refresh)
        lv.addWidget(self._target_list)

        # 选中/全选辅助按钮
        btn_row = QHBoxLayout(); btn_row.setSpacing(4)
        for txt, fn in [("全选", self._target_list.selectAll),
                        ("全不选", self._target_list.clearSelection)]:
            b = QPushButton(txt); b.setFixedHeight(22)
            b.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_header']};"
                f"color:{COLORS['text_muted']};"
                f"border:1px solid {COLORS['border']};border-radius:3px;"
                f"font-size:9px;padding:0 6px;}}"
                f"QPushButton:hover{{color:{COLORS['text_secondary']};}}")
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        lv.addLayout(btn_row)

        body.addWidget(left)

        # 右侧：matplotlib 画布
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavTB)
        self._fig = Figure(facecolor=COLORS['bg_card'])
        self._canvas = FigureCanvas(self._fig)
        nav = NavTB(self._canvas, right)
        nav.setStyleSheet(
            f"QToolBar{{background:{COLORS['bg_header']};"
            f"border-bottom:1px solid {COLORS['border']};spacing:4px;}}")
        rv.addWidget(nav)
        rv.addWidget(self._canvas)
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)

        root.addWidget(body)

        # ── 底部结果表格 ──────────────────────────────────────────────────
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        self._result_table = QTableWidget(0, 6)
        self._result_table.setFixedHeight(110)
        self._result_table.setHorizontalHeaderLabels(
            ["参考道", "目标道", "峰值时延 (s)", "相关系数", "台站间距 (km)", "视速度 (km/s)"])
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._result_table.setStyleSheet(
            f"QTableWidget{{background:{COLORS['bg_card']};"
            f"alternate-background-color:{COLORS['bg_deep']};"
            f"font-family:Consolas;font-size:10px;"
            f"border:1px solid {COLORS['border_bright']};border-radius:4px;}}"
            f"QHeaderView::section{{background:{COLORS['bg_header']};"
            f"color:{COLORS['text_secondary']};padding:4px;"
            f"font-size:10px;border:none;"
            f"border-bottom:1px solid {COLORS['border_bright']};}}")
        root.addWidget(self._result_table)

        # 保存结果供 CSV 导出
        self._last_results = []

    @staticmethod
    def _vsep():
        f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFixedWidth(1)
        f.setStyleSheet(f"background:{COLORS['border_bright']};"); return f

    # ── 互相关计算 ────────────────────────────────────────────────────────
    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(max(0.0, a)))

    def _get_sta_latlon(self, stats):
        """从 SAC 头段或 sta_coords 字典读取台站坐标，返回 (lat, lon) 或 None"""
        sta = stats.station.upper()
        net = stats.network.upper()
        for key in [f"{net}.{sta}", sta]:
            if key in self._sta_coords:
                c = self._sta_coords[key]
                return c['lat'], c['lon']
        if hasattr(stats, 'sac'):
            stla = getattr(stats.sac, 'stla', None)
            stlo = getattr(stats.sac, 'stlo', None)
            if stla is not None and stlo is not None:
                import math
                try:
                    if not (math.isnan(float(stla)) or math.isnan(float(stlo))):
                        return float(stla), float(stlo)
                except Exception:
                    pass
        coords = getattr(stats, 'coordinates', None)
        if coords:
            return coords.get('latitude'), coords.get('longitude')
        return None

    def _compute_xcorr(self, ref_data, tgt_data, sr, normalize):
        """
        计算归一化或原始互相关函数。
        返回 (lags_sec, xcorr, peak_lag_sec, peak_coeff)
        """
        import numpy as np
        try:
            from scipy.signal import correlate, correlation_lags
            cc = correlate(tgt_data, ref_data, mode='full', method='fft')
            lags = correlation_lags(len(tgt_data), len(ref_data), mode='full')
        except ImportError:
            n = len(ref_data)
            cc = np.correlate(tgt_data, ref_data, mode='full')
            lags = np.arange(-(n - 1), n)

        lags_sec = lags / float(sr)

        if normalize:
            denom = (np.std(ref_data) * np.std(tgt_data) * len(ref_data))
            if denom > 0:
                cc = cc / denom

        peak_idx    = int(np.argmax(np.abs(cc)))
        peak_lag    = float(lags_sec[peak_idx])
        peak_coeff  = float(cc[peak_idx])
        return lags_sec, cc, peak_lag, peak_coeff

    def _trim_segment(self, idx):
        """截取 xmin..xmax 时间段，预处理（demean+detrend），返回 numpy array"""
        import numpy as np
        tr = self._stream[idx].copy()
        t0 = tr.stats.starttime + self._xmin
        t1 = tr.stats.starttime + self._xmax
        tr.trim(t0, t1)
        if len(tr.data) < 8:
            return None, None
        tr.detrend('demean')
        tr.detrend('linear')
        return tr.data.astype(np.float64), float(tr.stats.sampling_rate)

    # ── 绘图 ─────────────────────────────────────────────────────────────
    def _refresh(self):
        if not hasattr(self, '_ref_combo'):
            return
        import numpy as np
        from PyQt5.QtWidgets import QTableWidgetItem

        ref_idx = self._ref_combo.currentData()
        if ref_idx is None or ref_idx >= len(self._raw_meta):
            return

        # 目标道索引
        tgt_indices = [
            self._target_list.item(r).data(Qt.UserRole)
            for r in range(self._target_list.count())
            if self._target_list.item(r).isSelected()
            and self._target_list.item(r).data(Qt.UserRole) != ref_idx
        ]
        if not tgt_indices:
            self._draw_message("请在左侧列表中选择至少一个目标道（不能与参考道相同）")
            return

        ref_stats, ref_color = self._raw_meta[ref_idx]
        ref_data, sr = self._trim_segment(ref_idx)
        if ref_data is None:
            self._draw_message("参考道时间窗内数据不足")
            return

        normalize  = self._norm_btn.isChecked()
        stacked    = self._stack_btn.isChecked()
        show_vel   = self._vel_btn.isChecked()

        # 清空图形
        self._fig.clear()
        n_plots = 1 if stacked else len(tgt_indices)
        axes = []
        for i in range(n_plots):
            ax = self._fig.add_subplot(n_plots, 1, i + 1,
                                       sharex=axes[0] if axes else None)
            axes.append(ax)
            ax.set_facecolor(COLORS['bg_card'])
            for sp in ax.spines.values():
                sp.set_edgecolor(COLORS['border_bright']); sp.set_linewidth(0.7)
            ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
            ax.grid(True, color=COLORS['grid'], linewidth=0.5, alpha=0.7)
            ax.axhline(0, color=COLORS['text_muted'], linewidth=0.5, alpha=0.6)
            ax.axvline(0, color=COLORS['text_muted'], linewidth=0.8,
                       linestyle='--', alpha=0.5)

        results = []
        from matplotlib.ticker import AutoMinorLocator

        for plot_i, tgt_idx in enumerate(tgt_indices):
            tgt_stats, tgt_color = self._raw_meta[tgt_idx]
            tgt_data, tgt_sr = self._trim_segment(tgt_idx)
            if tgt_data is None:
                continue

            # 对齐采样率：如不同则插值到参考道采样率
            if abs(tgt_sr - sr) > 0.01:
                try:
                    tgt_tr = self._stream[tgt_idx].copy()
                    tgt_tr.trim(tgt_tr.stats.starttime + self._xmin,
                                tgt_tr.stats.starttime + self._xmax)
                    tgt_tr.interpolate(sampling_rate=sr)
                    tgt_data = tgt_tr.data.astype(float)
                except Exception:
                    continue

            # 长度对齐（取短者）
            n_common = min(len(ref_data), len(tgt_data))
            ref_seg  = ref_data[:n_common]
            tgt_seg  = tgt_data[:n_common]

            lags, cc, peak_lag, peak_coeff = self._compute_xcorr(
                ref_seg, tgt_seg, sr, normalize)

            # 台站间距 + 视速度
            dist_km  = None
            app_vel  = None
            ref_ll   = self._get_sta_latlon(ref_stats)
            tgt_ll   = self._get_sta_latlon(tgt_stats)
            if ref_ll and tgt_ll and ref_ll[0] and tgt_ll[0]:
                dist_km = self._haversine_km(*ref_ll, *tgt_ll)
                if abs(peak_lag) > 1e-6:
                    app_vel = abs(dist_km / peak_lag)

            results.append({
                'ref':      f"{ref_stats.network}.{ref_stats.station}.{ref_stats.channel}",
                'tgt':      f"{tgt_stats.network}.{tgt_stats.station}.{tgt_stats.channel}",
                'peak_lag': peak_lag,
                'peak_cc':  peak_coeff,
                'dist_km':  dist_km,
                'app_vel':  app_vel,
                'lags':     lags,
                'cc':       cc,
                'color':    tgt_color,
            })

            # 绘制到对应 axes
            ax = axes[0] if stacked else axes[plot_i]
            tgt_label = f"{tgt_stats.network}.{tgt_stats.station}.{tgt_stats.channel}"
            ax.plot(lags, cc, color=tgt_color, linewidth=0.9, alpha=0.9,
                    label=tgt_label, rasterized=True)

            # 峰值标注
            ax.axvline(peak_lag, color=tgt_color, linewidth=1.1,
                       linestyle=':', alpha=0.85)

            vel_str = (f"  va={app_vel:.1f} km/s" if app_vel else "")
            dist_str = (f"  Δ={dist_km:.1f}km" if dist_km else "")
            ax.annotate(
                f"lag={peak_lag:.3f}s  cc={peak_coeff:.3f}{dist_str}{vel_str}",
                xy=(peak_lag, peak_coeff),
                xytext=(8, 0), textcoords='offset points',
                fontsize=7.5, fontfamily='Consolas',
                color=tgt_color,
                bbox=dict(boxstyle='round,pad=0.2', fc=COLORS['bg_header'],
                          ec=tgt_color, alpha=0.88, linewidth=0.8),
            )

            if not stacked:
                ref_label = (f"{ref_stats.network}.{ref_stats.station}."
                             f"{ref_stats.channel}")
                ax.set_title(
                    f"{ref_label}  ×  {tgt_label}",
                    color=tgt_color, fontsize=8, fontweight='bold', pad=3)
                ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                y_label = ('CC (normalized)' if normalize
                           else 'Cross-correlation')
                ax.set_ylabel(y_label, color=COLORS['text_secondary'],
                              fontsize=7.5)

        if stacked and results:
            from matplotlib.ticker import AutoMinorLocator
            axes[0].legend(loc='upper left', fontsize=6.5, framealpha=0.85,
                           facecolor=COLORS['bg_header'],
                           edgecolor=COLORS['border'],
                           labelcolor=COLORS['text_secondary'])
            ref_label = (f"{ref_stats.network}.{ref_stats.station}."
                         f"{ref_stats.channel}")
            axes[0].set_title(
                f"参考道：{ref_label}  |  叠加 {len(results)} 道互相关",
                color=ref_color, fontsize=8, fontweight='bold', pad=3)
            y_label = 'CC (normalized)' if normalize else 'Cross-correlation'
            axes[0].set_ylabel(y_label, color=COLORS['text_secondary'], fontsize=7.5)
            axes[0].yaxis.set_minor_locator(AutoMinorLocator(4))

        if axes:
            axes[-1].set_xlabel('Time lag  (s)', color=COLORS['text_secondary'],
                                fontsize=8)

        self._fig.patch.set_facecolor(COLORS['bg_card'])
        self._fig.subplots_adjust(left=0.09, bottom=0.10, right=0.97, top=0.93)
        self._canvas.draw_idle()

        # ── 更新结果表 ────────────────────────────────────────────────────
        self._last_results = results
        self._result_table.setRowCount(len(results))
        for row, r in enumerate(results):
            def cell(txt):
                from PyQt5.QtWidgets import QTableWidgetItem as TWI
                it = TWI(str(txt)); it.setTextAlignment(Qt.AlignCenter)
                return it
            self._result_table.setItem(row, 0, cell(r['ref']))
            self._result_table.setItem(row, 1, cell(r['tgt']))
            self._result_table.setItem(row, 2, cell(f"{r['peak_lag']:.4f}"))
            self._result_table.setItem(row, 3, cell(f"{r['peak_cc']:.4f}"))
            dist_str = f"{r['dist_km']:.2f}" if r['dist_km'] is not None else "—"
            vel_str  = f"{r['app_vel']:.2f}" if r['app_vel'] is not None else "—"
            self._result_table.setItem(row, 4, cell(dist_str))
            self._result_table.setItem(row, 5, cell(vel_str))

        # 更新信息栏
        win_len = self._xmax - self._xmin
        ref_id  = (f"{ref_stats.network}.{ref_stats.station}."
                   f"{ref_stats.location}.{ref_stats.channel}")
        self._info_lbl.setText(
            f"参考：{ref_id}   |   窗口：{self._xmin:.2f}s–{self._xmax:.2f}s"
            f"（{win_len:.2f}s）   |   {len(results)} 对")

    def _draw_message(self, msg: str):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self._fig.patch.set_facecolor(COLORS['bg_card'])
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha='center', va='center', fontsize=10,
                color=COLORS['text_muted'], style='italic')
        self._canvas.draw_idle()

    # ── 导出 CSV ──────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._last_results:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "无结果", "请先执行互相关分析")
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "导出互相关结果", "xcorr_results.csv",
            "CSV 文件 (*.csv);;所有文件 (*.*)")
        if not path:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["参考道", "目标道", "峰值时延(s)",
                        "相关系数", "台站间距(km)", "视速度(km/s)",
                        "时窗起始(s)", "时窗结束(s)"])
            for r in self._last_results:
                w.writerow([
                    r['ref'], r['tgt'],
                    f"{r['peak_lag']:.6f}", f"{r['peak_cc']:.6f}",
                    f"{r['dist_km']:.4f}" if r['dist_km'] is not None else "",
                    f"{r['app_vel']:.4f}" if r['app_vel'] is not None else "",
                    f"{self._xmin:.4f}", f"{self._xmax:.4f}",
                ])
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "导出成功",
                                f"结果已保存至：\n{path}")

    # ── 主窗口视图更新推送 ────────────────────────────────────────────────
    def push_update(self, stream, raw_meta, active_idx, xmin, xmax):
        self._stream     = stream
        self._raw_meta   = raw_meta
        self._active_idx = max(0, min(active_idx, len(raw_meta)-1)) if raw_meta else 0
        self._xmin = xmin; self._xmax = xmax
        self._refresh()
