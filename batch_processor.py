#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_processor.py
==================
Batch preprocessing pipeline dialog and worker thread for SeismoView.

Author
------
M Fang

Created
-------
2026-03-20

Last Modified
-------------
2026-03-20
"""

import os
import json

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QFileDialog, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QProgressBar, QTextEdit, QGroupBox, QFormLayout,
    QDialogButtonBox, QMessageBox, QFrame, QSplitter, QScrollArea
)

from config import COLORS

# ─────────────────────────────────────────────────────────────────────────────
# 预处理步骤描述符
# ─────────────────────────────────────────────────────────────────────────────

STEP_DEFS = [
    # (key, 显示名, 参数schema)
    ("demean",    "去均值 (Demean)",       {}),
    ("detrend",   "去趋势 (Detrend)",      {}),
    ("taper",     "Taper 锥化",            {"max_percentage": 0.05, "type": "cosine"}),
    ("bandpass",  "带通滤波 (Bandpass)",   {"freqmin": 1.0, "freqmax": 10.0, "corners": 4}),
    ("lowpass",   "低通滤波 (Lowpass)",    {"freq": 10.0, "corners": 4}),
    ("highpass",  "高通滤波 (Highpass)",   {"freq": 1.0,  "corners": 4}),
    ("resample",  "重采样 (Resample)",     {"sampling_rate": 100.0}),
    ("remove_response", "去仪器响应 (XML)", {"output": "VEL", "pre_filt": [0.005,0.01,45,50], "water_level": 60}),
    ("normalize", "归一化 (Max)",          {"method": "max"}),
]

STEP_KEYS  = [s[0]  for s in STEP_DEFS]
STEP_NAMES = {s[0]: s[1] for s in STEP_DEFS}

OUTPUT_FORMATS = [
    ("MSEED",   "MiniSEED (.mseed)"),
    ("SAC",     "SAC (.sac)"),
    ("SEGY",    "SEGY (.segy)"),
    ("GSE2",    "GSE2 (.gse)"),
    ("TSPAIR",  "TSPAIR (.txt)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 批量处理线程
# ─────────────────────────────────────────────────────────────────────────────

class BatchProcessThread(QThread):
    """
    在子线程中遍历输入文件，依次执行处理链，写出到输出目录。

    Signals
    -------
    progress(int, int, str)   当前文件序号, 总文件数, 文件名
    log(str)                  日志行
    finished_ok(int, int)     成功数, 失败数
    error(str)                致命错误信息
    """
    progress    = pyqtSignal(int, int, str)
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(int, int)
    error       = pyqtSignal(str)

    def __init__(self, file_list, steps, out_dir, out_fmt,
                 inventory=None, parent=None):
        """
        Parameters
        ----------
        file_list : list[str]  输入文件路径列表
        steps     : list[dict] 处理步骤，每项 {"key": ..., **params}
        out_dir   : str        输出目录
        out_fmt   : str        ObsPy write 格式字符串（如 'MSEED'）
        inventory : ObsPy Inventory 或 None
        """
        super().__init__(parent)
        self._file_list = file_list
        self._steps     = steps
        self._out_dir   = out_dir
        self._out_fmt   = out_fmt
        self._inventory = inventory
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from obspy import read
        n_total = len(self._file_list)
        n_ok = 0
        n_fail = 0

        for idx, fpath in enumerate(self._file_list):
            if self._cancelled:
                self.log.emit("⚠ 已取消")
                break

            fname = os.path.basename(fpath)
            self.progress.emit(idx + 1, n_total, fname)

            try:
                st = read(fpath)
                st = self._apply_steps(st)

                # 构建输出路径（保留扩展名或替换为目标格式扩展名）
                ext_map = {'MSEED': '.mseed', 'SAC': '.sac', 'SEGY': '.segy',
                           'GSE2': '.gse', 'TSPAIR': '.txt'}
                base = os.path.splitext(fname)[0]
                new_ext = ext_map.get(self._out_fmt, '.mseed')
                out_path = os.path.join(self._out_dir, base + new_ext)
                # 避免同名覆盖（追加序号）
                if os.path.exists(out_path) and os.path.abspath(fpath) != os.path.abspath(out_path):
                    out_path = os.path.join(self._out_dir,
                                            base + f"_proc{new_ext}")
                st.write(out_path, format=self._out_fmt)
                self.log.emit(f"✓  {fname}  →  {os.path.basename(out_path)}")
                n_ok += 1

            except Exception as e:
                self.log.emit(f"✗  {fname}  错误：{e}")
                n_fail += 1

        self.finished_ok.emit(n_ok, n_fail)

    def _apply_steps(self, st):
        """对 Stream 依次执行每个处理步骤"""
        for step in self._steps:
            key = step["key"]
            if key == "demean":
                st.detrend("demean")
            elif key == "detrend":
                st.detrend("linear")
            elif key == "taper":
                st.taper(max_percentage=step.get("max_percentage", 0.05),
                         type=step.get("type", "cosine"))
            elif key == "bandpass":
                st.filter("bandpass",
                          freqmin=step["freqmin"],
                          freqmax=step["freqmax"],
                          corners=step.get("corners", 4),
                          zerophase=True)
            elif key == "lowpass":
                st.filter("lowpass",
                          freq=step["freq"],
                          corners=step.get("corners", 4),
                          zerophase=True)
            elif key == "highpass":
                st.filter("highpass",
                          freq=step["freq"],
                          corners=step.get("corners", 4),
                          zerophase=True)
            elif key == "resample":
                for tr in st:
                    tr.resample(sampling_rate=step["sampling_rate"])
            elif key == "remove_response":
                if self._inventory is None:
                    raise RuntimeError("去仪器响应步骤需要仪器响应文件（未提供 Inventory）")
                pf = step.get("pre_filt", [0.005, 0.01, 45, 50])
                for tr in st:
                    tr.remove_response(
                        inventory=self._inventory,
                        output=step.get("output", "VEL"),
                        pre_filt=pf,
                        water_level=step.get("water_level", 60),
                    )
            elif key == "normalize":
                method = step.get("method", "max")
                for tr in st:
                    d = tr.data.astype(float)
                    if method == "max":
                        v = np.max(np.abs(d))
                        if v > 0: tr.data = d / v
                    elif method == "rms":
                        v = np.sqrt(np.mean(d ** 2))
                        if v > 0: tr.data = d / v
        return st


# ─────────────────────────────────────────────────────────────────────────────
# 步骤参数编辑区（内嵌小表单）
# ─────────────────────────────────────────────────────────────────────────────

class StepParamEditor(QWidget):
    """
    根据 step dict 动态生成参数编辑表单。
    调用 get_step() 返回含最新参数的 step dict。
    """
    def __init__(self, step: dict, parent=None):
        super().__init__(parent)
        self._key = step["key"]
        self._widgets = {}
        layout = QFormLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        def sp_float(val, lo, hi, dec=3, suf=""):
            w = QDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(dec)
            w.setValue(val); w.setFixedHeight(24)
            if suf: w.setSuffix(f"  {suf}")
            return w

        def sp_int(val, lo, hi):
            w = QSpinBox(); w.setRange(lo, hi)
            w.setValue(val); w.setFixedHeight(24); return w

        k = self._key
        if k == "taper":
            self._widgets["max_percentage"] = sp_float(
                step.get("max_percentage", 0.05), 0.001, 0.5, 3)
            layout.addRow("比例 (每端)：", self._widgets["max_percentage"])
            tc = QComboBox(); tc.setFixedHeight(24)
            for t in ["cosine","hanning","hamming","blackman","bartlett"]:
                tc.addItem(t)
            tc.setCurrentText(step.get("type","cosine"))
            self._widgets["type"] = tc
            layout.addRow("窗函数：", tc)

        elif k == "bandpass":
            self._widgets["freqmin"] = sp_float(step.get("freqmin",1.0), 0.001, 999, 3, "Hz")
            self._widgets["freqmax"] = sp_float(step.get("freqmax",10.0), 0.001, 999, 3, "Hz")
            self._widgets["corners"] = sp_int(step.get("corners",4), 1, 10)
            layout.addRow("最低频率：", self._widgets["freqmin"])
            layout.addRow("最高频率：", self._widgets["freqmax"])
            layout.addRow("阶数：",     self._widgets["corners"])

        elif k in ("lowpass","highpass"):
            self._widgets["freq"] = sp_float(step.get("freq",10.0), 0.001, 999, 3, "Hz")
            self._widgets["corners"] = sp_int(step.get("corners",4), 1, 10)
            layout.addRow("截止频率：", self._widgets["freq"])
            layout.addRow("阶数：",     self._widgets["corners"])

        elif k == "resample":
            self._widgets["sampling_rate"] = sp_float(
                step.get("sampling_rate",100.0), 0.001, 100000, 3, "Hz")
            layout.addRow("目标采样率：", self._widgets["sampling_rate"])

        elif k == "remove_response":
            oc = QComboBox(); oc.setFixedHeight(24)
            for o in ["VEL","DISP","ACC"]: oc.addItem(o)
            oc.setCurrentText(step.get("output","VEL"))
            self._widgets["output"] = oc
            pf = step.get("pre_filt",[0.005,0.01,45,50])
            self._widgets["pf0"] = sp_float(pf[0], 0.0001, 100, 4, "Hz")
            self._widgets["pf1"] = sp_float(pf[1], 0.0001, 100, 4, "Hz")
            self._widgets["pf2"] = sp_float(pf[2], 0.1, 5000, 2, "Hz")
            self._widgets["pf3"] = sp_float(pf[3], 0.1, 5000, 2, "Hz")
            self._widgets["water_level"] = sp_float(step.get("water_level",60), 1, 200, 0)
            layout.addRow("输出类型：",   oc)
            layout.addRow("预滤 f0：",   self._widgets["pf0"])
            layout.addRow("预滤 f1：",   self._widgets["pf1"])
            layout.addRow("预滤 f2：",   self._widgets["pf2"])
            layout.addRow("预滤 f3：",   self._widgets["pf3"])
            layout.addRow("水位 (dB)：", self._widgets["water_level"])

        elif k == "normalize":
            mc = QComboBox(); mc.setFixedHeight(24)
            for m in ["max","rms"]: mc.addItem(m)
            mc.setCurrentText(step.get("method","max"))
            self._widgets["method"] = mc
            layout.addRow("方式：", mc)

        else:
            layout.addRow(QLabel("（无参数）"))

    def get_step(self) -> dict:
        d = {"key": self._key}
        for k, w in self._widgets.items():
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                d[k] = w.value()
            elif isinstance(w, QComboBox):
                d[k] = w.currentText()
        # 重组 pre_filt 为 list
        if self._key == "remove_response" and "pf0" in d:
            d["pre_filt"] = [d.pop("pf0"), d.pop("pf1"),
                             d.pop("pf2"), d.pop("pf3")]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 批量处理主对话框
# ─────────────────────────────────────────────────────────────────────────────

class BatchProcessDialog(QDialog):
    """
    向导式批量预处理对话框，分三个 Tab：
      Tab 1  — 输入文件选择
      Tab 2  — 处理步骤编排（拖拽排序 + 参数编辑）
      Tab 3  — 输出设置 + 执行 + 进度日志
    """

    def __init__(self, inventory=None, last_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量预处理流水线")
        self.setMinimumSize(760, 600)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        self._inventory = inventory
        self._last_dir  = last_dir
        self._thread    = None
        self._step_editors = []   # list of StepParamEditor
        self._template_steps = [] # list[dict]，当前流水线

        self._build_ui()

    # ── UI 构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 标题
        title = QLabel("⚙  批量预处理流水线")
        title.setStyleSheet(
            f"font-size:14px;font-weight:600;"
            f"color:{COLORS['accent_blue']};padding:2px 0 6px;")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabBar::tab{{padding:6px 18px;font-size:11px;}}"
            f"QTabBar::tab:selected{{color:{COLORS['accent_blue']};"
            f"border-bottom:2px solid {COLORS['accent_blue']};}}")
        root.addWidget(self._tabs)

        self._build_tab_input()
        self._build_tab_steps()
        self._build_tab_output()

        # 底部按钮行
        btn_row = QHBoxLayout()

        self._save_tpl_btn = QPushButton("💾 保存模板")
        self._save_tpl_btn.setFixedHeight(30)
        self._save_tpl_btn.clicked.connect(self._save_template)
        btn_row.addWidget(self._save_tpl_btn)

        self._load_tpl_btn = QPushButton("📂 加载模板")
        self._load_tpl_btn.setFixedHeight(30)
        self._load_tpl_btn.clicked.connect(self._load_template)
        btn_row.addWidget(self._load_tpl_btn)

        btn_row.addStretch()

        self._run_btn = QPushButton("▶  开始批量处理")
        self._run_btn.setFixedHeight(32)
        self._run_btn.setStyleSheet(
            f"QPushButton{{background:{COLORS['accent_blue']};"
            f"color:#fff;border:none;border-radius:6px;"
            f"font-size:12px;font-weight:600;padding:0 18px;}}"
            f"QPushButton:hover{{background:{COLORS['accent_blue']}dd;}}"
            f"QPushButton:disabled{{background:{COLORS['border_bright']};"
            f"color:{COLORS['text_muted']};}}")
        self._run_btn.clicked.connect(self._start_batch)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("✕ 取消")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_batch)
        btn_row.addWidget(self._cancel_btn)

        close_btn = QPushButton("关闭")
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    # ── Tab 1：输入 ───────────────────────────────────────────────────────────
    def _build_tab_input(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(10, 10, 10, 10); vl.setSpacing(8)

        # 输入目录 / 文件
        src_box = QGroupBox("数据来源")
        src_box.setStyleSheet(self._gbox_style())
        sf = QFormLayout(src_box); sf.setSpacing(8); sf.setContentsMargins(10,10,10,10)

        dir_row = QHBoxLayout()
        self._in_dir_edit = QLineEdit()
        self._in_dir_edit.setPlaceholderText("选择包含地震波形文件的目录…")
        self._in_dir_edit.setFixedHeight(26)
        dir_browse = QPushButton("浏览…")
        dir_browse.setFixedSize(60, 26)
        dir_browse.clicked.connect(self._browse_in_dir)
        dir_row.addWidget(self._in_dir_edit)
        dir_row.addWidget(dir_browse)
        sf.addRow("输入目录：", dir_row)

        self._recursive_chk = QCheckBox("递归扫描子目录")
        self._recursive_chk.setChecked(True)
        sf.addRow("", self._recursive_chk)

        # 文件扩展名过滤
        ext_row = QHBoxLayout()
        self._ext_edit = QLineEdit("*.mseed *.seed *.sac *.msd *.miniseed")
        self._ext_edit.setFixedHeight(26)
        self._ext_edit.setToolTip("空格分隔多个扩展名，如：*.mseed *.sac")
        ext_row.addWidget(self._ext_edit)
        scan_btn = QPushButton("🔍 扫描文件"); scan_btn.setFixedHeight(26)
        scan_btn.clicked.connect(self._scan_files)
        ext_row.addWidget(scan_btn)
        sf.addRow("文件扩展名：", ext_row)
        vl.addWidget(src_box)

        # 文件列表
        vl.addWidget(QLabel("已发现的文件："))
        self._file_list_widget = QListWidget()
        self._file_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list_widget.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_card']};"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:4px;font-size:10px;font-family:Consolas;}}")
        vl.addWidget(self._file_list_widget, stretch=1)

        # 文件统计
        btn_row = QHBoxLayout()
        self._file_count_lbl = QLabel("共 0 个文件")
        self._file_count_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;")
        btn_row.addWidget(self._file_count_lbl)
        btn_row.addStretch()
        rm_btn = QPushButton("移除选中"); rm_btn.setFixedHeight(24)
        rm_btn.clicked.connect(lambda: [
            self._file_list_widget.takeItem(
                self._file_list_widget.row(it))
            for it in self._file_list_widget.selectedItems()
        ] or self._update_file_count())
        btn_row.addWidget(rm_btn)
        clr_btn = QPushButton("清空"); clr_btn.setFixedHeight(24)
        clr_btn.clicked.connect(lambda: [
            self._file_list_widget.clear(),
            self._update_file_count()])
        btn_row.addWidget(clr_btn)
        vl.addLayout(btn_row)

        self._tabs.addTab(w, "① 选择文件")

    # ── Tab 2：步骤编排 ───────────────────────────────────────────────────────
    def _build_tab_steps(self):
        w = QWidget()
        hl = QHBoxLayout(w); hl.setContentsMargins(10, 10, 10, 10); hl.setSpacing(8)

        # 左：步骤库
        left = QWidget(); left.setFixedWidth(180)
        lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0); lv.setSpacing(4)
        lv.addWidget(QLabel("可用步骤："))
        self._avail_list = QListWidget()
        self._avail_list.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_card']};"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:4px;font-size:10px;}}")
        for key, name, _ in STEP_DEFS:
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, key)
            self._avail_list.addItem(it)
        lv.addWidget(self._avail_list, stretch=1)
        add_btn = QPushButton("→ 添加到流水线")
        add_btn.setFixedHeight(26)
        add_btn.clicked.connect(self._add_step)
        lv.addWidget(add_btn)
        hl.addWidget(left)

        # 中：已添加步骤队列
        mid = QWidget()
        mv = QVBoxLayout(mid); mv.setContentsMargins(0,0,0,0); mv.setSpacing(4)
        mv.addWidget(QLabel("处理流水线（上→下依序执行）："))
        self._pipeline_list = QListWidget()
        self._pipeline_list.setDragDropMode(QAbstractItemView.InternalMove)
        self._pipeline_list.setStyleSheet(
            f"QListWidget{{background:{COLORS['bg_deep']};"
            f"border:1px solid {COLORS['accent_blue']}55;"
            f"border-radius:4px;font-size:10px;}}"
            f"QListWidget::item:selected{{background:{COLORS['accent_blue']}22;"
            f"color:{COLORS['accent_blue']};}}")
        self._pipeline_list.currentRowChanged.connect(self._on_step_selected)
        mv.addWidget(self._pipeline_list, stretch=1)
        step_btn_row = QHBoxLayout(); step_btn_row.setSpacing(4)
        for txt, fn in [("↑ 上移", self._move_step_up),
                        ("↓ 下移", self._move_step_down),
                        ("✕ 删除", self._remove_step)]:
            b = QPushButton(txt); b.setFixedHeight(24)
            b.clicked.connect(fn); step_btn_row.addWidget(b)
        mv.addLayout(step_btn_row)
        hl.addWidget(mid, stretch=1)

        # 右：参数编辑区
        right = QWidget(); right.setFixedWidth(260)
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0); rv.setSpacing(4)
        rv.addWidget(QLabel("步骤参数："))
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:4px;background:{COLORS['bg_card']};}}")
        self._param_placeholder = QLabel("← 在流水线中选择一个步骤以编辑参数")
        self._param_placeholder.setAlignment(Qt.AlignCenter)
        self._param_placeholder.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;")
        self._param_scroll.setWidget(self._param_placeholder)
        rv.addWidget(self._param_scroll, stretch=1)
        hl.addWidget(right)

        self._tabs.addTab(w, "② 编排步骤")

    # ── Tab 3：输出 + 执行 ────────────────────────────────────────────────────
    def _build_tab_output(self):
        w = QWidget()
        vl = QVBoxLayout(w); vl.setContentsMargins(10, 10, 10, 10); vl.setSpacing(8)

        out_box = QGroupBox("输出设置")
        out_box.setStyleSheet(self._gbox_style())
        of = QFormLayout(out_box); of.setSpacing(8); of.setContentsMargins(10,10,10,10)

        dir_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("选择处理结果保存目录…")
        self._out_dir_edit.setFixedHeight(26)
        out_browse = QPushButton("浏览…"); out_browse.setFixedSize(60, 26)
        out_browse.clicked.connect(self._browse_out_dir)
        dir_row.addWidget(self._out_dir_edit)
        dir_row.addWidget(out_browse)
        of.addRow("输出目录：", dir_row)

        self._fmt_combo = QComboBox(); self._fmt_combo.setFixedHeight(26)
        for key, label in OUTPUT_FORMATS:
            self._fmt_combo.addItem(label, key)
        of.addRow("输出格式：", self._fmt_combo)

        vl.addWidget(out_box)

        # 进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar{{background:{COLORS['bg_deep']};"
            f"border:none;border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{COLORS['accent_blue']};"
            f"border-radius:4px;}}")
        self._progress_bar.setValue(0)
        vl.addWidget(self._progress_bar)

        self._progress_lbl = QLabel("等待执行…")
        self._progress_lbl.setStyleSheet(
            f"color:{COLORS['text_muted']};font-size:10px;font-family:Consolas;")
        vl.addWidget(self._progress_lbl)

        # 日志区
        vl.addWidget(QLabel("处理日志："))
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet(
            f"QTextEdit{{background:{COLORS['bg_deep']};"
            f"color:{COLORS['text_secondary']};"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:4px;font-family:Consolas;font-size:10px;}}")
        vl.addWidget(self._log_edit, stretch=1)

        self._tabs.addTab(w, "③ 执行")

    # ── 辅助样式 ──────────────────────────────────────────────────────────────
    def _gbox_style(self):
        return (
            f"QGroupBox{{border:1px solid {COLORS['border_bright']};"
            f"border-radius:6px;margin-top:10px;padding-top:4px;"
            f"font-size:10px;color:{COLORS['text_secondary']};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;"
            f"subcontrol-position:top left;padding:0 6px;font-weight:600;}}")

    # ── Tab 1 逻辑 ────────────────────────────────────────────────────────────
    def _browse_in_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择输入目录", self._last_dir)
        if d:
            self._in_dir_edit.setText(d)
            self._last_dir = d
            self._scan_files()

    def _scan_files(self):
        in_dir = self._in_dir_edit.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, "无效目录", "请先选择有效的输入目录")
            return
        exts_raw = self._ext_edit.text().strip().split()
        exts = [e.lstrip("*").lower() for e in exts_raw]

        self._file_list_widget.clear()
        recursive = self._recursive_chk.isChecked()

        def _walk(root):
            for item in sorted(os.scandir(root), key=lambda e: e.name.lower()):
                if item.is_file():
                    _, ext = os.path.splitext(item.name)
                    if not exts or ext.lower() in exts or ext.lower().lstrip(".") in exts:
                        yield item.path
                elif item.is_dir() and recursive:
                    yield from _walk(item.path)

        count = 0
        for fp in _walk(in_dir):
            self._file_list_widget.addItem(fp)
            count += 1
        self._update_file_count()

    def _update_file_count(self):
        n = self._file_list_widget.count()
        self._file_count_lbl.setText(f"共 {n} 个文件")

    def _get_file_list(self):
        return [self._file_list_widget.item(i).text()
                for i in range(self._file_list_widget.count())]

    # ── Tab 2 逻辑 ────────────────────────────────────────────────────────────
    def _get_default_step(self, key):
        for k, _, params in STEP_DEFS:
            if k == key:
                d = dict(params); d["key"] = key; return d
        return {"key": key}

    def _add_step(self):
        items = self._avail_list.selectedItems()
        if not items:
            items = [self._avail_list.currentItem()]
        if not items or items[0] is None:
            return
        for it in items:
            key = it.data(Qt.UserRole)
            step = self._get_default_step(key)
            row_it = QListWidgetItem(f"  {STEP_NAMES[key]}")
            row_it.setData(Qt.UserRole, step)
            self._pipeline_list.addItem(row_it)
        self._pipeline_list.setCurrentRow(self._pipeline_list.count() - 1)

    def _remove_step(self):
        row = self._pipeline_list.currentRow()
        if row >= 0:
            self._pipeline_list.takeItem(row)
            self._param_scroll.setWidget(self._param_placeholder)

    def _move_step_up(self):
        row = self._pipeline_list.currentRow()
        if row > 0:
            it = self._pipeline_list.takeItem(row)
            self._pipeline_list.insertItem(row - 1, it)
            self._pipeline_list.setCurrentRow(row - 1)

    def _move_step_down(self):
        row = self._pipeline_list.currentRow()
        if row < self._pipeline_list.count() - 1:
            it = self._pipeline_list.takeItem(row)
            self._pipeline_list.insertItem(row + 1, it)
            self._pipeline_list.setCurrentRow(row + 1)

    def _on_step_selected(self, row):
        if row < 0:
            self._param_scroll.setWidget(self._param_placeholder)
            return
        it = self._pipeline_list.item(row)
        if it is None:
            return
        step = it.data(Qt.UserRole)
        editor = StepParamEditor(step)
        editor.setStyleSheet(
            f"QWidget{{background:{COLORS['bg_card']};}}"
            f"QLabel{{color:{COLORS['text_secondary']};font-size:10px;}}"
            f"QDoubleSpinBox,QSpinBox,QComboBox{{font-size:10px;"
            f"border:1px solid {COLORS['border_bright']};"
            f"border-radius:3px;background:{COLORS['bg_deep']};"
            f"color:{COLORS['text_primary']};}}")
        # Save current editor ref to update item data on-the-fly
        self._current_editor = editor
        self._current_row    = row

        def _on_change():
            r = self._pipeline_list.currentRow()
            it2 = self._pipeline_list.item(r)
            if it2 and self._current_editor:
                updated = self._current_editor.get_step()
                it2.setData(Qt.UserRole, updated)

        # Connect all child spinboxes/combos to _on_change
        for child in editor.findChildren(QDoubleSpinBox):
            child.valueChanged.connect(_on_change)
        for child in editor.findChildren(QSpinBox):
            child.valueChanged.connect(_on_change)
        for child in editor.findChildren(QComboBox):
            child.currentIndexChanged.connect(_on_change)

        self._param_scroll.setWidget(editor)

    def _build_pipeline(self):
        """从 QListWidget 读取步骤列表，更新每项的参数后返回 list[dict]"""
        steps = []
        for i in range(self._pipeline_list.count()):
            it = self._pipeline_list.item(i)
            steps.append(dict(it.data(Qt.UserRole)))
        return steps

    # ── Tab 3：输出目录浏览 ───────────────────────────────────────────────────
    def _browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self._last_dir)
        if d:
            self._out_dir_edit.setText(d)
            self._last_dir = d

    # ── 模板保存 / 加载 ───────────────────────────────────────────────────────
    def _save_template(self):
        steps = self._build_pipeline()
        if not steps:
            QMessageBox.warning(self, "无步骤", "流水线中没有步骤，无法保存模板")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存处理模板", "batch_template.json",
            "JSON 模板 (*.json);;所有文件 (*.*)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"steps": steps,
                       "output_format": self._fmt_combo.currentData()},
                      f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "已保存", f"模板已保存至：\n{path}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载处理模板", self._last_dir,
            "JSON 模板 (*.json);;所有文件 (*.*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                tpl = json.load(f)
            steps = tpl.get("steps", [])
            self._pipeline_list.clear()
            for step in steps:
                key = step.get("key", "")
                if key not in STEP_KEYS:
                    continue
                it = QListWidgetItem(f"  {STEP_NAMES.get(key, key)}")
                it.setData(Qt.UserRole, step)
                self._pipeline_list.addItem(it)
            # 恢复输出格式
            out_fmt = tpl.get("output_format", "MSEED")
            idx = self._fmt_combo.findData(out_fmt)
            if idx >= 0:
                self._fmt_combo.setCurrentIndex(idx)
            QMessageBox.information(
                self, "已加载",
                f"已从模板加载 {len(steps)} 个处理步骤")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法读取模板文件：\n{e}")

    # ── 批量执行 ─────────────────────────────────────────────────────────────
    def _start_batch(self):
        file_list = self._get_file_list()
        if not file_list:
            QMessageBox.warning(self, "无输入文件",
                                "请在「选择文件」标签页中选择要处理的文件")
            self._tabs.setCurrentIndex(0); return

        steps = self._build_pipeline()
        if not steps:
            QMessageBox.warning(self, "无处理步骤",
                                "请在「编排步骤」标签页中添加至少一个处理步骤")
            self._tabs.setCurrentIndex(1); return

        out_dir = self._out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "未指定输出目录",
                                "请在「执行」标签页中指定输出目录")
            self._tabs.setCurrentIndex(2); return

        # 检查去仪器响应步骤是否有 inventory
        if any(s["key"] == "remove_response" for s in steps):
            if self._inventory is None:
                QMessageBox.warning(
                    self, "缺少仪器响应",
                    "流水线中包含「去仪器响应」步骤，\n"
                    "但当前未加载仪器响应文件（StationXML）。\n"
                    "请先在主窗口通过「预处理 → 去仪器响应 → 加载 StationXML」\n"
                    "加载仪器响应后再运行批量处理。")
                return

        os.makedirs(out_dir, exist_ok=True)
        out_fmt = self._fmt_combo.currentData()

        self._tabs.setCurrentIndex(2)
        self._log_edit.clear()
        self._progress_bar.setRange(0, len(file_list))
        self._progress_bar.setValue(0)
        self._progress_lbl.setText("正在处理…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._thread = BatchProcessThread(
            file_list, steps, out_dir, out_fmt,
            inventory=self._inventory)
        self._thread.progress.connect(self._on_progress)
        self._thread.log.connect(self._on_log)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _cancel_batch(self):
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._progress_lbl.setText("正在取消…")
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, cur, total, fname):
        self._progress_bar.setValue(cur)
        self._progress_lbl.setText(f"[{cur}/{total}]  {fname}")

    def _on_log(self, msg):
        self._log_edit.append(msg)

    def _on_finished(self, n_ok, n_fail):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        summary = (f"✓ 批量处理完成  |  成功 {n_ok} 个，失败 {n_fail} 个  "
                   f"|  输出目录：{self._out_dir_edit.text()}")
        self._progress_lbl.setText(summary)
        self._log_edit.append("\n" + "─" * 50)
        self._log_edit.append(summary)
        self._progress_bar.setValue(self._progress_bar.maximum())

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "批量处理出错", msg)
