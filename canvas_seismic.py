#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canvas_seismic.py
=================
Main waveform canvas used by SeismoView.

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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from config import COLORS, WAVEFORM_COLORS

class SeismicCanvas(FigureCanvas):
    """Render seismic traces with smart downsampling and interactive picks."""
    status_message = pyqtSignal(str)
    # 拾取完成信号：(phase, t_sec, abs_time_str, amplitude, channel_label)
    pick_added     = pyqtSignal(str, float, str, float, str)
    # 视图变化信号：(xmin_sec, xmax_sec) — 供频谱面板实时更新
    view_changed   = pyqtSignal(float, float)
    # 截窗信号：(t_start_rel_sec, t_end_rel_sec) — 用户拖拽选区完成时发射
    trim_requested = pyqtSignal(float, float)

    # 超过此点数时启用智能降采样
    DOWNSAMPLE_THRESHOLD = 8_000
    # 每像素目标点数（2 = 每像素保留 min+max 两个极值）
    PX_DENSITY = 2
       
    # 拾取模式颜色
    PICK_COLORS = {
        'P': "#FF3333",   # 红
        'S': "#33AAFF",   # 蓝
    }
    PICK_LABEL_BG = {
        'P': '#8B0000',
        'S': '#00008B',
    }
    # 导入震相颜色（支持常见震相名，未知震相回退到默认色）
    IMPORTED_PHASE_COLORS = {
        'P':   '#FF4444',  'Pg':  '#FF7744',  'Pn':  '#FF2200',  'Pb': '#FF6600',
        'S':   '#44AAFF',  'Sg':  '#66BBFF',  'Sn':  '#2288FF',  'Sb': '#5599FF',
        'Lg':  '#AA55FF',  'Rg':  '#FF55AA',
        'PKP': '#FF99CC',  'SKS': '#99CCFF',
        'PP':  '#FFCC44',  'SS':  '#44FFCC',
    }
    IMPORTED_PHASE_DEFAULT_COLOR = '#BB8844'   # 未知震相用琥珀色

    def __init__(self, parent=None):
        """Create the matplotlib figure, caches, and interaction state."""
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor=COLORS['bg_card'])
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        self._axes      = []
        self._xlim_orig = None
        self._press     = None
        self._dragging  = False

        # ── 原始数据缓存（供动态重绘使用）──────────────────────────────────
        self._raw_t    = []    # list of np.ndarray, 每道的完整时间轴
        self._raw_data = []    # list of np.ndarray, 每道的完整振幅数据
        self._raw_meta = []    # list of (stats, color)
        self._n_traces = 0
        # ── 原始数据备份（预处理前，永不修改）──────────────────────────────
        self._orig_data = []   # list of np.ndarray, 与 _raw_data 等长

        # ── 拾取模式 ────────────────────────────────────────────────────────
        # mode: 'pan' | 'P' | 'S'
        self._pick_mode = 'pan'
        self._pick_artists = {'P': [], 'S': []}
        # 十字丝 artists（每 axes 一条竖线 + 一个浮动标注，不跨 axes 移动）
        self._crosshair_v    = []    # axvline，每 axes 一条
        self._crosshair_anns = []    # annotate，每 axes 一个（修复跨 axes 崩溃）
        self._crosshair_visible = False
        # 导入震相（外部文件读取，区别于手动 P/S 拾取）
        self._imported_picks_data    = []   # [(trace_idx, phase, t_sec), ...]
        self._imported_pick_artists  = []   # [[artists], ...] 与 data 平行
        # 震中距标签（排列时传入，render 时使用）
        self._dist_labels  = {}              # render_idx → dist_str
        self._time_offsets = {}              # render_idx → float (t shift in sec)
        # ── 截窗模式（两次点击：第一次定起点，第二次定终点）──────────────
        self._trim_mode    = False   # 是否处于手动截窗模式
        self._trim_start_x = None    # 第一次点击确定的起点（data 坐标），None 表示尚未点击
        self._trim_spans   = []      # axvspan artists
        self._trim_vlines  = []      # 起止竖线
        self._trim_label   = None    # 顶部时间标签
        # ── 防抖重绘定时器（缩放/平移结束后 120ms 触发高清重绘）────────────
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self._redraw_current_view)

        # 连接事件
        self.mpl_connect('scroll_event',        self._on_scroll)
        self.mpl_connect('button_press_event',  self._on_press)
        self.mpl_connect('button_release_event',self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('axes_leave_event',    self._on_axes_leave)
        self.mpl_connect('figure_leave_event',  self._on_figure_leave)

        self._draw_welcome()

    # ══════════════════════════════════════════════════════════════════════════
    # 降采样算法
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _lttb(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple:
        """
        Largest Triangle Three Buckets (LTTB) 降采样。
        视觉保真度最高的时序降采样算法，完整保留波形形态、峰谷位置。

        参数
        ----
        x, y   : 原始时间/振幅数组（等长 1D）
        n_out  : 目标输出点数（含首尾两点）

        返回
        ----
        (x_ds, y_ds) 降采样后的数组对
        """
        n = len(x)
        if n_out >= n or n_out < 3:
            return x, y

        indices = np.empty(n_out, dtype=np.intp)
        indices[0] = 0
        indices[-1] = n - 1

        # 把中间点均匀分成 (n_out-2) 个桶
        bucket_size = (n - 2) / (n_out - 2)
        a = 0  # 上一个选中点的索引

        for i in range(n_out - 2):
            # 当前桶的范围
            b_start = int((i + 1) * bucket_size) + 1
            b_end   = int((i + 2) * bucket_size) + 1
            b_end   = min(b_end, n - 1)

            if b_start >= b_end:
                # 桶为空（数据点数极少时），直接保留当前索引
                indices[i + 1] = b_start
                a = b_start
                continue

            # 下一个桶的平均值（用作三角形第三顶点）
            c_start = b_end
            c_end   = min(int((i + 3) * bucket_size) + 1, n)
            if c_start >= c_end:
                avg_x = x[b_end - 1]
                avg_y = y[b_end - 1]
            else:
                avg_x   = x[c_start:c_end].mean()
                avg_y   = y[c_start:c_end].mean()

            # 在当前桶内找使三角形面积最大的点
            ax_, ay_ = x[a], y[a]
            bx = x[b_start:b_end]
            by = y[b_start:b_end]
            # 三角形面积（×2，省去 0.5）= |det([b-a, c-a])|
            areas = np.abs(
                (ax_ - avg_x) * (by - ay_) -
                (bx  - ax_)   * (avg_y - ay_)
            )
            best_local = np.argmax(areas)
            a = b_start + best_local
            indices[i + 1] = a

        return x[indices], y[indices]

    @staticmethod
    def _minmax_downsample(x: np.ndarray, y: np.ndarray, n_out: int) -> tuple:
        """
        Min-Max 包络降采样。
        将数据分成 n_out//2 个桶，每桶取 min 和 max，
        确保任何幅度极值都不会被丢失。
        适合振幅分析，与 LTTB 互为补充。
        """
        n = len(x)
        n_buckets = max(1, n_out // 2)
        if n_buckets * 2 >= n:
            return x, y

        # 重塑为 (n_buckets, bucket_size) —— 截断尾部多余点
        bucket_size = n // n_buckets
        trim = bucket_size * n_buckets
        x_r = x[:trim].reshape(n_buckets, bucket_size)
        y_r = y[:trim].reshape(n_buckets, bucket_size)

        # 每桶的 min/max 索引（局部）
        min_idx = np.argmin(y_r, axis=1)
        max_idx = np.argmax(y_r, axis=1)

        # 全局索引
        offsets    = np.arange(n_buckets) * bucket_size
        global_min = offsets + min_idx
        global_max = offsets + max_idx

        # 按时间顺序交织 min/max
        pairs = np.column_stack([
            np.where(min_idx <= max_idx, global_min, global_max),
            np.where(min_idx <= max_idx, global_max, global_min),
        ]).ravel()

        # 去重并排序（同一桶 min==max 时会有重复）
        pairs = np.unique(pairs)
        return x[pairs], y[pairs]

    def _smart_downsample(self,
                          t: np.ndarray,
                          data: np.ndarray,
                          xmin: float,
                          xmax: float,
                          canvas_px_width: int) -> tuple:
        """
        智能降采样主入口：
        1. 裁剪到当前可见时间窗口（避免渲染屏幕外的点）
        2. 计算目标点数 = canvas 像素宽 × PX_DENSITY
        3. 点数不多时直接返回原始数据（无损）
        4. 超过阈值时：先用 Min-Max 保证极值，再用 LTTB 美化形态
        """
        # ── Step 1: 按可见窗口裁剪（+10% padding 避免边缘截断）────────────
        pad = (xmax - xmin) * 0.10
        mask = (t >= xmin - pad) & (t <= xmax + pad)
        t_vis = t[mask]
        d_vis = data[mask]
        n_vis = len(t_vis)

        if n_vis == 0:
            return t_vis, d_vis

        # ── Step 2: 计算目标点数 ────────────────────────────────────────────
        n_target = max(400, canvas_px_width * self.PX_DENSITY)

        # ── Step 3: 不需要降采样 ────────────────────────────────────────────
        if n_vis <= self.DOWNSAMPLE_THRESHOLD or n_vis <= n_target:
            return t_vis, d_vis

        # ── Step 4: 双阶段降采样 ────────────────────────────────────────────
        # 阶段 A：Min-Max 先降到目标点数的 4 倍（保留所有极值）
        intermediate = min(n_vis, n_target * 4)
        if n_vis > intermediate:
            t_mm, d_mm = self._minmax_downsample(t_vis, d_vis, intermediate)
        else:
            t_mm, d_mm = t_vis, d_vis

        # 阶段 B：LTTB 精化到目标点数（优化视觉形态）
        t_out, d_out = self._lttb(t_mm, d_mm, n_target)
        return t_out, d_out

    def _get_canvas_px_width(self) -> int:
        """获取画布实际像素宽度，用于计算目标点数"""
        try:
            w = self.fig.get_size_inches()[0] * self.fig.dpi
            return max(800, int(w))
        except Exception:
            return 1200

    def _draw_welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        ax.tick_params(colors='none')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # 主标题（纯 ASCII，不受字体影响）
        ax.text(0.5, 0.58, 'SeismoView',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=36, fontweight='bold',
                color=COLORS['accent_blue'],
                alpha=0.9,
                fontfamily='Consolas')
        ax.text(0.5, 0.46, '专业地震波形数据查看器  |  Seismic Waveform Viewer',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=13,
                color=COLORS['text_secondary'])
        ax.text(0.5, 0.34, 'MiniSEED  ·  SAC  ·  SEED  ·  GSE2等',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=11,
                color=COLORS['text_muted'])
        ax.text(0.5, 0.22, 'Open File  /  打开文件  —  Drag & Drop supported',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=11,
                color=COLORS['text_muted'],
                style='italic')

        # 装饰线
        for y, alpha in [(0.68, 0.15), (0.16, 0.15)]:
            ax.axhline(y=y, xmin=0.15, xmax=0.85,
                       color=COLORS['accent_blue'], alpha=alpha, linewidth=1)

        self.fig.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.97)
        self.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # 波形绘制（高性能入口）
    # ══════════════════════════════════════════════════════════════════════════

    def plot_stream(self, stream, selected_indices=None, dist_labels=None,
                    time_offsets=None):
        """
        加载 Stream 数据并初次渲染。
        原始数据全部缓存在内存，渲染时按当前视窗智能降采样。

        Parameters
        ----------
        dist_labels : dict | None
            {stream_index: "35.2 km"} — 若提供，在台站徽章旁显示震中距。
        time_offsets : dict | None
            {stream_index: offset_sec} — t_display = tr.times() - offset_sec
            正偏移 = 波形向左平移（如：starttime 比 origin 晚 offset 秒）
        """
        self.fig.clear()
        self._axes = []

        if selected_indices is None:
            traces_to_plot = list(stream)
            indices_used   = list(range(len(stream)))
        else:
            traces_to_plot = [stream[i] for i in selected_indices if i < len(stream)]
            indices_used   = [i for i in selected_indices if i < len(stream)]

        if not traces_to_plot:
            self._draw_welcome()
            return

        # render_idx → dist_str / offset_sec
        self._dist_labels    = {}
        self._time_offsets   = {}   # render_idx → float
        for render_i, stream_i in enumerate(indices_used):
            if dist_labels and stream_i in dist_labels:
                self._dist_labels[render_i] = dist_labels[stream_i]
            if time_offsets and stream_i in time_offsets:
                self._time_offsets[render_i] = float(time_offsets[stream_i])

        # ── 缓存原始数据（时间轴已应用偏移）──────────────────────────────
        self._raw_t    = []
        self._raw_data = []
        self._raw_meta = []
        for idx, tr in enumerate(traces_to_plot):
            offset = self._time_offsets.get(idx, 0.0)
            t    = tr.times() - offset              # 相对秒（已偏移）
            data = tr.data.astype(np.float64)
            color = WAVEFORM_COLORS[idx % len(WAVEFORM_COLORS)]
            self._raw_t.append(t)
            self._raw_data.append(data)
            self._raw_meta.append((tr.stats, color))
        self._n_traces = len(traces_to_plot)

        # ── 建立子图框架 ───────────────────────────────────────────────────
        n = self._n_traces
        axes = self.fig.subplots(n, 1, sharex=True,
                                  gridspec_kw={'hspace': 0.08})
        if n == 1:
            axes = [axes]

        px_w = self._get_canvas_px_width()
        total_raw = sum(len(d) for d in self._raw_data)
        any_downsampled = False

        for idx, ax in enumerate(axes):
            stats, color = self._raw_meta[idx]
            t_raw   = self._raw_t[idx]
            d_raw   = self._raw_data[idx]

            # 初始视图 = 全部数据范围
            xmin = float(t_raw[0])  if len(t_raw) else 0.0
            xmax = float(t_raw[-1]) if len(t_raw) else 1.0

            t_ds, d_ds = self._smart_downsample(t_raw, d_raw, xmin, xmax, px_w)
            if len(t_ds) < len(t_raw):
                any_downsampled = True

            dist_str = self._dist_labels.get(idx, None)
            self._render_trace(ax, t_ds, d_ds, stats, color, idx,
                               dist_label=dist_str)
            self._axes.append(ax)

        # ── X 轴标签（根据对齐模式动态调整）────────────────────────────
        if self._time_offsets and any(v != 0.0 for v in self._time_offsets.values()):
            # 判断是否所有偏移都相同（发震时刻对齐）
            offvals = list(self._time_offsets.values())
            if len(set(f"{v:.6f}" for v in offvals)) == 1:
                # 单一偏移 → 发震时刻对齐
                axes[-1].set_xlabel(
                    'Time relative to origin (s) / 相对发震时刻（秒）',
                    color=COLORS['text_secondary'], fontsize=10)
            else:
                axes[-1].set_xlabel(
                    'Reduced time (s) / 对齐时间（秒）',
                    color=COLORS['text_secondary'], fontsize=10)
        else:
            axes[-1].set_xlabel('Time (s) / 时间（秒）',
                                color=COLORS['text_secondary'], fontsize=10)

        # ── 标题：仅显示时间范围 + 道数（台站已在各道内标注）────────────────
        st0 = traces_to_plot[0].stats
        t_start = str(st0.starttime)[:19].replace('T', ' ')
        t_end   = str(traces_to_plot[-1].stats.endtime)[:19].replace('T', ' ')
        title = f"{t_start} – {t_end} UTC  |  {n} trace(s)"
        self.fig.suptitle(title,
                          color=COLORS['text_primary'],
                          fontsize=10, fontweight='bold', y=0.99)
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        self.fig.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.97)

        if self._axes:
            self._xlim_orig = self._axes[0].get_xlim()

        # 重置拾取状态（新数据）
        self._pick_artists = {'P': [], 'S': []}
        # 重置导入震相
        self._imported_picks_data   = []
        self._imported_pick_artists = []
        # 重置截窗状态
        self._trim_mode    = False
        self._trim_start_x = None
        self._trim_spans   = []
        self._trim_vlines  = []
        self._trim_label   = None
        self.draw()
        # 初始化十字丝
        self._rebuild_crosshair_after_redraw()

        # 通知频谱面板（全量视图）
        if self._xlim_orig:
            self.view_changed.emit(float(self._xlim_orig[0]),
                                   float(self._xlim_orig[1]))

        # 状态栏提示
        if any_downsampled:
            ratio = sum(len(d) for d in self._raw_data) / max(
                1, sum(len(self._smart_downsample(
                    self._raw_t[i], self._raw_data[i],
                    float(self._raw_t[i][0]), float(self._raw_t[i][-1]), px_w
                )[0]) for i in range(self._n_traces))
            )
            self.status_message.emit(
                f"智能降采样已启用  —  原始 {total_raw:,} pts  →  "
                f"渲染 ~{total_raw // max(1,int(ratio)):,} pts  "
                f"（LTTB + MinMax，振幅极值完整保留）"
            )

    def _render_trace(self, ax, t_ds, d_ds, stats, color, idx, dist_label=None):
        """将一道降采样后的数据绘制到指定 Axes"""
        if len(t_ds) == 0:
            return

        ax.plot(t_ds, d_ds, color=color, linewidth=0.65, alpha=0.92, rasterized=True)

        # 半透明填充（仅在数据量不太大时，避免影响性能）
        if len(t_ds) <= 10_000:
            ax.fill_between(t_ds, d_ds, 0,
                             where=(d_ds >= 0), color=color, alpha=0.05)
            ax.fill_between(t_ds, d_ds, 0,
                             where=(d_ds <  0), color=color, alpha=0.05)

        ax.axhline(0, color=COLORS['text_muted'], linewidth=0.5, alpha=0.5)

        # 样式
        ax.set_facecolor(COLORS['bg_card'])
        ax.tick_params(axis='x', colors=COLORS['text_secondary'],
                       labelsize=9, length=4)
        ax.tick_params(axis='y', colors=COLORS['text_secondary'],
                       labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['border_bright'])
            spine.set_linewidth(0.8)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.tick_params(which='minor', length=2, color=COLORS['text_muted'])
        ax.grid(True, which='major', color=COLORS['grid'],
                linewidth=0.6, alpha=0.7)
        ax.grid(True, which='minor', color=COLORS['grid'],
                linewidth=0.3, alpha=0.4)

        # ── Y 轴：仅显示采样率，不占用横向空间 ────────────────────────────
        # ax.set_ylabel(f"{stats.sampling_rate:.0f} Hz",
        #               color=COLORS['text_muted'],
        #               fontsize=7, rotation=90,
        #               va='center', labelpad=4)

        # ── 台站·通道徽章（左上角，带彩色左边框）────────────────────────────
        sta = stats.station  or ''
        cha = stats.channel  or ''
        
        # 构建显示字符串：台站.通道 (例如 "IU.COLA.BHZ" 或 "IU.COLA..BHZ")
        display_str = f"{sta}.{cha}"  # 如果希望包含网络，可改为 f"{net}.{sta}.{cha}"

        # 第一行：台站名（大字，彩色）
        ax.text(0.008, 0.97,
                display_str,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=11, fontweight='bold',
                color=color,
                fontfamily='Consolas',
                zorder=5,
                bbox=dict(boxstyle='round,pad=0.18',
                          facecolor='white',
                          edgecolor=color,
                          linewidth=1.2,
                          alpha=0.88))

        # # 第二行：NET.LOC.CHA（小字，灰色）
        # loc_str = f".{loc}" if loc and loc.strip() else ''
        # sub_label = f"{net}{loc_str}.{cha}"
        # ax.text(0.008, 0.68,
        #         sub_label,
        #         transform=ax.transAxes,
        #         ha='left', va='top',
        #         fontsize=8,
        #         color=COLORS['text_secondary'],
        #         fontfamily='Consolas',
        #         zorder=5,
        #         bbox=dict(boxstyle='round,pad=0.15',
        #                   facecolor='white',
        #                   edgecolor='none',
        #                   alpha=0.75))

        # 第三行（可选）：震中距标签（橙色小字）
        if dist_label:
            ax.text(0.008, 0.44,
                    f"Δ {dist_label}",
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=8, fontweight='bold',
                    color='#EA580C',
                    fontfamily='Consolas',
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='#FFF7ED',
                              edgecolor='#FED7AA',
                              linewidth=0.8,
                              alpha=0.90))

    # ══════════════════════════════════════════════════════════════════════════
    # 动态分辨率重绘（缩放/平移后调用）
    # ══════════════════════════════════════════════════════════════════════════

    def _redraw_current_view(self):
        """
        根据当前 xlim 范围重新采样并更新所有线条数据。
        只更新 line.set_data，不重建 Axes，速度极快。
        """
        if not self._axes or not self._raw_t:
            return

        ax0 = self._axes[0]
        xmin, xmax = ax0.get_xlim()
        px_w = self._get_canvas_px_width()

        n_rendered = 0
        for i, ax in enumerate(self._axes):
            if i >= self._n_traces:
                break
            t_raw = self._raw_t[i]
            d_raw = self._raw_data[i]

            t_ds, d_ds = self._smart_downsample(t_raw, d_raw, xmin, xmax, px_w)
            n_rendered += len(t_ds)

            # 更新已有线条（第 0 条线 = 波形主线）
            lines = ax.get_lines()
            if lines:
                lines[0].set_data(t_ds, d_ds)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # 恢复拾取线标签（Y 坐标跟随新的 ylim）
        active = self._active_pick_times()
        for phase in ['P', 'S']:
            if phase in active:
                self._remove_pick_artists(phase)
                self._place_pick_silent(phase, active[phase])

        # 恢复导入震相标注
        self._rebuild_imported_picks()

        # 重建十字丝（relim 后旧 artist 坐标已失效）
        self._rebuild_crosshair_after_redraw()

        self.draw_idle()

        # 状态栏：显示当前可见窗口的信息
        span = xmax - xmin
        zoom_pct = 100.0 * span / (
            self._xlim_orig[1] - self._xlim_orig[0]
        ) if self._xlim_orig else 100.0
        self.status_message.emit(
            f"视图  {xmin:.2f}s — {xmax:.2f}s  "
            f"（跨度 {span:.2f}s，缩放 {zoom_pct:.1f}%）  "
            f"渲染 {n_rendered:,} pts"
        )
        # 通知频谱面板随视图更新
        self.view_changed.emit(float(xmin), float(xmax))

    # ══════════════════════════════════════════════════════════════════════════
    # 拾取模式管理
    # ══════════════════════════════════════════════════════════════════════════

    def set_pick_mode(self, mode: str):
        """切换模式：'pan' | 'P' | 'S'"""
        self._pick_mode = mode
        self._hide_crosshair()
        if mode in ('P', 'S'):
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    # ── 手动截窗模式 ──────────────────────────────────────────────────────────
    def set_trim_mode(self, on: bool):
        """开启 / 关闭手动截窗模式（两次点击交互）"""
        self._trim_mode    = on
        self._trim_start_x = None
        self._clear_trim_overlay()
        if on:
            self.setCursor(QCursor(Qt.CrossCursor))
            self.status_message.emit(
                "✂  截窗模式  —  【第 1 次点击】设置起点，【第 2 次点击】设置终点，右键取消")
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

    def _clear_trim_overlay(self):
        """移除所有截窗预览 artist"""
        for sp in self._trim_spans:
            try: sp.remove()
            except Exception: pass
        for vl in self._trim_vlines:
            try: vl.remove()
            except Exception: pass
        if self._trim_label:
            try: self._trim_label.remove()
            except Exception: pass
        self._trim_spans  = []
        self._trim_vlines = []
        self._trim_label  = None
        self.draw_idle()

    def _update_trim_overlay(self, x_start: float, x_cursor: float):
        """实时更新截窗选区预览（橙色半透明矩形 + 两侧虚线 + 顶部标签）"""
        # 清除旧预览
        for sp in self._trim_spans:
            try: sp.remove()
            except Exception: pass
        self._trim_spans = []
        for vl in self._trim_vlines:
            try: vl.remove()
            except Exception: pass
        self._trim_vlines = []
        if self._trim_label:
            try: self._trim_label.remove()
            except Exception: pass
        self._trim_label = None

        if not self._axes:
            self.draw_idle()
            return

        lo, hi = min(x_start, x_cursor), max(x_start, x_cursor)
        FILL  = '#F97316'
        ALPHA = 0.18

        for ax in self._axes:
            sp = ax.axvspan(lo, hi, facecolor=FILL, alpha=ALPHA, zorder=4)
            l0 = ax.axvline(lo, color=FILL, linewidth=1.8, linestyle='--',
                            alpha=0.9, zorder=5)
            l1 = ax.axvline(hi, color=FILL, linewidth=1.8, linestyle='--',
                            alpha=0.9, zorder=5)
            self._trim_spans.append(sp)
            self._trim_vlines += [l0, l1]

        # 顶部信息标签（绝对时间）
        t0_str = self._t_to_abstime(lo)
        t1_str = self._t_to_abstime(hi)
        dur    = hi - lo
        label  = f"{t0_str}  →  {t1_str}  （{dur:.3f} s）"
        self._trim_label = self._axes[0].text(
            0.5, 0.99, label,
            transform=self._axes[0].transAxes, ha='center', va='top',
            fontsize=9, fontweight='bold', color=FILL, zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=FILL, linewidth=1.2, alpha=0.92)
        )
        self.draw_idle()

    def _amplitude_at(self, trace_idx: int, t_sec: float) -> float:
        """在原始数据上线性插值得到 t_sec 处的振幅"""
        if trace_idx >= len(self._raw_t):
            return 0.0
        t_arr = self._raw_t[trace_idx]
        d_arr = self._raw_data[trace_idx]
        if len(t_arr) == 0:
            return 0.0
        idx = int(np.searchsorted(t_arr, t_sec))
        idx = max(0, min(idx, len(t_arr) - 1))
        if 0 < idx < len(t_arr) - 1:
            t0, t1 = t_arr[idx-1], t_arr[idx]
            if t1 > t0:
                frac = (t_sec - t0) / (t1 - t0)
                return float(d_arr[idx-1] + frac * (d_arr[idx] - d_arr[idx-1]))
        return float(d_arr[idx])

    def _t_to_abstime(self, t_sec: float) -> str:
        """相对秒 → UTC 绝对时间字符串"""
        if not self._raw_meta:
            return f"{t_sec:.4f}s"
        try:
            abs_t = self._raw_meta[0][0].starttime + t_sec
            s = str(abs_t).replace('T', ' ').rstrip('Z')
            if '.' in s:
                main, frac = s.split('.')
                s = main + '.' + frac[:3]
            return s
        except Exception:
            return f"{t_sec:.4f}s"

    # ── 十字丝 ───────────────────────────────────────────────────────────────
    def _init_crosshair(self):
        """每 axes 创建独立的竖线 + 浮动标注，完全避免跨 axes 移动 artist 的崩溃"""
        self._crosshair_v    = []
        self._crosshair_anns = []
        for ax in self._axes:
            vl = ax.axvline(x=0, color=COLORS['accent_green'], linewidth=1.0,
                            linestyle='--', alpha=0.85, zorder=10, visible=False)
            ann = ax.annotate(
                '', xy=(0, 0), xycoords='data',
                xytext=(14, 14), textcoords='offset points',
                fontsize=8.5, fontfamily='Consolas',
                color=COLORS['text_primary'],
                bbox=dict(boxstyle='round,pad=0.45',
                          fc=COLORS['bg_header'], ec=COLORS['accent_green'],
                          alpha=0.93, linewidth=1.1),
                zorder=15, visible=False,
            )
            self._crosshair_v.append(vl)
            self._crosshair_anns.append(ann)

    def _update_crosshair(self, t_sec: float, ax_hit):
        """更新竖向时间线位置与浮动标注（各 axes 独立 artist，无跨轴移动）"""
        if not self._crosshair_v:
            return
        phase = self._pick_mode
        color = self.PICK_COLORS.get(phase, COLORS['accent_green'])
        ax_idx = self._axes.index(ax_hit) if ax_hit in self._axes else 0
        amp    = self._amplitude_at(ax_idx, t_sec)
        abs_t  = self._t_to_abstime(t_sec)
        phase_tag = f"[ {phase}-wave ]" if phase != 'pan' else '[ Cursor ]'
        text = (f"{phase_tag}\n"
                f"Time  = {abs_t}\n"
                f"Ampl  = {amp:.6g}")

        for i, (vl, ann) in enumerate(zip(self._crosshair_v, self._crosshair_anns)):
            vl.set_xdata([t_sec, t_sec])
            vl.set_color(color)
            vl.set_visible(True)
            if i == ax_idx:
                ann.set_text(text)
                ann.xy = (t_sec, amp)
                ann.set_color(color)
                ann.get_bbox_patch().set_edgecolor(color)
                ann.set_visible(True)
            else:
                ann.set_visible(False)

        self._crosshair_visible = True
        self.draw_idle()

    def _hide_crosshair(self):
        """
        强制隐藏所有十字丝元素。
        不在此处调用 draw_idle()——由调用方统一触发渲染，
        确保"隐藏 + 移动视图"在同一帧内完成，避免残影。
        """
        for vl in self._crosshair_v:
            vl.set_visible(False)
        for ann in self._crosshair_anns:
            ann.set_visible(False)
        self._crosshair_visible = False

    # ── 放置拾取 ─────────────────────────────────────────────────────────────
    def _place_pick(self, phase: str, t_sec: float, ax_hit):
        """在所有 axes 绘制 P 或 S 拾取竖线 + 标签"""
        self._remove_pick_artists(phase)  # 清除同类旧拾取
        color  = self.PICK_COLORS[phase]
        bg_col = self.PICK_LABEL_BG[phase]
        ax_idx = self._axes.index(ax_hit) if ax_hit in self._axes else 0
        amp    = self._amplitude_at(ax_idx, t_sec)
        abs_t  = self._t_to_abstime(t_sec)
        new_artists = []

        for i, ax in enumerate(self._axes):
            vl = ax.axvline(x=t_sec, color=color, linewidth=1.6,
                            linestyle='-', alpha=0.9, zorder=9)
            # 相位标签（仅第一道）
            lbl = None
            if i == 0:
                ylim  = ax.get_ylim()
                y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.04
                lbl   = ax.text(t_sec, y_top, f'  {phase}  ',
                                color=color, fontsize=10, fontweight='bold',
                                fontfamily='Consolas', va='top', ha='left',
                                zorder=11,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          fc=bg_col, ec=color,
                                          alpha=0.9, linewidth=1.2))
            # 时刻标签（每道底部）
            ylim2 = ax.get_ylim()
            y_bot = ylim2[0] + (ylim2[1] - ylim2[0]) * 0.015
            time_lbl = ax.text(
                t_sec, y_bot,
                f" {abs_t.split(' ')[-1]} ",   # 只显示 HH:MM:SS.mmm
                color=color, fontsize=7.5, alpha=0.82,
                fontfamily='Consolas', va='bottom', ha='left', zorder=11,
            )
            new_artists.extend([a for a in [vl, lbl, time_lbl] if a])

        self._pick_artists[phase] = new_artists
        self.draw_idle()

        # S-P 时差
        sp_info = ''
        active = self._active_pick_times()
        if 'P' in active and 'S' in active:
            sp = active['S'] - active['P']
            if sp >= 0:
                dist_km = sp * 8.0   # 粗估（8 km/s Vp）
                sp_info = f"   │   S-P = {sp:.3f}s  (~{dist_km:.0f} km)"

        # 发射信号
        ch_label = ''
        if self._raw_meta:
            s = self._raw_meta[0][0]
            ch_label = f"{s.network}.{s.station}.{s.channel}"
        self.pick_added.emit(phase, t_sec, abs_t, amp, ch_label)
        self.status_message.emit(
            f"✓ {phase}波拾取  ─  "
            f"Time = {abs_t}   Ampl = {amp:.6g}{sp_info}"
        )

    def _active_pick_times(self) -> dict:
        """返回当前有效拾取 {phase: t_sec}"""
        result = {}
        for phase, arts in self._pick_artists.items():
            if arts:
                try:
                    result[phase] = float(arts[0].get_xdata()[0])
                except Exception:
                    pass
        return result

    def _remove_pick_artists(self, phase: str):
        for art in self._pick_artists.get(phase, []):
            try:
                art.remove()
            except Exception:
                pass
        self._pick_artists[phase] = []

    def clear_picks(self, phase: str = None):
        phases = ['P', 'S'] if phase is None else [phase]
        for p in phases:
            self._remove_pick_artists(p)
        self.draw_idle()
        label = '全部' if phase is None else phase
        self.status_message.emit(f"{label}波拾取已清除")

    def get_picks(self) -> dict:
        """返回 {phase: (t_sec, abs_time, amp)}"""
        result = {}
        for phase, arts in self._pick_artists.items():
            if arts:
                try:
                    t   = float(arts[0].get_xdata()[0])
                    result[phase] = (t, self._t_to_abstime(t),
                                     self._amplitude_at(0, t))
                except Exception:
                    pass
        return result

    def _rebuild_crosshair_after_redraw(self):
        """
        重绘后重建十字丝 artists（每 axes 独立 axvline + annotate）。
        必须先 .remove() 旧 artist，否则它们仍存活于 Axes 并造成残影。
        """
        for vl in self._crosshair_v:
            try: vl.remove()
            except Exception: pass
        for ann in self._crosshair_anns:
            try: ann.remove()
            except Exception: pass
        self._crosshair_v    = []
        self._crosshair_anns = []
        self._crosshair_visible = False
        self._init_crosshair()

    def _place_pick_silent(self, phase: str, t_sec: float):
        """重绘后重建拾取线，不发射信号"""
        color  = self.PICK_COLORS[phase]
        bg_col = self.PICK_LABEL_BG[phase]
        abs_t  = self._t_to_abstime(t_sec)
        new_artists = []
        for i, ax in enumerate(self._axes):
            vl = ax.axvline(x=t_sec, color=color, linewidth=1.6,
                            linestyle='-', alpha=0.9, zorder=9)
            lbl = None
            if i == 0:
                ylim  = ax.get_ylim()
                y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.04
                lbl   = ax.text(t_sec, y_top, f'  {phase}  ',
                                color=color, fontsize=10, fontweight='bold',
                                fontfamily='Consolas', va='top', ha='left',
                                zorder=11,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          fc=bg_col, ec=color,
                                          alpha=0.9, linewidth=1.2))
            ylim2 = ax.get_ylim()
            y_bot = ylim2[0] + (ylim2[1] - ylim2[0]) * 0.015
            time_lbl = ax.text(
                t_sec, y_bot, f" {abs_t.split(' ')[-1]} ",
                color=color, fontsize=7.5, alpha=0.82,
                fontfamily='Consolas', va='bottom', ha='left', zorder=11)
            new_artists.extend([a for a in [vl, lbl, time_lbl] if a])
        self._pick_artists[phase] = new_artists

    # ── 导入震相（外部文件读入的标注，独立于手动 P/S 拾取）────────────────────
    def add_imported_picks(self, picks):
        """
        绘制并存储从外部文件读入的震相标注。

        Parameters
        ----------
        picks : list of (trace_idx, phase, t_sec)
            trace_idx : int    对应 self._axes 的索引（第几道）
            phase     : str    震相名（'P'/'Pg'/'S'/'Sg'/... 任意字符串）
            t_sec     : float  相对于数据起始时刻的秒数
        """
        self.clear_imported_picks()   # 先清空旧的
        for (tidx, phase, t_sec) in picks:
            if tidx < 0 or tidx >= len(self._axes):
                continue
            arts = self._draw_single_imported_pick(tidx, phase, t_sec)
            self._imported_picks_data.append((tidx, phase, t_sec))
            self._imported_pick_artists.append(arts)
        self.draw_idle()

    def clear_imported_picks(self):
        """移除所有导入震相的 artist"""
        for arts in self._imported_pick_artists:
            for a in arts:
                try: a.remove()
                except Exception: pass
        self._imported_picks_data   = []
        self._imported_pick_artists = []
        self.draw_idle()

    def _draw_single_imported_pick(self, trace_idx: int, phase: str,
                                   t_sec: float) -> list:
        """
        在指定道的 axes 上绘制一个导入震相标注。
        使用虚线竖线 + 三角标记 + 相位标签（与手动拾取的实线竖线视觉区分）。
        返回所有新建 artist 的列表。
        """
        if trace_idx >= len(self._axes):
            return []
        ax    = self._axes[trace_idx]
        color = self.IMPORTED_PHASE_COLORS.get(
            phase, self.IMPORTED_PHASE_DEFAULT_COLOR)
        abs_t = self._t_to_abstime(t_sec)
        arts  = []

        # 竖线（长虚线，略细，区别于手动拾取的实线）
        vl = ax.axvline(x=t_sec, color=color, linewidth=1.4,
                        linestyle=(0, (6, 3)),   # 长虚线
                        alpha=0.88, zorder=8)
        arts.append(vl)

        # 顶部三角形标记 + 震相名
        ylim  = ax.get_ylim()
        y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.02
        lbl = ax.text(
            t_sec, y_top,
            f" ▼{phase} ",
            color=color, fontsize=9, fontweight='bold',
            fontfamily='Consolas', va='top', ha='center', zorder=12,
            bbox=dict(boxstyle='round,pad=0.2',
                      fc=color + '22', ec=color,
                      linewidth=1.0, alpha=0.92)
        )
        arts.append(lbl)

        # 底部时间标签
        y_bot = ylim[0] + (ylim[1] - ylim[0]) * 0.015
        t_lbl = ax.text(
            t_sec, y_bot,
            f" {abs_t.split(' ')[-1]} ",
            color=color, fontsize=7.5, alpha=0.80,
            fontfamily='Consolas', va='bottom', ha='left', zorder=12,
        )
        arts.append(t_lbl)
        return arts

    def _rebuild_imported_picks(self):
        """重绘后重建导入震相 artists（仅更新 artist，不改变数据）"""
        for arts in self._imported_pick_artists:
            for a in arts:
                try: a.remove()
                except Exception: pass
        self._imported_pick_artists = []
        for (tidx, phase, t_sec) in self._imported_picks_data:
            if tidx < len(self._axes):
                arts = self._draw_single_imported_pick(tidx, phase, t_sec)
            else:
                arts = []
            self._imported_pick_artists.append(arts)



    def _clamp_xlim(self, new_min, new_max):
        """确保视图不超出原始数据范围"""
        if not self._xlim_orig:
            return new_min, new_max
        orig_min, orig_max = self._xlim_orig
        span = new_max - new_min
        if new_min < orig_min:
            new_min, new_max = orig_min, orig_min + span
        if new_max > orig_max:
            new_min, new_max = orig_max - span, orig_max
        return new_min, new_max

    def _apply_xlim(self, new_min, new_max, schedule_redraw=True):
        """应用新的 xlim 并按需调度高清重绘。滚轮/平移前先隐藏十字丝避免残影。"""
        # 移动视图前强制清除十字丝，防止残影
        self._hide_crosshair()
        new_min, new_max = self._clamp_xlim(new_min, new_max)
        for ax in self._axes:
            ax.set_xlim(new_min, new_max)
        self.draw_idle()
        if schedule_redraw:
            self._redraw_timer.start()  # 防抖：停止操作后 120ms 触发精绘

    def _on_scroll(self, event):
        if not self._axes or event.xdata is None:
            return
        ax = self._axes[0]
        xmin, xmax = ax.get_xlim()
        factor = 0.82 if event.button == 'up' else 1.0 / 0.82
        cx = event.xdata
        self._apply_xlim(cx - (cx - xmin) * factor,
                         cx + (xmax - cx) * factor)

    # ── 交互：平移 / 截窗（两次点击）/ 拾取 ────────────────────────────────
    def _on_press(self, event):
        if not self._axes or event.xdata is None:
            return

        if event.button == 1:
            # ── 截窗模式：两次点击 ──────────────────────────────────────────
            if self._trim_mode:
                if self._trim_start_x is None:
                    # 第一次点击：记录起点，显示单条竖线
                    self._trim_start_x = event.xdata
                    self._clear_trim_overlay()
                    FILL = '#F97316'
                    for ax in self._axes:
                        vl = ax.axvline(event.xdata, color=FILL,
                                        linewidth=2.0, linestyle='--',
                                        alpha=0.9, zorder=5)
                        self._trim_vlines.append(vl)
                    t_str = self._t_to_abstime(event.xdata)
                    self._trim_label = self._axes[0].text(
                        0.5, 0.99,
                        f"✂  起点已设置：{t_str}  —  再次点击设置终点，右键取消",
                        transform=self._axes[0].transAxes,
                        ha='center', va='top', fontsize=9, fontweight='bold',
                        color=FILL, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=FILL, linewidth=1.2, alpha=0.92)
                    )
                    self.draw_idle()
                    self.status_message.emit(
                        f"✂  起点：{t_str}  —  请点击设置终点，右键取消")
                else:
                    # 第二次点击：确认终点，执行截窗
                    x_end = event.xdata
                    if abs(x_end - self._trim_start_x) < 0.001:
                        # 两次点击位置几乎相同，忽略，重新选
                        self.status_message.emit(
                            "✂  起点与终点位置太近，请重新点击终点位置")
                        return
                    t0 = min(self._trim_start_x, x_end)
                    t1 = max(self._trim_start_x, x_end)
                    self.set_trim_mode(False)          # 退出截窗模式（清除所有 overlay）
                    self.trim_requested.emit(float(t0), float(t1))
                return

            # ── 拾取模式 ────────────────────────────────────────────────────
            if self._pick_mode in ('P', 'S'):
                self._place_pick(self._pick_mode, event.xdata, event.inaxes)
                return

            # ── 平移模式 ────────────────────────────────────────────────────
            self._hide_crosshair()
            self._press    = event.xdata
            self._dragging = True
            self._redraw_timer.stop()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

        elif event.button == 3:
            if self._trim_mode:
                # 右键：若已有起点则重置起点，否则退出截窗模式
                if self._trim_start_x is not None:
                    self._trim_start_x = None
                    self._clear_trim_overlay()
                    self.status_message.emit(
                        "✂  截窗模式  —  【第 1 次点击】设置起点，【第 2 次点击】设置终点，右键取消")
                else:
                    self.set_trim_mode(False)
                    self.status_message.emit("截窗已取消")
            else:
                # 右键：清除最近一次拾取（按 P → S → 无 的顺序）
                active = self._active_pick_times()
                if 'S' in active:
                    self.clear_picks('S')
                elif 'P' in active:
                    self.clear_picks('P')

    def _on_release(self, event):
        # 截窗模式由点击（press）驱动，release 不处理截窗逻辑
        was_dragging = self._dragging
        self._press    = None
        self._dragging = False
        if self._trim_mode:
            return   # 截窗模式下 release 不改变游标
        if self._pick_mode == 'pan':
            self.setCursor(QCursor(Qt.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CrossCursor))
        if was_dragging and self._axes:
            self._redraw_timer.start()

    def _on_motion(self, event):
        # ── 截窗模式：起点已设置时，实时预览选区 ───────────────────────────
        if self._trim_mode and self._trim_start_x is not None:
            if event.xdata is not None and event.inaxes in self._axes:
                self._update_trim_overlay(self._trim_start_x, event.xdata)
            return

        # ── 平移拖拽 ────────────────────────────────────────────────────────
        if self._dragging and self._press is not None:
            if self._axes and event.xdata is not None:
                dx = self._press - event.xdata
                ax = self._axes[0]
                xmin, xmax = ax.get_xlim()
                self._apply_xlim(xmin + dx, xmax + dx, schedule_redraw=False)
                if not self._redraw_timer.isActive():
                    self._redraw_timer.start(200)
            return

        # ── 十字丝跟踪（拾取模式或普通游标模式）────────────────────────────
        if not self._axes:
            return
        if event.xdata is None or event.inaxes not in self._axes:
            self._hide_crosshair()
            return
        if not self._crosshair_v:
            self._init_crosshair()
        self._update_crosshair(event.xdata, event.inaxes)

    def _on_axes_leave(self, event):
        self._hide_crosshair()
        self.draw_idle()
        if self._pick_mode == 'pan':
            self.setCursor(QCursor(Qt.ArrowCursor))

    def _on_figure_leave(self, event):
        self._hide_crosshair()
        self.draw_idle()
        
    def reset_view(self) -> None:
        """Reset the view to the original xlim."""
        if self._axes and self._xlim_orig:
            for ax in self._axes:
                ax.set_xlim(self._xlim_orig)
            self._redraw_timer.start()
            self.status_message.emit("视图已重置")

    def zoom_in(self) -> None:
        """Zoom in the plot by 35% of the current view range."""
        if not self._axes:
            return
        ax = self._axes[0]
        xmin, xmax = ax.get_xlim()
        cx = (xmin + xmax) / 2
        half = (xmax - xmin) * 0.35
        self._apply_xlim(cx - half, cx + half)

    def zoom_out(self) -> None:
        """Zoom out the plot by 75% of the current view range."""
        if not self._axes:
            return
        ax = self._axes[0]
        xmin, xmax = ax.get_xlim()
        cx = (xmin + xmax) / 2
        half = (xmax - xmin) * 0.75
        if self._xlim_orig:
             half = min(half, (self._xlim_orig[1] - self._xlim_orig[0]) / 2)
        self._apply_xlim(cx - half, cx + half)
