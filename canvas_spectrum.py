#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canvas_spectrum.py
==================
Spectrum plotting canvases for SeismoView.

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
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from config import COLORS

# scipy imported lazily — avoids access violation in PyInstaller frozen exe
SCIPY_AVAILABLE = None

def _check_scipy():
    global SCIPY_AVAILABLE
    if SCIPY_AVAILABLE is None:
        try:
            import scipy.signal  # noqa: F401
            SCIPY_AVAILABLE = True
        except Exception:
            SCIPY_AVAILABLE = False
    return SCIPY_AVAILABLE

class SpectrumCanvas(FigureCanvas):
    """
    显示当前可见时间窗口的振幅频谱（单道）。
    - 默认显示第 0 道；用 set_active_trace(idx) 切换到指定道
    - Welch 平均 / 单次 FFT
    - 纵轴：振幅(counts) 或 dB re 1 count
    - 横轴：线性 / 对数频率
    - Nyquist 标线、峰值频率注释
    """

    def __init__(self, parent=None):
        """Create the spectrum canvas and initialize display options."""
        # 使用 constrained_layout 避免 tight_layout 与 suptitle 的兼容性警告
        self.fig = Figure(figsize=(4, 6), dpi=100,
                          facecolor=COLORS['bg_card'],
                          layout='constrained')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        self._raw_t      = []
        self._raw_data   = []
        self._raw_meta   = []
        self._active_idx = 0   # 当前显示哪一道
        self._last_xlim  = (0.0, 1.0)  # 记录上次视图范围，供选道切换时复用

        # 显示选项
        self.opt_welch = True
        self.opt_log_x = True
        self.opt_db    = True
        self.opt_log_y = False

        self._draw_placeholder()

    # ── 占位图 ──────────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        """Render a placeholder view before waveform data is loaded."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        for sp in ax.spines.values():
            sp.set_color(COLORS['border'])
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.58, 'Spectrum',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold',
                color=COLORS['accent_green'], alpha=0.7,
                fontfamily='Consolas')
        ax.text(0.5, 0.46, 'FFT / Welch',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=COLORS['text_muted'])
        ax.text(0.5, 0.36, 'load waveform to\nactivate spectrum',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color=COLORS['text_muted'], style='italic')
        self.draw()

    # ── 数据注入（由主窗口在每次 plot_stream 后调用）────────────────────────
    def load_data(self, raw_t: list, raw_data: list, raw_meta: list):
        """Inject waveform arrays copied from the main seismic canvas."""
        self._raw_t    = raw_t
        self._raw_data = raw_data
        self._raw_meta = raw_meta
        # 道数变化时把活跃索引钳位到合法范围
        self._active_idx = min(self._active_idx, max(0, len(raw_t) - 1))

    def clear_data(self):
        """Remove all cached input data and restore the placeholder plot."""
        self._raw_t = []; self._raw_data = []; self._raw_meta = []
        self._active_idx = 0
        self._draw_placeholder()

    def set_active_trace(self, idx: int):
        """切换要显示的道，idx 是 _raw_t 中的下标"""
        if not self._raw_t:
            return
        self._active_idx = max(0, min(idx, len(self._raw_t) - 1))
        # 用上次记录的视图范围立即重绘
        self.update_spectra(*self._last_xlim)

    def set_options(self, **kwargs):
        """Update rendering options such as Welch/log/dB toggles."""
        for k, v in kwargs.items():
            if hasattr(self, f'opt_{k}'):
                setattr(self, f'opt_{k}', v)

    # ── 频谱计算 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _next_pow2(n: int) -> int:
        return max(16, 1 << (max(n, 1) - 1).bit_length())

    def _compute(self, t_arr, d_arr, sr, xmin, xmax):
        """返回 (freqs, amp_linear)，或 (None, None)"""
        mask = (t_arr >= xmin) & (t_arr <= xmax)
        seg  = d_arr[mask].astype(np.float64)
        n = len(seg)
        if n < 16:
            return None, None
        seg -= seg.mean()

        if self.opt_welch and _check_scipy():
            from scipy.signal import welch as sp_welch
            nperseg = min(self._next_pow2(n // 4 + 1), n)
            nperseg = max(16, nperseg)
            freqs, pxx = sp_welch(seg, fs=sr, window='hann',
                                   nperseg=nperseg, noverlap=nperseg // 2,
                                   scaling='spectrum')
            amp = np.sqrt(np.maximum(pxx, 0.0))
        else:
            win      = np.hanning(n)
            fft_vals = np.fft.rfft(seg * win)
            freqs    = np.fft.rfftfreq(n, d=1.0 / sr)
            amp      = np.abs(fft_vals) * 2.0 / (win.sum() + 1e-30)

        valid = freqs > 0
        freqs = freqs[valid]
        amp   = amp[valid]
        amp   = np.where(amp <= 0, 1e-30, amp)
        return freqs, amp

    # ── 主绘图入口（由 view_changed 信号或 set_active_trace 触发）─────────
    def update_spectra(self, xmin: float, xmax: float):
        """Recompute and redraw the amplitude spectrum for the current view."""
        if not self._raw_t:
            return

        # 记录视图范围供 set_active_trace 复用
        self._last_xlim = (xmin, xmax)

        idx = self._active_idx
        if idx >= len(self._raw_t):
            return

        stats, color = self._raw_meta[idx]
        sr = float(stats.sampling_rate)
        nyq = sr / 2.0
        span_s   = max(xmax - xmin, 1e-6)
        mode_str = 'Welch' if (self.opt_welch and _check_scipy()) else 'FFT'

        freqs, amp = self._compute(self._raw_t[idx], self._raw_data[idx],
                                    sr, xmin, xmax)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS['border_bright'])
            sp.set_linewidth(0.8)
        ax.tick_params(axis='x', colors=COLORS['text_secondary'], labelsize=8, length=3)
        ax.tick_params(axis='y', colors=COLORS['text_secondary'], labelsize=8, length=3)
        ax.tick_params(which='minor', length=2, color=COLORS['text_muted'])
        ax.grid(True, which='major', color=COLORS['grid'], linewidth=0.6, alpha=0.8)
        ax.grid(True, which='minor', color=COLORS['grid'], linewidth=0.3, alpha=0.5)

        ch_id = f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}   SR={sr:.0f}Hz"
        ax.set_title(ch_id, color=color, fontsize=8, fontweight='bold', pad=4)

        if freqs is None:
            ax.text(0.5, 0.5, 'window too short (< 16 samples)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color=COLORS['text_muted'], style='italic')
            ax.set_xlabel('Frequency  (Hz)',
                          color=COLORS['text_secondary'], fontsize=8)
            self.draw()
            return

        if self.opt_db:
            y_plot = 20.0 * np.log10(amp)
            y_label = 'Amplitude [dB re 1 count]'
        else:
            y_plot = amp
            y_label = 'Amplitude [count]'

        # ── 主谱线 + 填充 ────────────────────────────────────────────────
        ax.plot(freqs, y_plot, color=color,
                linewidth=1.0, alpha=0.9, rasterized=True)
        y_floor = y_plot.min() - (y_plot.max() - y_plot.min()) * 0.05
        ax.fill_between(freqs, y_plot, y_floor, color=color, alpha=0.12)

        # ── 频率轴刻度 ───────────────────────────────────────────────────
        if self.opt_log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # ── 振幅轴刻度 ───────────────────────────────────────────────────
        if not self.opt_db and self.opt_log_y:
            ax.set_yscale('log')
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        # ── X 轴范围 ─────────────────────────────────────────────────────
        ax.set_xlim(max(freqs[0], 0.01), nyq)

        # ── 轴标签 ───────────────────────────────────────────────────────
        ax.set_xlabel('Frequency  (Hz)',
                      color=COLORS['text_secondary'], fontsize=8)
        ax.set_ylabel(y_label, color=COLORS['text_secondary'], fontsize=8)

        # ── Nyquist 标线 ─────────────────────────────────────────────────
        ax.axvline(nyq * 0.999, color=COLORS['accent_amber'],
                   linewidth=0.9, linestyle=':', alpha=0.8, zorder=5)
        ax.text(0.985, 0.97, f"Nyq {nyq:.0f}Hz",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7, color=COLORS['accent_amber'],
                fontfamily='Consolas')

        # ── 峰值频率标线 ─────────────────────────────────────────────────
        peak_f = freqs[int(np.argmax(amp))]
        ax.axvline(peak_f, color=color, linewidth=0.8,
                   linestyle='--', alpha=0.5, zorder=4)
        ax.text(0.985, 0.84,
                f"Peak: {peak_f:.4g} Hz\nWin: {span_s:.2f}s\n{mode_str}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7, color=COLORS['text_muted'],
                fontfamily='Consolas')

        self.draw()


class PSDCanvas(FigureCanvas):
    """
    功率谱密度（PSD）画布。

    接受已预处理好的数据段（numpy array）及其元信息，计算 Welch PSD
    并叠加 ObsPy 内置的 Peterson NLNM/NHNM 背景噪声模型。

    当 is_physical=True 时，数据已转换为加速度（m/s²），
    PSD 单位为 dB rel. 1 (m/s²)²/Hz，可与 NLNM/NHNM 直接比较。
    """

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 4), dpi=100,
                          facecolor=COLORS['bg_card'],
                          layout='constrained')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        self.opt_log_x = True
        self.opt_db = True
        self.opt_nhnm = True
        self.opt_period_axis = False

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        for sp in ax.spines.values():
            sp.set_color(COLORS['border'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.58, 'PSD',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold',
                color=COLORS['accent_blue'], alpha=0.7, fontfamily='Consolas')
        ax.text(0.5, 0.44, 'Power Spectral Density',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color=COLORS['text_muted'])
        ax.text(0.5, 0.34, 'dB rel. 1 unit^2/Hz',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color=COLORS['text_muted'], style='italic')
        self.draw()

    def set_options(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, f'opt_{k}'):
                setattr(self, f'opt_{k}', v)

    @staticmethod
    def _next_pow2(n):
        return max(32, 1 << (max(n, 1) - 1).bit_length())

    def _suggest_nperseg(self, n):
        if n <= 256:
            return n
        target = max(256, n // 2)
        return min(self._next_pow2(target), n)

    def _compute_psd(self, data_seg, sr):
        seg = np.asarray(data_seg, dtype=np.float64)
        n = len(seg)
        if n < 32:
            return None, None, None

        if _check_scipy():
            from scipy.signal import welch as sp_welch
            nperseg = min(self._next_pow2(n // 4 + 1), n)
            nperseg = max(32, nperseg)
            freqs, pxx = sp_welch(
                seg, fs=sr, window='hann',
                nperseg=nperseg,
                noverlap=nperseg // 2,
                nfft=max(nperseg, self._next_pow2(nperseg)),
                detrend=False,
                scaling='density',
                return_onesided=True,
            )
            mode = f"Welch nperseg={nperseg}"
        else:
            win = np.hanning(n)
            scale = sr * np.sum(win ** 2)
            fft_vals = np.fft.rfft(seg * win)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            pxx = (np.abs(fft_vals) ** 2) / max(scale, 1e-30)
            if len(pxx) > 2:
                pxx[1:-1] *= 2.0
            mode = f"Periodogram n={n}"

        valid = freqs > 0
        freqs = freqs[valid]
        pxx = np.where(pxx[valid] <= 0, 1e-60, pxx[valid])
        return freqs, pxx, mode

    @staticmethod
    def _get_peterson_curves():
        try:
            from obspy.signal.spectral_estimation import get_nhnm, get_nlnm
            periods_n, db_n = get_nlnm()
            periods_h, db_h = get_nhnm()
            freqs_n = 1.0 / periods_n[::-1]
            freqs_h = 1.0 / periods_h[::-1]
            return freqs_n, db_n[::-1], freqs_h, db_h[::-1]
        except Exception:
            return None, None, None, None

    @staticmethod
    def _convert_psd_unit(freqs, pxx, src_unit, dst_unit):
        src = (src_unit or 'COUNTS').upper()
        dst = (dst_unit or src).upper()
        if src == dst:
            return pxx
        omega2 = (2.0 * np.pi * freqs) ** 2
        if src == 'DISP':
            if dst == 'VEL':
                return pxx * omega2
            if dst == 'ACC':
                return pxx * (omega2 ** 2)
        if src == 'VEL':
            if dst == 'ACC':
                return pxx * omega2
            if dst == 'DISP':
                return pxx / np.maximum(omega2, 1e-30)
        if src == 'ACC':
            if dst == 'VEL':
                return pxx / np.maximum(omega2, 1e-30)
            if dst == 'DISP':
                return pxx / np.maximum(omega2 ** 2, 1e-30)
        return pxx

    @staticmethod
    def _unit_label(unit):
        unit = (unit or 'COUNTS').upper()
        return {
            'COUNTS': 'count^2/Hz',
            'DISP': 'm^2/Hz',
            'VEL': '(m/s)^2/Hz',
            'ACC': '(m/s^2)^2/Hz',
        }.get(unit, 'unit^2/Hz')

    @staticmethod
    def _correct_response(freqs, pxx_counts, inventory, stats):
        try:
            seed_id = f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}"
            resp = inventory.get_response(seed_id, stats.starttime)
            h_vel = resp.get_evalresp_response_for_frequencies(freqs, output='VEL')
            h_sq = np.abs(h_vel) ** 2
            h_sq = np.where(h_sq < 1e-100, 1e-100, h_sq)
            pxx_vel = pxx_counts / h_sq
            pxx_acc = pxx_vel * (2.0 * np.pi * freqs) ** 2
            return pxx_acc, 'ACC', ''
        except Exception as e:
            return pxx_counts, 'COUNTS', str(e)

    def plot_psd(self, data_seg, sr, stats, color,
                 inventory=None, apply_resp=False,
                 preproc_info='', data_unit='COUNTS'):
        """Compute and draw a PSD for the supplied waveform segment."""
        freqs, pxx, mode_str = self._compute_psd(data_seg, float(sr))

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['bg_card'])
        self.fig.patch.set_facecolor(COLORS['bg_card'])
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS['border_bright'])
            sp.set_linewidth(0.8)
        ax.tick_params(axis='x', colors=COLORS['text_secondary'], labelsize=8, length=3)
        ax.tick_params(axis='y', colors=COLORS['text_secondary'], labelsize=8, length=3)
        ax.tick_params(which='minor', length=2, color=COLORS['text_muted'])
        ax.grid(True, which='major', color=COLORS['grid'], linewidth=0.6, alpha=0.8)
        ax.grid(True, which='minor', color=COLORS['grid'], linewidth=0.3, alpha=0.5)

        ch_id = f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}   SR={float(sr):.0f}Hz"
        ax.set_title(ch_id, color=color, fontsize=8, fontweight='bold', pad=4)

        if freqs is None:
            ax.text(0.5, 0.5, 'window too short (< 32 samples)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color=COLORS['text_muted'], style='italic')
            self.draw()
            return

        display_unit = (data_unit or 'COUNTS').upper()
        resp_err = ''
        if apply_resp and inventory is not None:
            pxx, display_unit, resp_err = self._correct_response(freqs, pxx, inventory, stats)

        can_compare_nm = self.opt_db and self.opt_nhnm
        nm_hint = ''
        if can_compare_nm:
            if display_unit == 'COUNTS':
                nm_hint = 'NLNM/NHNM requires physical units'
            else:
                if display_unit != 'ACC':
                    pxx = self._convert_psd_unit(freqs, pxx, display_unit, 'ACC')
                    nm_hint = f'Converted {display_unit} PSD to ACC for NM'
                    display_unit = 'ACC'

        x_vals = freqs
        x_label = 'Frequency (Hz)'
        if self.opt_period_axis:
            x_vals = 1.0 / freqs
            x_label = 'Period (s)'

        if self.opt_db:
            y_plot = 10.0 * np.log10(np.where(pxx > 0, pxx, 1e-200))
            y_label = f"PSD [dB rel. 1 {self._unit_label(display_unit)}]"
        else:
            y_plot = pxx
            y_label = f"PSD [{self._unit_label(display_unit)}]"

        has_nm = False
        if can_compare_nm and display_unit == 'ACC':
            fn, dn, fh, dh = self._get_peterson_curves()
            if fn is not None:
                has_nm = True
                if self.opt_period_axis:
                    pn = 1.0 / fn
                    ph = 1.0 / fh
                    p_common = x_vals[(x_vals >= max(min(pn), min(ph))) & (x_vals <= min(max(pn), max(ph)))]
                    if len(p_common) > 0:
                        dn_interp = np.interp(np.log10(p_common), np.log10(pn[::-1]), dn[::-1])
                        dh_interp = np.interp(np.log10(p_common), np.log10(ph[::-1]), dh[::-1])
                        ax.fill_between(p_common, dn_interp, dh_interp,
                                        color=COLORS['accent_blue'], alpha=0.08, zorder=1,
                                        label='NLNM-NHNM range')
                    ax.plot(pn, dn, color=COLORS['accent_blue'], linewidth=1.0,
                            linestyle='--', alpha=0.65, zorder=2, label='NLNM')
                    ax.plot(ph, dh, color='#E05252', linewidth=1.0,
                            linestyle='--', alpha=0.65, zorder=2, label='NHNM')
                else:
                    f_common = x_vals[(x_vals >= max(freqs[0], 1e-4)) & (x_vals <= float(sr) / 2.0)]
                    if len(f_common) > 0:
                        dn_interp = np.interp(np.log10(f_common), np.log10(fn), dn)
                        dh_interp = np.interp(np.log10(f_common), np.log10(fh), dh)
                        ax.fill_between(f_common, dn_interp, dh_interp,
                                        color=COLORS['accent_blue'], alpha=0.08, zorder=1,
                                        label='NLNM-NHNM range')
                    ax.plot(fn, dn, color=COLORS['accent_blue'], linewidth=1.0,
                            linestyle='--', alpha=0.65, zorder=2, label='NLNM')
                    ax.plot(fh, dh, color='#E05252', linewidth=1.0,
                            linestyle='--', alpha=0.65, zorder=2, label='NHNM')

        sort_idx = np.argsort(x_vals)
        x_plot = x_vals[sort_idx]
        y_plot = y_plot[sort_idx]
        ax.plot(x_plot, y_plot, color=color, linewidth=1.0,
                alpha=0.92, zorder=3, rasterized=True, label='PSD')
        if self.opt_db:
            y_floor = y_plot.min() - max((y_plot.max() - y_plot.min()) * 0.05, 1.0)
            ax.fill_between(x_plot, y_plot, y_floor, color=color, alpha=0.12, zorder=2)
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        else:
            ax.set_yscale('log')

        if self.opt_log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50)
            )
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        if self.opt_period_axis:
            ax.set_xlim(np.max(x_plot), np.min(x_plot))
        else:
            ax.set_xlim(max(freqs[0], 1e-4), float(sr) / 2.0)

        ax.set_xlabel(x_label, color=COLORS['text_secondary'], fontsize=8)
        ax.set_ylabel(y_label, color=COLORS['text_secondary'], fontsize=7.5)

        info_lines = [mode_str or 'PSD', f"Peak: {freqs[int(np.argmax(pxx))]:.4g} Hz"]
        if preproc_info:
            info_lines.append(preproc_info)
        if nm_hint:
            info_lines.append(nm_hint)
        if resp_err:
            info_lines.append(f"Resp err: {resp_err[:60]}")
        ax.text(0.985, 0.97, '\n'.join(info_lines),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color=COLORS['text_muted'], fontfamily='Consolas')

        if can_compare_nm and display_unit != 'ACC':
            ax.text(0.02, 0.98, 'NLNM/NHNM compare only valid for ACC PSD',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=7, color=COLORS['accent_amber'],
                    fontfamily='Consolas',
                    bbox=dict(fc=COLORS['bg_header'], ec=COLORS['accent_amber'],
                              alpha=0.85, boxstyle='round,pad=0.3', linewidth=0.8))
        elif can_compare_nm and 'Likely event' in preproc_info:
            ax.text(0.02, 0.98, 'Current window likely includes event energy',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=7, color=COLORS['accent_amber'],
                    fontfamily='Consolas',
                    bbox=dict(fc=COLORS['bg_header'], ec=COLORS['accent_amber'],
                              alpha=0.85, boxstyle='round,pad=0.3', linewidth=0.8))

        if has_nm:
            ax.legend(loc='lower right', fontsize=6.5, framealpha=0.85,
                      facecolor=COLORS['bg_header'], edgecolor=COLORS['border'],
                      labelcolor=COLORS['text_secondary'])

        self.draw()
