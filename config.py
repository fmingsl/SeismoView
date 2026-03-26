#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Main application configuration for SeismoView.

Purpose
-------
Centralize UI theme constants, supported file extensions, plotting defaults,
and matplotlib font initialization so that hard-coded values are not scattered
across the code base.

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

from __future__ import annotations

import os
import matplotlib
import matplotlib.font_manager as fm

APP_NAME = "SeismoView"
APP_VERSION = "1.0"
APP_ORGANIZATION = "SeismoView"
DEFAULT_FONT_FAMILY = "Segoe UI"
DEFAULT_FONT_SIZE = 9
DEFAULT_EXPORT_DPI = 200

SUPPORTED_SEISMIC_EXTENSIONS = {
    '.mseed', '.miniseed', '.seed', '.sac', '.gse2',
    '.gse', '.seisan', '.dat', '.msd', '.ascii', '.segy', '.su',
}

COLORS = {
    'bg_deep': '#F5F5F5',
    'bg_panel': '#FFFFFF',
    'bg_card': '#FFFFFF',
    'bg_header': '#F5F5F5',
    'accent_blue': '#2D7FF9',
    'accent_green': '#27AE60',
    'accent_amber': '#F39C12',
    'accent_red': '#E74C3C',
    'text_primary': '#111827',
    'text_secondary': '#374151',
    'text_muted': '#6B7280',
    'border': '#E5E7EB',
    'border_bright': '#D1D5DB',
    'waveform': '#00D4FF',
    'waveform_alt': '#00FF9F',
    'grid': '#E5E7EB',
    'selection': '#8C9197',
}

WAVEFORM_COLORS = [
    '#1D4ED8', '#059669', '#D97706', '#FF6B9D',
    '#A78BFA', '#34D399', '#F97316', '#60A5FA',
]


def setup_chinese_font() -> None:
    """Configure matplotlib with a best-effort CJK-capable font.

    The application may render mixed Chinese and English labels. This helper
    checks common Windows/macOS/Linux font families first, then falls back to a
    direct Windows font file lookup when available.
    """
    candidates = [
        'Microsoft YaHei',      # 微软雅黑
        'Microsoft YaHei UI',
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'NSimSun',
        'FangSong',
        'KaiTi',
        'DengXian',
        'Source Han Sans CN',   # 思源黑体
        'Noto Sans CJK SC',
        'PingFang SC',          # macOS
        'Hiragino Sans GB',
        'WenQuanYi Micro Hei',  # Linux
        'AR PL UMing CN',
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)

    if chosen:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        windows_font = r'C:\Windows\Fonts\msyh.ttc'
        if os.path.exists(windows_font):
            fm.fontManager.addfont(windows_font)
            prop = fm.FontProperties(fname=windows_font)
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        # 若仍找不到，将图表中的中文文字改为英文（在代码里已做兼容）

    matplotlib.rcParams['axes.unicode_minus'] = False


setup_chinese_font()

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_deep']};
}}
QWidget {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
    font-size: 13px;
}}
QMenuBar {{
    background-color: {COLORS['bg_deep']};
    color: {COLORS['text_primary']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 2px 4px;
    font-size: 13px;
}}
QMenuBar::item:selected {{
    background-color: {COLORS['bg_header']};
    color: {COLORS['accent_blue']};
    border-radius: 4px;
}}
QMenu {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_bright']};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item:selected {{
    background-color: {COLORS['selection']};
    color: {COLORS['accent_blue']};
    border-radius: 4px;
}}
QMenu::separator {{
    height: 1px;
    background: {COLORS['border']};
    margin: 4px 8px;
}}
QToolBar {{
    background-color: {COLORS['bg_deep']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px 8px;
    spacing: 4px;
}}
QToolBar::separator {{
    background: {COLORS['border_bright']};
    width: 1px;
    margin: 4px 6px;
}}
QPushButton {{
    background-color: {COLORS['bg_header']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_bright']};
    border-radius: 6px;
    padding: 4px 4px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {COLORS['selection']};
    border-color: {COLORS['accent_blue']};
    color: {COLORS['accent_blue']};
}}
QPushButton:pressed {{
    background-color: #0D2040;
}}
QPushButton#accent_btn {{
    background-color: #00375A;
    border-color: {COLORS['accent_blue']};
    color: {COLORS['accent_blue']};
    font-weight: 600;
}}
QPushButton#accent_btn:hover {{
    background-color: #004A75;
}}
QTreeWidget {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    outline: none;
    font-size: 12px;
}}
QTreeWidget::item {{
    padding: 5px 8px;
    border-bottom: 1px solid {COLORS['border']};
}}
QTreeWidget::item:selected {{
    background-color: {COLORS['selection']};
    color: {COLORS['accent_blue']};
}}
QTreeWidget::item:hover {{
    background-color: {COLORS['bg_header']};
}}
QHeaderView::section {{
    background-color: {COLORS['bg_header']};
    color: {COLORS['text_secondary']};
    border: none;
    border-bottom: 1px solid {COLORS['border_bright']};
    border-right: 1px solid {COLORS['border']};
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QSplitter::handle {{
    background-color: {COLORS['border']};
}}
QSplitter::handle:hover {{
    background-color: {COLORS['accent_blue']};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QStatusBar {{
    background-color: {COLORS['bg_deep']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
    font-size: 12px;
    padding: 2px 8px;
}}
QScrollBar:vertical {{
    background: {COLORS['bg_deep']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border_bright']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent_blue']};
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_deep']};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['border_bright']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['accent_blue']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QProgressBar {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    text-align: center;
    color: {COLORS['text_primary']};
    font-size: 11px;
    height: 16px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent_blue']}, stop:1 {COLORS['accent_green']});
    border-radius: 3px;
}}
QLabel#section_title {{
    color: {COLORS['text_secondary']};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 6px 8px 4px 8px;
    border-bottom: 1px solid {COLORS['border']};
}}
QLabel#stat_value {{
    color: {COLORS['accent_blue']};
    font-size: 13px;
    font-weight: 600;
    font-family: 'Consolas', 'Courier New', monospace;
}}
QFrame#separator {{
    background-color: {COLORS['border']};
    max-height: 1px;
}}
QGroupBox {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: 11px;
    color: {COLORS['text_secondary']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {COLORS['text_secondary']};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}}
"""
