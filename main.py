#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
Application entry point for SeismoView.

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
import sys
import traceback
import faulthandler
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__),
                        "seismoview_crash.log")

def _log(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# 原生崩溃也尽量记录
_log("\n" + "=" * 80)
_log(f"Start: {datetime.now().isoformat()}")
faulthandler.enable(open(LOG_PATH, "a", encoding="utf-8"))

def excepthook(exc_type, exc_value, exc_tb):
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _log(text)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = excepthook

_log("before imports")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

_log("before config import")
from config import (
    APP_NAME,
    APP_ORGANIZATION,
    APP_VERSION,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
)
_log("before main_window import")
from main_window import MainWindow

_log("imports done")

def main() -> None:
    """Start the Qt application and open an optional file argument."""
    _log("enter main")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    _log("QApplication created")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)
    app.setFont(QFont(DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE))

    _log("before MainWindow()")
    window = MainWindow()
    _log("MainWindow created")
    
    window.show()
    _log("window shown")

    # Support `python main.py <path>` for quick local testing.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window._load_paths([sys.argv[1]])

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()