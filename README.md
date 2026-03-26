# SeismoView

**SeismoView** 是一款专为地震数据分析设计的跨平台桌面应用，基于 Python 和 PyQt5 开发。它提供直观的图形界面，支持常见地震数据格式（MiniSEED、SAC、SEED 等），帮助研究人员和学生快速查看、处理和分析波形数据。

## ✨ 主要功能

- **数据加载**：拖拽或通过文件菜单打开，支持 MiniSEED、SAC、SEED 等格式
- **头段信息展示**：自动解析并列出所有通道的完整元数据（台网、台站、采样率、起止时间等）
- **高性能波形绘图**：采用 LTTB + Min-Max 智能降采样，轻松处理数小时连续数据
- **实时频谱分析**：可切换 FFT / Welch 方法，支持对数频率轴与 dB 显示
- **震相拾取**：支持 P 波和 S 波手动拾取，自动计算 S-P 时差与估算震中距，结果可导出 CSV/TXT
- **波形预处理**：提供去均值、去趋势、归一化、数字滤波（带通/低通/高通/陷波）及振幅缩放
- **视图交互**：滚轮缩放、拖拽平移、一键重置
- **导出功能**：波形图可保存为 PNG/SVG/PDF，拾取结果可导出为 CSV 或文本

- **更多功能等待作者偷懒后添加**

## 🖥️ 平台支持

- **Windows 10/11 (64-bit)** – 提供打包好的便携式 `.exe` 文件（无需安装 Python）
- 理论上支持 Linux/macOS（需自行配置 Python 环境）

## 📦 技术栈

- Python 3.9+
- PyQt5 (GUI)
- ObsPy (地震数据读取与处理)
- Matplotlib (绘图)
- NumPy / SciPy (可选，用于滤波和高级分析)

- ---------------------------------

# SeismoView — 模块化工程说明

## 文件结构

```
SeismoView/
├── main.py               # ▶ 程序入口，运行此文件启动
├── config.py             # 全局配置：颜色主题、样式表、字体初始化
├── loader.py             # 数据加载线程（DataLoaderThread）
├── canvas_seismic.py     # 主波形画布（SeismicCanvas）
│                         #   · LTTB + MinMax 智能降采样
│                         #   · P/S 波交互拾取 + 十字游标
│                         #   · 动态分辨率重绘（缩放/平移）
├── canvas_spectrum.py    # 频谱画布
│                         #   · SpectrumCanvas（振幅谱，FFT/Welch）
│                         #   · PSDCanvas（功率谱密度 + NLNM/NHNM 参考曲线）
├── spectrum_windows.py   # 独立弹出分析窗口
│                         #   · SpectrumWindow（含 NavigationToolbar）
│                         #   · PSDWindow（含 NavigationToolbar）
├── panels.py             # 侧边栏面板（HeaderPanel 通道列表）
└── main_window.py        # 主窗口（MainWindow）
                          #   · 菜单 / 工具栏 / 上下文面板
                          #   · 波形预处理（去均值/去趋势/归一化/滤波）
                          #   · 震相拾取管理 + CSV 导出
                          #   · 图片导出
```

## 模块依赖关系

```
main.py
  └─ main_window.py
       ├─ config.py          (COLORS, STYLESHEET 等常量)
       ├─ loader.py          (DataLoaderThread)
       ├─ canvas_seismic.py  (SeismicCanvas)
       │    └─ config.py
       ├─ panels.py          (HeaderPanel)
       │    └─ config.py
       └─ spectrum_windows.py (SpectrumWindow, PSDWindow)
            ├─ config.py
            └─ canvas_spectrum.py (SpectrumCanvas, PSDCanvas)
                 └─ config.py
```

## 运行方式

```bash
# 安装依赖
pip install obspy PyQt5 matplotlib numpy scipy

# 启动程序
python main.py
```

## 打包为 Windows EXE（不建议，scipy包在win端运行有bug，obspy依赖包使用pyinstaller打包特别复杂）

```bash
pip install pyinstaller
pyinstaller --onedir --windowed --name SeismoView --paths . main.py
# 输出在 dist/SeismoView/SeismoView.exe
```
