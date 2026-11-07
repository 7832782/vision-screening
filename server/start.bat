@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 视力筛查 AI 后端
echo 正在启动 Python 后端...
python app.py
if errorlevel 1 (
  echo.
  echo 启动失败。请检查：
  echo   1. 是否已安装 Python 3.12 及 requirements.txt
  echo   2. 是否已运行 python download_models.py 下载模型
  pause
)
