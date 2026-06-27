#!/bin/bash

# 1. 切换到脚本所在目录 (等同于 Windows 的 cd /d "%~dp0")
cd "$(dirname "$0")"

# 2. 设置默认的 Python 执行程序为 python3
PYTHON_EXE="python3"

# 3. 检查是否存在我们在上一节创建的虚拟环境 (venv)
# 如果存在，就自动使用虚拟环境里的 python，避免污染系统环境
if [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
    echo "检测到虚拟环境，使用: $PYTHON_EXE"
fi

# 4. 打印启动信息并运行后端程序
echo "Starting FastAPI backend..."
"$PYTHON_EXE" run_backend.py

# 5. 程序退出后暂停 (等同于 Windows 的 pause)
echo ""
read -p "Backend exited. Press [Enter] key to close this window..."

