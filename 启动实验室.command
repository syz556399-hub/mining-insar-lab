#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python server.py --open "$@"
fi
print '请先按照 README 创建 .venv 并安装依赖：'
print 'python3 -m venv .venv'
print 'source .venv/bin/activate'
print 'python -m pip install -r requirements.txt'
print '完成后再双击此文件。'
read 'task_reply?按回车关闭。'
