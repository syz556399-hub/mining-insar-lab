#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ -x .venv/bin/python ]]; then
  task_review_run="${1:-runs/first_review}"
  if [[ ! -f "$task_review_run/manifest.json" ]]; then
    .venv/bin/python simulator_review.py build --run "$task_review_run" || exit 1
  fi
  exec .venv/bin/python simulator_review.py serve --run "$task_review_run" --port 8781 --open
fi
print '请先按照 README 创建 .venv 并安装依赖：'
print 'python3 -m venv .venv'
print 'source .venv/bin/activate'
print 'python -m pip install -r requirements.txt'
print '完成后再双击此文件。'
read 'task_reply?按回车关闭。'
