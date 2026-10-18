#!/bin/bash
# Overnight watchdog: restart NeuroBrain v2 training if it died (e.g. VM replacement).
cd ~/workspace/neuro-brain || exit 0
[ -f model_v2/DONE ] && exit 0
pgrep -f "train_v2.py" >/dev/null 2>&1 && exit 0
python3 -c "import torch" 2>/dev/null || pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
nohup python3 train_v2.py --resume > train_v2.log 2>&1 &
echo "training restarted at $(date -u +%H:%M) UTC"
