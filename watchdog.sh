#!/bin/bash
# Overnight watchdog: restart NeuroBrain v3 training if it died (e.g. VM replacement).
cd ~/workspace/neuro-brain || exit 0
[ -f model_v2/DONE ] && exit 0
# pgrep bracket trick: [t]rain_v2.py matches the training process but not
# this script's own wrapper shell (whose cmdline contains the literal check text).
pgrep -f "[t]rain_v2.py" >/dev/null 2>&1 && exit 0
python3 -c "import torch" 2>/dev/null || pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
nohup python3 train_v2.py --resume > train_v3.log 2>&1 &
echo "training restarted at $(date -u +%H:%M) UTC"
