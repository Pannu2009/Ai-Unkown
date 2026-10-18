#!/bin/bash
# NeuroBrain v3 training watchdog: restart the run if it died (e.g. VM replacement).
# - Does nothing while training is running or after DONE.
# - Corpus guard: if data/ changed since the run started, a --resume would crash
#   on vocab mismatch (this killed v2 at iter 11500). New corpus files must be
#   staged for the NEXT version, never added mid-run.
#   NOTE: when launching a fresh training run manually, regenerate the
#   fingerprint first: find data -type f -exec md5sum {} + | sort > model_v2/corpus.md5
cd ~/workspace/neuro-brain || exit 0
[ -f model_v2/DONE ] && exit 0
# pgrep bracket trick: [t]rain_v2.py matches the training process but not
# this script's own wrapper shell (whose cmdline contains the literal check text).
pgrep -f "[t]rain_v2.py" >/dev/null 2>&1 && exit 0
# --- corpus guard ---
if [ -f model_v2/corpus.md5 ]; then
  if ! find data -type f -exec md5sum {} + 2>/dev/null | sort | \
       diff -q model_v2/corpus.md5 - >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) watchdog: data/ changed since run start - NOT restarting (vocab mismatch risk; stage new files for next version)" >> train_v3.log
    exit 0
  fi
fi
python3 -c "import torch" 2>/dev/null || pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
nohup python3 train_v2.py >> train_v3.log 2>&1 &
echo "training restarted at $(date -u +%H:%M) UTC" >> train_v3.log
