#!/usr/bin/env bash
# Long-job launcher — kills the recurring "nohup died when shell exited" failure
# mode (3 incidents on 2026-09-10: first pip, v6, v6.1).
#
# Usage: scripts/train_wrapper.sh <logname> <command...>
#   scripts/train_wrapper.sh bluff_v61_train python3 -m nn.training ...
#
# The child survives the parent shell (setsid = own session, detached stdio),
# stdout/stderr append to /tmp/<logname>.log. Log the printed PID in
# AGENT_CHAT.md per worksplit.md §7 rule 5.
set -euo pipefail
LOG="/tmp/${1:?usage: train_wrapper.sh <logname> <cmd...>}.log"
shift
setsid nohup "$@" >>"$LOG" 2>&1 </dev/null &
PID=$!
echo "PID $PID -> $LOG"
