#!/bin/bash
# Dynamic one-at-a-time job queue for a single GPU.
#   bash scripts/gpu_queue.sh outputs/local/queue.txt
# Each line of the queue file is a shell command; lines are executed in order, one at a time, and appended lines
# are picked up while the queue runs. Finished lines are recorded in <queue>.done (with exit code and duration),
# so the runner can be restarted without re-running them. Stops when a line "STOP" is reached.
q=${1:-outputs/local/queue.txt}
done_f="$q.done"
touch "$q" "$done_f"
i=0
while true; do
  i=$((i + 1))
  line=$(sed -n "${i}p" "$q")
  if [ -z "$line" ]; then
    sleep 20; i=$((i - 1)); continue
  fi
  [ "$line" = "STOP" ] && break
  case "$line" in \#*) continue ;; esac
  if grep -qxF "$i	$line" <(cut -f1,2 "$done_f"); then continue; fi
  echo "[queue] $(date +%H:%M:%S) start #$i: $line"
  t0=$(date +%s)
  bash -c "$line"
  rc=$?
  printf '%s\t%s\t%s\t%ss\n' "$i" "$line" "$rc" "$(( $(date +%s) - t0 ))" >> "$done_f"
  echo "[queue] $(date +%H:%M:%S) done #$i rc=$rc"
done
