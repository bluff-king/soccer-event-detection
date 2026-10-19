#!/bin/bash
# Dynamic one-at-a-time job queue for a single GPU.
#   bash scripts/gpu_queue.sh outputs/local/queue.txt
# Each line of the queue file is a shell command; lines are executed in order, one at a time, and appended lines
# are picked up while the queue runs. Finished lines are recorded in <queue>.done (with exit code and duration),
# so the runner can be restarted without re-running them. Stops when a line "STOP" is reached.
# To stop it cleanly, append "STOP" (the running job finishes first); killing only the runner leaves its child job.
q=${1:-outputs/local/queue.txt}
done_f="$q.done"
lock="$q.lock"
touch "$q" "$done_f"
# one runner per queue file: two runners on one GPU both spill out of memory and crawl (seen: 20 s per step)
if [ -f "$lock" ] && kill -0 "$(cat "$lock")" 2>/dev/null; then
  echo "[queue] another runner (pid $(cat "$lock")) owns $q - exiting"; exit 1
fi
echo $$ > "$lock"
trap 'rm -f "$lock"' EXIT
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
  # wait until no other process uses the GPU (two jobs on a 4 GB card both spill and crawl); QUEUE_WAIT_GPU=0 skips
  if [ "${QUEUE_WAIT_GPU:-1}" = "1" ] && command -v nvidia-smi >/dev/null; then
    waited=0
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; do
      [ $waited = 0 ] && echo "[queue] $(date +%H:%M:%S) GPU busy ($(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '
' ' ')) - waiting before #$i"
      waited=1; sleep 60
    done
    [ $waited = 1 ] && echo "[queue] $(date +%H:%M:%S) GPU free"
  fi
  echo "[queue] $(date +%H:%M:%S) start #$i: $line"
  t0=$(date +%s)
  bash -c "$line"
  rc=$?
  printf '%s\t%s\t%s\t%ss\n' "$i" "$line" "$rc" "$(( $(date +%s) - t0 ))" >> "$done_f"
  echo "[queue] $(date +%H:%M:%S) done #$i rc=$rc"
done
