#!/usr/bin/env bash
# One-line progress summary from the bench instance (for monitoring loops).
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/remote.sh" -t 60 'echo "image=$(ls /opt/bench/results/image 2>/dev/null | grep -c json) audio=$(ls /opt/bench/results/audio 2>/dev/null | grep -c json) video=$(ls /opt/bench/results/video 2>/dev/null | grep -c json) stab=$(ls /opt/bench/results/stability 2>/dev/null | wc -l) procs=$(pgrep -fc "bench_|stability.py") tracebacks=$(cat /opt/bench/work/logs/*.log | grep -c Traceback)"' 2>&1 | grep '^image='
