#!/usr/bin/env bash
# One-line progress summary from the bench instance (for monitoring loops).
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/remote.sh" -t 60 'echo "desc=$(ls /opt/bench/work/desc 2>/dev/null | tr "\n" " ") results=$(find /opt/bench/results -name "*.json" 2>/dev/null | wc -l) procs=$(pgrep -fc "bench_|extraction.py|edits.py") tracebacks=$(cat /opt/bench/work/logs/*.log 2>/dev/null | grep -c Traceback) queue=$(tail -1 /opt/bench/work/logs/queue.log 2>/dev/null)"' 2>&1 | grep '^desc='
