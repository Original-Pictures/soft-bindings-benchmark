#!/usr/bin/env bash
# Run a shell snippet on the bench instance through SSM send-command and print
# its output. Usage: remote.sh [-t timeout_s] [-b] 'commands...'
#   -b  fire and forget (prints CommandId); poll later with remote.sh -w <id>
set -euo pipefail
STATE=${BENCH_STATE:-${BENCH_WORK:-$HOME/op-fp-bench-work}/gpu-state.env}
# shellcheck disable=SC1090
. "$STATE"
AWS=(aws --profile "$PROFILE" --region "$REGION")
timeout=3600; bg=0; wait_id=""
while getopts "t:bw:" o; do case $o in t) timeout=$OPTARG;; b) bg=1;; w) wait_id=$OPTARG;; *) exit 64;; esac; done
shift $((OPTIND-1))

if [[ -z "$wait_id" ]]; then
  # SSM runs /bin/sh (dash); hand the snippet to bash explicitly.
  params=$(python3 -c 'import json,shlex,sys; print(json.dumps({"commands":["exec /bin/bash -c "+shlex.quote("set -o pipefail; export HOME=/root; cd /root; "+sys.argv[1])],"executionTimeout":[sys.argv[2]]}))' "$1" "$timeout")
  wait_id=$("${AWS[@]}" ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
      --parameters "$params" --timeout-seconds 600 --query Command.CommandId --output text)
  [[ $bg == 1 ]] && { echo "$wait_id"; exit 0; }
fi
while :; do
  st=$("${AWS[@]}" ssm get-command-invocation --command-id "$wait_id" --instance-id "$IID" --query Status --output text 2>/dev/null || echo Pending)
  case $st in Pending|InProgress|Delayed) sleep 5;; *) break;; esac
done
"${AWS[@]}" ssm get-command-invocation --command-id "$wait_id" --instance-id "$IID" \
  --query '[StandardOutputContent,StandardErrorContent]' --output text | tail -c 20000
echo "[status=$st]"
[[ $st == Success ]]
