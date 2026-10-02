#!/usr/bin/env bash
# Tear down everything provision.sh created. Idempotent; finds resources by the
# purpose=watermark-bench tag / fixed names only, and verifies each deletion.
# Appends a timestamped record to $BENCH_WORK/teardown.log for the README.
set -uo pipefail

PROFILE=${AWS_PROFILE_BENCH:?set AWS_PROFILE_BENCH}
REGION=${AWS_REGION_BENCH:-us-east-1}
ACCOUNT=${BENCH_AWS_ACCOUNT:?set BENCH_AWS_ACCOUNT}
TAG=watermark-bench
ROLE=op-watermark-bench-ssm
BUCKET=op-watermark-bench-${ACCOUNT}-20260923
LOG=${BENCH_WORK:-$HOME/op-wm-bench-work}/teardown.log
AWS=(aws --profile "$PROFILE" --region "$REGION")
PROD=${BENCH_PROTECTED_INSTANCE:-none}  # an instance teardown must never touch

acct=$("${AWS[@]}" sts get-caller-identity --query Account --output text)
[[ "$acct" == "$ACCOUNT" ]] || { echo "refusing: account $acct" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

ids=$("${AWS[@]}" ec2 describe-instances --filters Name=tag:purpose,Values=$TAG \
      Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
      --query 'Reservations[].Instances[].InstanceId' --output text)
for iid in $ids; do
  [[ "$iid" == "$PROD" ]] && { log "REFUSED to terminate prod $iid"; exit 3; }
  "${AWS[@]}" ec2 terminate-instances --instance-ids "$iid" >/dev/null
  "${AWS[@]}" ec2 wait instance-terminated --instance-ids "$iid"
  log "instance $iid: $("${AWS[@]}" ec2 describe-instances --instance-ids "$iid" --query 'Reservations[0].Instances[0].State.Name' --output text)"
done
vols=$("${AWS[@]}" ec2 describe-volumes --filters Name=tag:purpose,Values=$TAG --query 'Volumes[].VolumeId' --output text)
for v in $vols; do "${AWS[@]}" ec2 delete-volume --volume-id "$v" && log "volume $v deleted"; done
log "tagged volumes remaining: $("${AWS[@]}" ec2 describe-volumes --filters Name=tag:purpose,Values=$TAG --query 'length(Volumes)')"

for sg in $("${AWS[@]}" ec2 describe-security-groups --filters Name=tag:purpose,Values=$TAG --query 'SecurityGroups[].GroupId' --output text); do
  for _ in 1 2 3 4 5 6; do "${AWS[@]}" ec2 delete-security-group --group-id "$sg" 2>/dev/null && break; sleep 10; done
  log "security group $sg remaining: $("${AWS[@]}" ec2 describe-security-groups --filters Name=group-id,Values="$sg" --query 'length(SecurityGroups)')"
done

if aws --profile "$PROFILE" iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws --profile "$PROFILE" iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null
  aws --profile "$PROFILE" iam delete-instance-profile --instance-profile-name "$ROLE"
fi
if aws --profile "$PROFILE" iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws --profile "$PROFILE" iam delete-role-policy --role-name "$ROLE" --policy-name bench-bucket-only 2>/dev/null
  aws --profile "$PROFILE" iam detach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null
  aws --profile "$PROFILE" iam delete-role --role-name "$ROLE"
fi
aws --profile "$PROFILE" iam get-role --role-name "$ROLE" >/dev/null 2>&1 && log "role $ROLE STILL EXISTS" || log "role $ROLE deleted (get-role: NoSuchEntity)"
aws --profile "$PROFILE" iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1 && log "instance profile STILL EXISTS" || log "instance profile $ROLE deleted"

if "${AWS[@]}" s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  "${AWS[@]}" s3 rm "s3://$BUCKET" --recursive --only-show-errors
  "${AWS[@]}" s3api delete-bucket --bucket "$BUCKET"
fi
"${AWS[@]}" s3api head-bucket --bucket "$BUCKET" 2>/dev/null && log "bucket $BUCKET STILL EXISTS" || log "bucket $BUCKET deleted (head-bucket: 404)"
[[ "$PROD" != none ]] && log "prod $PROD state (untouched): $("${AWS[@]}" ec2 describe-instances --instance-ids $PROD --query 'Reservations[0].Instances[0].State.Name' --output text)"
