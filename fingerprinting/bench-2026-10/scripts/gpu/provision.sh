#!/usr/bin/env bash
# Provision the TEMPORARY fingerprint-bench GPU instance (one instance; same guard rails as the 2026-09 watermark bench). Idempotent: re-running
# reuses the tagged SG / role / bucket / running instance instead of duplicating.
#
# Guard rails:
#  - one SSO profile and one account (BENCH_AWS_ACCOUNT); refuses any other account.
#  - never touches instances it did not create (everything is found by the
#    purpose=fingerprint-bench tag); no other instance is referenced.
#  - no inbound rules; access is SSM Session Manager / send-command only.
#  - instance-initiated shutdown = terminate, plus a 72 h `shutdown -h +4320`.
#  - the G-instance vCPU quota was 16 with 4 in use elsewhere, so the bench
#    uses an 8-vCPU g5.2xlarge (same A10G as the 2026-09 watermark bench).
set -euo pipefail

PROFILE=${AWS_PROFILE_BENCH:?set AWS_PROFILE_BENCH}
REGION=${AWS_REGION_BENCH:-us-east-1}
ACCOUNT=${BENCH_AWS_ACCOUNT:?set BENCH_AWS_ACCOUNT}
TAG=fingerprint-bench
# The CPU helper (provision.sh --cpu) shares SG, role and bucket but carries its own
# instance tag so each script finds exactly the instance it launched.
ITAG=$TAG; ROLE_NAME=op-fingerprint-bench
if [[ "${1:-}" == "--cpu" ]]; then ITAG=fingerprint-bench-cpu; ROLE_NAME=op-fingerprint-bench-cpu; INSTANCE_TYPES=${INSTANCE_TYPES:-c5a.16xlarge c5a.24xlarge}; fi
ROLE=op-fingerprint-bench-ssm
# One bucket per run (BENCH_RUN_DATE): 20260926 first run, 20260928 re-run with per-query outputs.
BUCKET=op-fingerprint-bench-${ACCOUNT}-${BENCH_RUN_DATE:-20260928}
STATE=${BENCH_STATE:-${BENCH_WORK:-$HOME/op-fp-bench-work}/gpu-state.env}
AWS=(aws --profile "$PROFILE" --region "$REGION")
TAGSPEC="{Key=purpose,Value=$ITAG},{Key=owner,Value=op},{Key=auto-terminate-after,Value=72h},{Key=Name,Value=$ROLE_NAME}"

acct=$("${AWS[@]}" sts get-caller-identity --query Account --output text)
[[ "$acct" == "$ACCOUNT" ]] || { echo "refusing: account $acct != $ACCOUNT" >&2; exit 2; }
"${AWS[@]}" sts get-caller-identity --query Arn --output text

vpc=$("${AWS[@]}" ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

# --- security group: no inbound rules --------------------------------------------------
sg=$("${AWS[@]}" ec2 describe-security-groups --filters Name=vpc-id,Values="$vpc" Name=tag:purpose,Values=$TAG \
      --query 'SecurityGroups[0].GroupId' --output text)
if [[ "$sg" == "None" ]]; then
  sg=$("${AWS[@]}" ec2 create-security-group --vpc-id "$vpc" --group-name op-fingerprint-bench \
        --description "fingerprint bench - no inbound, SSM only" \
        --tag-specifications "ResourceType=security-group,Tags=[$TAGSPEC]" --query GroupId --output text)
fi
inbound=$("${AWS[@]}" ec2 describe-security-groups --group-ids "$sg" --query 'length(SecurityGroups[0].IpPermissions)')
[[ "$inbound" == "0" ]] || { echo "refusing: $sg has inbound rules" >&2; exit 3; }

# --- scratch bucket (bench artefacts only, never customer data) -------------------------
if ! "${AWS[@]}" s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  "${AWS[@]}" s3api create-bucket --bucket "$BUCKET" >/dev/null
  "${AWS[@]}" s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  "${AWS[@]}" s3api put-bucket-tagging --bucket "$BUCKET" \
    --tagging "TagSet=[{Key=purpose,Value=$TAG},{Key=owner,Value=op}]"
fi

# --- scoped SSM role + instance profile -------------------------------------------------
if ! aws --profile "$PROFILE" iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws --profile "$PROFILE" iam create-role --role-name "$ROLE" \
    --tags Key=purpose,Value=$TAG Key=owner,Value=op \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws --profile "$PROFILE" iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
fi
aws --profile "$PROFILE" iam put-role-policy --role-name "$ROLE" --policy-name bench-bucket-only \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
if ! aws --profile "$PROFILE" iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws --profile "$PROFILE" iam create-instance-profile --instance-profile-name "$ROLE" \
    --tags Key=purpose,Value=$TAG Key=owner,Value=op >/dev/null
  aws --profile "$PROFILE" iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 15  # IAM propagation before EC2 can use the profile
fi

# --- instance -------------------------------------------------------------------------
iid=$("${AWS[@]}" ec2 describe-instances --filters Name=tag:purpose,Values=$ITAG \
       Name=instance-state-name,Values=pending,running,stopping,stopped \
       --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [[ "$iid" == "None" ]]; then
  ami=$("${AWS[@]}" ssm get-parameter \
        --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
        --query Parameter.Value --output text)
  userdata=$(printf '#!/bin/bash\nshutdown -h +4320 "fingerprint-bench 72h safety timer"\n' | base64)
  for itype in ${INSTANCE_TYPES:-g5.2xlarge g6.2xlarge g5.xlarge}; do
    if iid=$("${AWS[@]}" ec2 run-instances --image-id "$ami" --instance-type "$itype" --count 1 \
        --security-group-ids "$sg" --iam-instance-profile Name="$ROLE" \
        --instance-initiated-shutdown-behavior terminate \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}' \
        --user-data "$userdata" \
        --tag-specifications "ResourceType=instance,Tags=[$TAGSPEC]" "ResourceType=volume,Tags=[$TAGSPEC]" \
        --query 'Instances[0].InstanceId' --output text 2>/tmp/op-fp-run.err); then
      echo "launched $iid ($itype, $ami)"; break
    fi
    echo "run-instances $itype failed: $(tail -1 /tmp/op-fp-run.err)" >&2; iid=None
  done
  [[ "$iid" != "None" ]] || { echo "no instance type could be launched" >&2; exit 4; }
fi
"${AWS[@]}" ec2 wait instance-running --instance-ids "$iid"
mkdir -p "$(dirname "$STATE")"
cat > "$STATE" <<EOF
IID=$iid
SG=$sg
BUCKET=$BUCKET
ROLE=$ROLE
PROFILE=$PROFILE
REGION=$REGION
EOF
"${AWS[@]}" ec2 describe-instances --instance-ids "$iid" \
  --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,ImageId,LaunchTime,InstanceLifecycle]' --output text
echo "waiting for SSM registration..."
for _ in $(seq 60); do
  ping=$("${AWS[@]}" ssm describe-instance-information --filters Key=InstanceIds,Values="$iid" \
        --query 'InstanceInformationList[0].PingStatus' --output text)
  [[ "$ping" == "Online" ]] && { echo "SSM online"; exit 0; }
  sleep 10
done
echo "SSM agent did not come online in 10 min" >&2; exit 5
