#!/usr/bin/env bash
# Delete everything after marking (stops all charges except the EC2 instance,
# which you stop/terminate in the console).
set -euo pipefail
source "$(dirname "$0")/stack_outputs.env"
aws s3 rm "s3://$BUCKET" --recursive --region "$REGION"
sam delete --stack-name "$STACK" --region "$REGION" --no-prompts
