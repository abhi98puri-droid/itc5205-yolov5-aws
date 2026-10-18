#!/usr/bin/env bash
# Task 5.1/5.2 experiment: remove cold starts with provisioned concurrency on the
# "live" alias, plus a target-tracking auto-scaling policy (scale between MIN and
# MAX at 70% utilisation).  COSTS MONEY WHILE ENABLED - run "off" after measuring.
#   ./provisioned_concurrency.sh on   [MIN] [MAX]
#   ./provisioned_concurrency.sh off
# Needs an account concurrency quota above 10 (check: aws lambda get-account-settings).
set -euo pipefail
source "$(dirname "$0")/stack_outputs.env"
FN="${STACK}-detector"; ALIAS=live
RID="function:${FN}:${ALIAS}"
case "${1:-}" in
  on)
    MIN="${2:-1}"; MAX="${3:-5}"
    aws lambda put-provisioned-concurrency-config --function-name "$FN" --qualifier "$ALIAS" \
      --provisioned-concurrent-executions "$MIN" --region "$REGION"
    aws application-autoscaling register-scalable-target --service-namespace lambda \
      --resource-id "$RID" --scalable-dimension lambda:function:ProvisionedConcurrency \
      --min-capacity "$MIN" --max-capacity "$MAX" --region "$REGION"
    aws application-autoscaling put-scaling-policy --service-namespace lambda \
      --resource-id "$RID" --scalable-dimension lambda:function:ProvisionedConcurrency \
      --policy-name "${FN}-pc-tracking" --policy-type TargetTrackingScaling --region "$REGION" \
      --target-tracking-scaling-policy-configuration \
      '{"TargetValue":0.7,"PredefinedMetricSpecification":{"PredefinedMetricType":"LambdaProvisionedConcurrencyUtilization"}}'
    echo "Provisioned concurrency $MIN..$MAX on $FN:$ALIAS. Wait until status READY:"
    echo "  aws lambda get-provisioned-concurrency-config --function-name $FN --qualifier $ALIAS --region $REGION"
    ;;
  off)
    aws application-autoscaling deregister-scalable-target --service-namespace lambda \
      --resource-id "$RID" --scalable-dimension lambda:function:ProvisionedConcurrency --region "$REGION" || true
    aws lambda delete-provisioned-concurrency-config --function-name "$FN" --qualifier "$ALIAS" --region "$REGION" || true
    echo "Provisioned concurrency removed."
    ;;
  *) echo "usage: $0 on [MIN] [MAX] | off"; exit 1 ;;
esac
