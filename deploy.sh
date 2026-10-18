#!/usr/bin/env bash
# Build the YOLOv5 Lambda image and deploy the whole stack with AWS SAM.
# Run from the deploy/ folder on the EC2 build instance (Docker + SAM CLI installed).
#   ./deploy.sh                       # stack "yolo-detect" in ap-southeast-2
#   STACK=my-stack REGION=us-east-1 ./deploy.sh
set -euo pipefail
STACK="${STACK:-yolo-detect}"          # lowercase only - used in the bucket name
REGION="${REGION:-ap-southeast-2}"
MEMORY="${MEMORY:-3008}"

echo ">> sam build (docker build of the Lambda image - first build takes ~5-10 min)"
sam build --template-file template.yaml

echo ">> sam deploy"
sam deploy \
  --stack-name "$STACK" --region "$REGION" \
  --resolve-s3 --resolve-image-repos \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides "MemorySize=$MEMORY" \
  --no-confirm-changeset --no-fail-on-empty-changeset

out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
API_URL=$(out ApiUrl); KEY_ID=$(out ApiKeyId); BUCKET=$(out BucketName)
API_KEY=$(aws apigateway get-api-key --api-key "$KEY_ID" --include-value --region "$REGION" --query value --output text)

echo ">> uploading sample images to s3://$BUCKET/uploads/"
for f in ../code/sample_images/*; do [ -f "$f" ] && aws s3 cp "$f" "s3://$BUCKET/uploads/$(basename "$f")" --region "$REGION"; done

cat > stack_outputs.env << EOV
API_URL=$API_URL
API_KEY=$API_KEY
BUCKET=$BUCKET
REGION=$REGION
STACK=$STACK
EOV
echo
echo "Deployed. Values saved to deploy/stack_outputs.env (do NOT commit this file):"
echo "  API_URL=$API_URL"
echo "  BUCKET=$BUCKET"
echo "Smoke test:  curl $API_URL/health"
