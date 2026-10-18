ITC5205 Assignment 2 - Image Object Identification System on AWS
Student name:   Sudhir Puri
Student number: 12203235

Description
-----------
A serverless object-detection service. Clients upload an image to Amazon S3
through a pre-signed URL issued by Amazon API Gateway, then call POST /detect.
API Gateway invokes an AWS Lambda function packaged as a container image that
runs the YOLOv5s model (COCO, 80 classes). The function draws bounding boxes,
stores the annotated image and a JSON result in S3, and writes a queryable
record to DynamoDB. The image is built and YOLOv5 is verified on an EC2
instance; the stack is deployed with AWS SAM.

Folders
-------
code/     Lambda source (app.py), Dockerfile, requirements, test events,
          client_demo.py (user-side demo), load_test.py (scaling / cold
          starts), ec2_verify_yolov5.py (YOLOv5 verification on EC2),
          sample_images/
deploy/   template.yaml (AWS SAM), deploy.sh, provisioned_concurrency.sh,
          teardown.sh, iam-policy-lambda.json
images/   all figures used in the report
DEPLOYMENT_GUIDE.md  step-by-step developer manual, user manual, troubleshooting

Quick start (on the EC2 build instance)
---------------------------------------
cd deploy && ./deploy.sh
source stack_outputs.env
python ../code/client_demo.py --api "$API_URL" --key "$API_KEY" --image ../code/sample_images/bus.jpg

Dataset
-------
Pre-trained YOLOv5s weights (COCO 2017, https://cocodataset.org) from
https://github.com/ultralytics/yolov5/releases/tag/v7.0 ; test images are the
YOLOv5 sample images plus the author's own photos.
