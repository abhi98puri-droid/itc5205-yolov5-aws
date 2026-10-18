"""
ITC5205 Assignment 2 - Image Object Identification on AWS
AWS Lambda handler (container image) running YOLOv5s.

Routes (API Gateway REST API, Lambda proxy integration):
    GET  /health                 -> liveness + cold-start info (no API key)
    POST /upload-url             -> pre-signed S3 PUT URL for a new image
    POST /detect                 -> run YOLOv5 on an image already in S3
    GET  /results/{image_id}     -> read a stored detection record from DynamoDB

Design notes
    * The model is loaded ONCE at module level (init phase) and re-used by every
      warm invocation - this is the main cold-start optimisation.
    * The function is stateless: everything it produces is written to S3
      (annotated image + JSON) and DynamoDB (queryable record). /tmp is only
      used as a scratch/config directory.
    * Images never travel through API Gateway. Clients upload straight to S3
      with a pre-signed URL, which avoids the 10 MB API Gateway / 6 MB Lambda
      payload limits.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

_INIT_START = time.time()

# ---------------------------------------------------------------- logging ---
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def log(level, message, **fields):
    """Structured (JSON) log line - easy to query in CloudWatch Logs Insights."""
    logger.log(level, json.dumps({"msg": message, **fields}, default=str))


# ----------------------------------------------------------------- config ---
BUCKET = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
UPLOAD_PREFIX = os.environ.get("UPLOAD_PREFIX", "uploads/")
DETECTED_PREFIX = os.environ.get("DETECTED_PREFIX", "detected/")
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "results/")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
RECORD_TTL_DAYS = int(os.environ.get("RECORD_TTL_DAYS", "30"))
YOLO_DIR = os.environ.get("YOLO_DIR", "/opt/yolov5")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "/opt/yolov5/yolov5s.pt")
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WRITE_ATTEMPTS = 3

# botocore retries transient/throttling errors itself ("adaptive" mode adds
# client-side rate limiting); the app-level retry below adds logging and a
# clear failure path on top of that.
_boto_cfg = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=20,
)
s3 = boto3.client("s3", config=_boto_cfg)
table = boto3.resource("dynamodb", config=_boto_cfg).Table(TABLE_NAME)

# ------------------------------------------------------------------ model ---
import numpy as np  # noqa: E402
import torch  # noqa: E402  (import after env/config so failures log clearly)
from PIL import Image, ImageOps  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
_t = time.time()
MODEL = torch.hub.load(YOLO_DIR, "custom", path=YOLO_WEIGHTS, source="local", verbose=False)
MODEL.conf = CONF_THRESHOLD
MODEL.eval()
MODEL_LOAD_SECONDS = round(time.time() - _t, 3)
INIT_SECONDS = round(time.time() - _INIT_START, 3)
_COLD = True  # flips to False after the first invocation in this sandbox

log(logging.INFO, "init complete", init_seconds=INIT_SECONDS,
    model_load_seconds=MODEL_LOAD_SECONDS, torch=torch.__version__)


# ---------------------------------------------------------------- helpers ---
class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def to_decimal(obj):
    """DynamoDB rejects Python floats - convert recursively to Decimal."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 4)))
    if isinstance(obj, dict):
        return {k: to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_decimal(v) for v in obj]
    return obj


def from_decimal(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_decimal(v) for v in obj]
    return obj


_NON_RETRYABLE = {"AccessDenied", "NoSuchBucket", "ValidationException",
                  "ResourceNotFoundException", "ConditionalCheckFailedException"}


def with_retry(what, fn, request_id):
    """Run a storage write with exponential back-off and logging.

    Returns fn()'s result, or raises ApiError(502) after the final attempt.
    """
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            return fn()
        except (ClientError, BotoCoreError) as exc:
            code = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else type(exc).__name__
            final = attempt == WRITE_ATTEMPTS or code in _NON_RETRYABLE
            log(logging.ERROR if final else logging.WARNING, "storage write failed",
                target=what, attempt=attempt, error_code=code, error=str(exc),
                request_id=request_id)
            if final:
                raise ApiError(502, f"Could not write {what} ({code}). See CloudWatch logs, request {request_id}.")
            time.sleep(0.4 * (2 ** (attempt - 1)))


def response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,x-api-key",
        },
        "body": json.dumps(body, default=str),
    }


def parse_body(event):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError(400, "Request body must be valid JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object.")
    return body


def validate_key(key):
    if not key or not isinstance(key, str):
        raise ApiError(400, "Field 'key' (S3 object key of the image) is required.")
    if ".." in key or key.startswith("/"):
        raise ApiError(400, "Invalid object key.")
    if not key.lower().endswith(ALLOWED_EXT):
        raise ApiError(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")
    return key


# ----------------------------------------------------------------- routes ---
def route_health(event, ctx):
    return response(200, {
        "status": "ok",
        "cold_start": _COLD,
        "init_seconds": INIT_SECONDS,
        "model": os.path.basename(YOLO_WEIGHTS),
        "memory_mb": ctx.memory_limit_in_mb,
    })


def route_upload_url(event, ctx):
    body = parse_body(event)
    filename = os.path.basename(str(body.get("filename", "image.jpg"))).replace(" ", "_")
    if not filename.lower().endswith(ALLOWED_EXT):
        raise ApiError(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")
    content_type = body.get("content_type", "image/jpeg")
    key = f"{UPLOAD_PREFIX}{uuid.uuid4().hex[:12]}-{filename}"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=300,
    )
    return response(200, {"key": key, "upload_url": url, "content_type": content_type,
                          "expires_in": 300,
                          "next": "PUT the file to upload_url, then POST /detect with {\"key\": key}"})


def route_detect(event, ctx):
    request_id = ctx.aws_request_id
    body = parse_body(event)
    key = validate_key(body.get("key"))
    bucket = body.get("bucket", BUCKET)
    if bucket != BUCKET:  # never let callers point the function at other buckets
        raise ApiError(403, "Only the project bucket can be used.")

    t0 = time.time()
    # 1. read image from S3
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head["ContentLength"] > MAX_IMAGE_BYTES:
            raise ApiError(413, f"Image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        # Without s3:ListBucket, S3 answers HeadObject for a missing key with
        # 403 instead of 404 (it will not reveal whether the key exists). We keep
        # least privilege and translate both into a clean 404 for the client.
        if code in ("404", "NoSuchKey", "NotFound", "403", "AccessDenied", "Forbidden"):
            raise ApiError(404, f"s3://{bucket}/{key} not found.")
        raise
    t_read = time.time()

    try:
        img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    except Exception:
        raise ApiError(400, "File is not a readable image.")

    # 2. inference - np.array() gives a *writable* RGB copy; passing the PIL
    # image directly makes YOLOv5's render() fail with newer OpenCV builds
    # ("img marked as output argument, but provided NumPy array marked as readonly").
    with torch.no_grad():
        results = MODEL(np.array(img), size=IMG_SIZE)
    t_infer = time.time()

    df = results.pandas().xyxy[0]
    detections = [
        {
            "label": row["name"],
            "class_id": int(row["class"]),
            "confidence": round(float(row["confidence"]), 4),
            "box": {k: round(float(row[k]), 1) for k in ("xmin", "ymin", "xmax", "ymax")},
        }
        for _, row in df.iterrows()
    ]
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    # 3. annotated image -> S3
    annotated = Image.fromarray(results.render()[0])
    buf = BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    image_id = uuid.uuid4().hex
    stem = os.path.splitext(os.path.basename(key))[0]
    annotated_key = f"{DETECTED_PREFIX}{image_id}-{stem}.jpg"
    results_key = f"{RESULTS_PREFIX}{image_id}.json"
    now = datetime.now(timezone.utc)

    record = {
        "image_id": image_id,
        "source_bucket": bucket,
        "source_key": key,
        "annotated_key": annotated_key,
        "results_key": results_key,
        "created_at": now.isoformat(),
        "created_date": now.strftime("%Y-%m-%d"),
        "model": os.path.basename(YOLO_WEIGHTS),
        "conf_threshold": CONF_THRESHOLD,
        "image_width": img.width,
        "image_height": img.height,
        "object_count": len(detections),
        "label_counts": counts,
        "detections": detections,
        "timings_ms": {
            "s3_read": round((t_read - t0) * 1000, 1),
            "inference": round((t_infer - t_read) * 1000, 1),
        },
        "cold_start": _COLD,
        "request_id": request_id,
    }

    with_retry("annotated image to S3", lambda: s3.put_object(
        Bucket=BUCKET, Key=annotated_key, Body=buf.getvalue(), ContentType="image/jpeg"), request_id)
    with_retry("results JSON to S3", lambda: s3.put_object(
        Bucket=BUCKET, Key=results_key, Body=json.dumps(record, indent=2).encode(),
        ContentType="application/json"), request_id)

    # 4. queryable record -> DynamoDB
    item = to_decimal(dict(record, expires_at=int((now + timedelta(days=RECORD_TTL_DAYS)).timestamp())))
    with_retry("record to DynamoDB", lambda: table.put_item(Item=item), request_id)
    t_end = time.time()

    record["timings_ms"]["total"] = round((t_end - t0) * 1000, 1)
    record["annotated_url"] = s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": annotated_key}, ExpiresIn=3600)
    record["init_seconds"] = INIT_SECONDS

    log(logging.INFO, "detection complete", image_id=image_id, source_key=key,
        objects=len(detections), labels=counts, cold_start=_COLD,
        timings_ms=record["timings_ms"], request_id=request_id)
    return response(200, record)


def route_results(event, ctx):
    image_id = (event.get("pathParameters") or {}).get("image_id")
    if not image_id:
        raise ApiError(400, "image_id path parameter is required.")
    item = table.get_item(Key={"image_id": image_id}).get("Item")
    if not item:
        raise ApiError(404, f"No record for image_id {image_id}.")
    return response(200, from_decimal(item))


ROUTES = {
    ("GET", "/health"): route_health,
    ("POST", "/upload-url"): route_upload_url,
    ("POST", "/detect"): route_detect,
    ("GET", "/results/{image_id}"): route_results,
}


# ---------------------------------------------------------------- handler ---
def handler(event, context):
    global _COLD
    method = event.get("httpMethod", "POST")
    resource = event.get("resource", "/detect")  # direct invokes default to /detect
    route = ROUTES.get((method, resource))
    try:
        if route is None:
            raise ApiError(404, f"No route for {method} {resource}.")
        return route(event, context)
    except ApiError as exc:
        log(logging.WARNING, "request rejected", status=exc.status, error=exc.message,
            route=resource, request_id=context.aws_request_id)
        return response(exc.status, {"error": exc.message, "request_id": context.aws_request_id})
    except Exception as exc:  # unexpected - log full trace, return generic 500
        logger.exception("unhandled error")
        return response(500, {"error": "Internal error.", "detail": type(exc).__name__,
                              "request_id": context.aws_request_id})
    finally:
        _COLD = False
