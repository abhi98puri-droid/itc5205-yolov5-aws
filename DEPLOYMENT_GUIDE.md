# ITC5205 A2 — Deployment Guide & Evidence Checklist

YOLOv5 object detection on AWS: **API Gateway → Lambda (container image) → S3 + DynamoDB**, built on an EC2 instance.

Region used throughout: **ap-southeast-2 (Sydney)**. Stack name: **yolo-detect** (must be lowercase).

> 📸 = take a screenshot here for the report. Save them into the `images/` folder as you go, named by step (e.g. `03-yolov5-detect-ec2.png`). The checklist at the end maps every screenshot to a rubric row.

---

## 0. Architecture (what you are building)

```
 Client (curl / client_demo.py)
   │ 1. POST /upload-url  (x-api-key)          ┌──────────────────────────┐
   ├──────────────────────────────────────────►│ API Gateway (REST, prod) │  API key + usage plan
   │ 3. POST /detect {"key": "uploads/..."}     │  /health  /upload-url    │  (20 req/s, burst 40,
   │ 5. GET  /results/{image_id}                │  /detect  /results/{id}  │   5000/day)
   │                                            └────────────┬─────────────┘
   │ 2. PUT image (pre-signed URL)                           │ Lambda proxy
   ▼                                                         ▼
 ┌──────────────────────┐   GetObject uploads/*   ┌─────────────────────────────┐
 │ S3 bucket            │◄────────────────────────│ Lambda: yolo-detect-detector│
 │  uploads/  (input)   │   PutObject detected/*  │  container image (ECR)      │
 │  detected/ (boxes)   │◄────────────────────────│  Python 3.11, torch CPU,    │
 │  results/  (JSON)    │                         │  YOLOv5s v7.0 baked in      │
 └──────────────────────┘                         └──────┬──────────────┬───────┘
                                       PutItem/GetItem   │              │ logs, metrics, traces
                              ┌──────────────────────────▼───┐   ┌──────▼───────────────────┐
                              │ DynamoDB yolo-detect-detections│   │ CloudWatch Logs / X-Ray │
                              │ PK image_id, GSI by-date,     │   │ error alarm             │
                              │ auto-scaling 5–50 RCU/WCU, TTL │   └─────────────────────────┘
                              └───────────────────────────────┘
 EC2 (t3.medium, Amazon Linux 2023): YOLOv5 install + verification, Docker build, SAM deploy → ECR
```

Why each piece (put this reasoning in the report's architecture section):

| Component | Purpose | Key decision |
|---|---|---|
| EC2 | Install/verify YOLOv5 (Task 1.2); build the Docker image | Lambda can't build images; EC2 gives a Linux x86 build host matching Lambda |
| ECR + Lambda container image | Run YOLOv5 serverless | PyTorch is far over the 250 MB zip limit; images allow 10 GB |
| API Gateway (REST) | Public front door, API keys, throttling, quotas | REST (not HTTP API) because it supports API keys + usage plans |
| S3 | Input images, annotated images, JSON results | Pre-signed upload URLs keep images out of API Gateway (10 MB limit) |
| DynamoDB | Queryable detection records | Lookup by `image_id`; GSI `by-date` lists a day's detections |
| CloudWatch / X-Ray | Logs, metrics, cold-start evidence | Structured JSON logs → Logs Insights queries |

---

## 1. Task 1.1 — AWS environment

### 1.1 IAM role for the EC2 build instance
IAM → Roles → Create role → *AWS service* → **EC2** → attach **PowerUserAccess** and **IAMFullAccess** → name `sudhir-yolov5-ec2-role`. 📸

(SAM needs to create the Lambda execution role, hence IAMFullAccess. This is deliberately broad for a build box — a good candidate when you answer reflection question (b).)

> If your account is an **AWS Academy Learner Lab**, you cannot create roles: use the existing `LabInstanceProfile`/`LabRole` and tell your tutor, because SAM's role creation will be blocked.

### 1.2 Launch EC2
EC2 → Launch instance:
- Name `yolo-builder`, AMI **Amazon Linux 2023**, type **t3.medium** (t2/t3.micro runs out of memory installing PyTorch)
- Key pair: create/download one
- Storage: **30 GiB gp3** (the default 8 GiB is too small for Docker + PyTorch)
- Advanced details → IAM instance profile: `sudhir-yolov5-ec2-role`

Connect with **EC2 Instance Connect**. 📸 instance summary page

### 1.3 Install tools on EC2
```bash
sudo dnf install -y docker git python3.11 python3.11-pip mesa-libGL unzip
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && newgrp docker
curl -sSLo sam.zip https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
unzip -q sam.zip -d sam-install && sudo ./sam-install/install
aws --version && sam --version && docker --version && aws sts get-caller-identity
```
📸 the version output + `get-caller-identity` (proves the instance role works)

Get the project onto EC2 (after you push it to your GitHub repo):
```bash
git clone https://github.com/<you>/itc5205-yolov5-aws.git ~/itc5205-yolov5-aws
```

---

## 2. Task 1.2 — Install and verify YOLOv5 on EC2

```bash
git clone https://github.com/ultralytics/yolov5 ~/yolov5 && cd ~/yolov5 && git checkout v7.0
python3.11 -m venv ~/yolo-env && source ~/yolo-env/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r ~/itc5205-yolov5-aws/code/requirements.txt
curl -sSL -o yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# official YOLOv5 check
python detect.py --weights yolov5s.pt --source data/images
# same loading path the Lambda uses
python ~/itc5205-yolov5-aws/code/ec2_verify_yolov5.py --yolo ~/yolov5 --image data/images/bus.jpg --out ~/verify_bus.jpg
```
Expected: `bus.jpg` → 4 person + 1 bus; `zidane.jpg` → 2 person + 2 tie; inference a few hundred ms on CPU.
📸 terminal output of both commands. Download `~/verify_bus.jpg` (or `runs/detect/exp/bus.jpg`) and put it in `images/`.

Why the pinned versions: torch ≥ 2.6 changed `torch.load` defaults (breaks YOLOv5 v7.0 weights loading) and numpy 2.x breaks wheels compiled for numpy 1.x. Mention this under *challenges* in the report.

---

## 3. Task 2 — Build and deploy (Lambda + API + S3 + DynamoDB)

```bash
deactivate 2>/dev/null; cd ~/itc5205-yolov5-aws/deploy
./deploy.sh
```
First build takes 5–10 minutes (downloads PyTorch, bakes model into image; the Dockerfile's last `RUN` line loads the model as a build-time smoke test, so a broken image can't be deployed). 📸 the end of `sam build`, and the `sam deploy` outputs table.

Then load the outputs into your shell:
```bash
source stack_outputs.env && echo $API_URL
```
**Never** commit `stack_outputs.env` (it holds your API key) or paste the key into the report.

📸 Console screenshots: CloudFormation stack resources, Lambda function (Image tab, configuration → memory/timeout/env vars), ECR repository, S3 bucket, DynamoDB table (+ Indexes tab + Additional settings → auto-scaling), API Gateway resources tree.

### 3.1 Test the Lambda directly (Task 2.2)
Lambda console → Test → new event → paste `code/events/detect_event.json` → Test. 📸 the result + log output (look for the `"msg": "init complete"` line with `init_seconds`).

Error-handling tests (📸 each response):
```bash
curl -s -X POST "$API_URL/detect" -H "x-api-key: $API_KEY" -d '{}'                              # 400 missing key
curl -s -X POST "$API_URL/detect" -H "x-api-key: $API_KEY" -d '{"key":"uploads/nope.jpg"}'     # 404
curl -s -X POST "$API_URL/detect" -H "x-api-key: $API_KEY" -d '{"key":"uploads/a.txt"}'        # 400 type
curl -s -X POST "$API_URL/detect" -H "x-api-key: $API_KEY" -d '{"key":"uploads/bus.jpg","bucket":"someone-else"}'  # 403
```

---

## 4. Task 3 — API Gateway integration and security

```bash
curl -s "$API_URL/health"                                                    # open route
curl -s -X POST "$API_URL/detect" -d '{"key":"uploads/bus.jpg"}'             # no key → 403 Forbidden
curl -s -X POST "$API_URL/detect" -H "x-api-key: $API_KEY" -d '{"key":"uploads/bus.jpg"}' | python3 -m json.tool
```
⚠️ **The very first call after a deploy may return `504 Endpoint request timed out`.** API Gateway gives up after 29 s, and a brand-new container image can take longer than that to pull and initialise. Lambda keeps running, so call again 30 s later and it succeeds. Record this — it's genuine material for reflection questions (a) and (e).

📸 all three responses, and API Gateway → Usage plans → your plan (throttle + quota) and API keys.

Concurrency check (Task 3.2): see section 6.

---

## 5. Task 4 — End-to-end pipeline and storage validation

From your laptop (Python 3 only, no AWS credentials needed):
```bash
python code/client_demo.py --api "$API_URL" --key "$API_KEY" --image my_own_photo.jpg
```
Use **your own** photos (street, desk, pets, kitchen) as well as the samples — it shows originality and gives more interesting results. 📸 the terminal output and the downloaded `annotated_output.jpg`.

Validate storage (📸 each):
```bash
aws s3 ls "s3://$BUCKET/" --recursive
aws dynamodb scan --table-name yolo-detect-detections --max-items 3
aws dynamodb query --table-name yolo-detect-detections --index-name by-date \
  --key-condition-expression "created_date = :d" \
  --expression-attribute-values "{\":d\":{\"S\":\"$(date -u +%F)\"}}"
```
And in the console: S3 → `detected/` object preview; DynamoDB → Explore items.

Write-failure handling (Task 4.1): every S3/DynamoDB write is retried 3× with exponential back-off on top of botocore's adaptive retries; after the last attempt the API returns **502** with the request ID and logs `"storage write failed"`. To demonstrate: temporarily remove `dynamodb:PutItem` from the role in the IAM console, call `/detect`, 📸 the 502 response and the CloudWatch log lines, then restore it.

---

## 6. Task 5 — Performance and scaling

### 6.1 Memory experiment (Task 5.1)
Lambda CPU is proportional to memory, so this is the main performance lever. For each of **1024, 2048, 3008 MB**:
```bash
cd ~/itc5205-yolov5-aws/deploy && MEMORY=2048 ./deploy.sh     # rebuild is cached, takes ~1 min
cd .. && python code/load_test.py --api "$API_URL" --key "$API_KEY" --keys uploads/bus.jpg uploads/zidane.jpg --requests 20 --concurrency 1
```
(Change memory through the redeploy, not the console: the API calls the `live` alias, which only moves to a new version on deploy.) Record mean warm inference ms, init seconds and Max Memory Used (from the REPORT lines) in a table and chart it. Cost per request = GB-seconds × price, so faster-at-higher-memory can be *cheaper*; work out which memory size is the best value from your own numbers.

Other optimisations already built in (explain in the report): model loaded once at init, CPU-only torch (smaller image → faster cold start), weights baked into the image (no runtime download), YOLOv5 auto-install disabled (saved ~0.7 s of init in testing), structured logging.

### 6.2 Load / scaling test (Task 5.2)
```bash
python code/load_test.py --api "$API_URL" --key "$API_KEY" --keys uploads/bus.jpg uploads/zidane.jpg --requests 60 --concurrency 10
```
📸 the summary + CloudWatch Lambda metrics (Invocations, Duration, ConcurrentExecutions, Throttles) and API Gateway metrics (Count, Latency, 4XX/5XX).

⚠️ Check your account limit first: `aws lambda get-account-settings` → `ConcurrentExecutions`. New accounts are often capped at **10**; above that you will see 429/throttles. That's a real finding — report it, and request an increase via Service Quotas if you have time.

Auto-scaling policies:
- **DynamoDB**: target tracking at 70% for read and write capacity, 5–50 units (in the template). 📸 table → Additional settings.
- **Lambda**: scales out by itself; to remove cold starts, enable provisioned concurrency with its own target-tracking policy, measure, then switch it off (it is billed per hour):
  ```bash
  ./deploy/provisioned_concurrency.sh on 1 5
  # wait until READY, re-run the load test, compare cold-start count
  ./deploy/provisioned_concurrency.sh off
  ```
  This requires account concurrency > 10.

### 6.3 Cold-start evidence (reflection e)
1. Wait 15+ minutes (sandboxes get reclaimed), then run the load test with `--concurrency 5`. The output counts requests with `cold_start=true` and their `init_seconds`.
2. CloudWatch → Logs Insights → log group `/aws/lambda/yolo-detect-detector`:
   ```
   filter @type = "REPORT"
   | stats count(*) as invocations, count(@initDuration) as cold_starts,
           avg(@initDuration) as avg_init_ms, avg(@duration) as avg_duration_ms,
           max(@maxMemoryUsed/1000/1000) as max_mem_mb by bin(5m)
   ```
   Only cold invocations have an `Init Duration` in their REPORT line, which is how you confirm them.
3. X-Ray → Traces: a cold trace shows an **Initialization** segment; a warm one doesn't.

📸 all three. Three independent sources agreeing is strong evidence.

---

## 7. User manual (short)

1. Get the API URL and an API key from the project owner.
2. `GET /health` — checks the service is up (no key).
3. `POST /upload-url` with `{"filename":"cat.jpg","content_type":"image/jpeg"}` → returns `upload_url` and `key`.
4. `PUT` the image bytes to `upload_url` with header `Content-Type: image/jpeg` (within 5 minutes).
5. `POST /detect` with `{"key":"<key from step 3>"}` → list of objects, counts, confidence, boxes, and an `annotated_url` (valid for 1 hour).
6. `GET /results/{image_id}` → retrieve the stored record later.

All calls except `/health` need header `x-api-key`. Or simply run `client_demo.py`, which performs steps 3–6.

Formats: jpg, jpeg, png, bmp, webp; max 15 MB.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `504 Endpoint request timed out` on first call | Cold start of a large image > 29 s API Gateway limit | Call again; warm with `/health`; provisioned concurrency |
| `403 Forbidden` / `{"message":"Forbidden"}` | Missing or wrong `x-api-key` | Use the key from `stack_outputs.env` |
| `429 Too Many Requests` | Usage plan throttle (20 rps) or Lambda concurrency limit | Lower concurrency; raise limits |
| `502 Could not write ...` | IAM permission or DynamoDB throttling | Check CloudWatch log `storage write failed` lines |
| `sam build` fails "no space left on device" | 8 GiB EBS | Use 30 GiB; `docker system prune -af` |
| pip killed during torch install on EC2 | t2/t3.micro out of RAM | Use t3.medium |
| `Unable to import module 'app'` | Image built for arm64 (e.g. on a Mac M-series) | Build on the x86 EC2 instance |
| Bucket already exists | Stack name reused in another account/region | Change `STACK=` |
| `SignatureDoesNotMatch` on upload | PUT `Content-Type` differs from the one requested | Send the same content type |

---

## 9. Cost control
Stop the EC2 instance whenever you're not building. After marking: `./deploy/teardown.sh`, then terminate EC2 and delete the ECR repo SAM created. Turn provisioned concurrency off as soon as you've measured it.

---

## 10. Evidence checklist → rubric

| Rubric row (marks) | Evidence to include |
|---|---|
| AWS setup + YOLOv5 (2) | IAM role, EC2 summary, tool versions, `detect.py` + `ec2_verify_yolov5.py` output, verification image |
| Lambda dev + testing (2) | Dockerfile + `app.py` snippets (model at init, retries, logging), console test, error-case responses |
| API Gateway + integration (8 — the biggest row) | Resource tree, usage plan + key, 403 without key, successful `/detect`, `/results`, client demo run, load-test output, 504 first-call story |
| Architecture / distribution (3) | Diagram (section 0) redrawn in draw.io with AWS icons; explain separate processing (Lambda) vs storage (S3/DynamoDB) vs build (EC2) |
| DynamoDB, validation, performance, scaling (3) | Table design (PK, GSI, TTL, auto-scaling), scan/query output, S3 listing, memory table + chart, load test, cold-start evidence |
| Report (3) | All sections a–g of the brief, numbered figures, references |
| Video (2) + communication (2) | Your own walkthrough of a live call; be ready to explain every design choice above |
