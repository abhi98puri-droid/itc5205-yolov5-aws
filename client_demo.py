"""
User-side demo of the full pipeline (no AWS credentials needed - only the API key).

    python client_demo.py --api https://xxxx.execute-api.ap-southeast-2.amazonaws.com/prod \
                          --key YOUR_API_KEY --image dog.jpg

Steps: 1) ask the API for a pre-signed upload URL  2) PUT the image straight to S3
       3) POST /detect  4) download the annotated image  5) GET /results/{image_id}
Uses only the Python standard library.
"""
import argparse, json, mimetypes, os, sys, time, urllib.request, urllib.error


def call(method, url, api_key=None, body=None, raw=None, headers=None):
    hdrs = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    if api_key:
        hdrs["x-api-key"] = api_key
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        payload, status = e.read(), e.code
    ms = round((time.time() - t) * 1000)
    try:
        return status, json.loads(payload or b"{}"), ms
    except json.JSONDecodeError:
        return status, payload, ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True, help="ApiUrl output of the stack")
    p.add_argument("--key", required=True, help="API key value")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="annotated_output.jpg")
    a = p.parse_args()
    api = a.api.rstrip("/")
    ctype = mimetypes.guess_type(a.image)[0] or "image/jpeg"

    st, up, ms = call("POST", f"{api}/upload-url", a.key, {"filename": os.path.basename(a.image), "content_type": ctype})
    print(f"[1] POST /upload-url -> {st} ({ms} ms) key={up.get('key')}")
    if st != 200:
        sys.exit(up)

    with open(a.image, "rb") as f:
        st, _, ms = call("PUT", up["upload_url"], raw=f.read(), headers={"Content-Type": ctype})
    print(f"[2] PUT image to S3   -> {st} ({ms} ms)")

    st, res, ms = call("POST", f"{api}/detect", a.key, {"key": up["key"]})
    print(f"[3] POST /detect      -> {st} ({ms} ms)")
    if st != 200:
        sys.exit(res)
    print(f"    objects={res['object_count']} labels={res['label_counts']} "
          f"cold_start={res['cold_start']} lambda_timings_ms={res['timings_ms']}")
    for d in res["detections"]:
        print(f"      - {d['label']:<14} {d['confidence']:.2f}  {d['box']}")

    urllib.request.urlretrieve(res["annotated_url"], a.out)
    print(f"[4] annotated image saved to {a.out}")

    st, rec, ms = call("GET", f"{api}/results/{res['image_id']}", a.key)
    print(f"[5] GET /results/{res['image_id']} -> {st} ({ms} ms) record has {rec.get('object_count')} objects")


if __name__ == "__main__":
    main()
