"""
Scalability / cold-start test for the detection API (Task 5.2 + reflection e).

    python load_test.py --api <ApiUrl> --key <API key> \
        --keys uploads/bus.jpg uploads/zidane.jpg --requests 60 --concurrency 10

Fires N POST /detect requests with C in parallel and reports status codes,
latency percentiles and how many requests landed on a cold Lambda sandbox
(the function returns cold_start=true on the first request a sandbox serves).
Writes every request to load_test_results.csv for charts in the report.
"""
import argparse, csv, json, statistics, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor


def detect(api, key, s3_key, i):
    req = urllib.request.Request(f"{api}/detect", method="POST",
                                 data=json.dumps({"key": s3_key}).encode(),
                                 headers={"Content-Type": "application/json", "x-api-key": key})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, body = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        status, body = e.code, {"error": e.read().decode(errors="replace")[:200]}
    except Exception as e:  # timeouts, connection resets
        status, body = 0, {"error": str(e)}
    return {
        "i": i, "status": status, "client_ms": round((time.time() - t) * 1000),
        "cold_start": body.get("cold_start"), "init_seconds": body.get("init_seconds"),
        "inference_ms": (body.get("timings_ms") or {}).get("inference"),
        "lambda_total_ms": (body.get("timings_ms") or {}).get("total"),
        "objects": body.get("object_count"), "error": body.get("error") or body.get("message"),
    }


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--keys", nargs="+", default=["uploads/bus.jpg"])
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--csv", default="load_test_results.csv")
    a = ap.parse_args()
    api = a.api.rstrip("/")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(lambda i: detect(api, a.key, a.keys[i % len(a.keys)], i), range(a.requests)))
    wall = time.time() - t0

    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == 200]
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    cold = [r for r in ok if r["cold_start"]]
    warm = [r for r in ok if r["cold_start"] is False]
    print(f"\n{a.requests} requests, concurrency {a.concurrency}, wall time {wall:.1f}s "
          f"-> {a.requests / wall:.2f} req/s")
    print(f"status codes: {by_status}")
    if ok:
        lat = [r["client_ms"] for r in ok]
        print(f"client latency ms: p50={pct(lat,50)} p95={pct(lat,95)} max={max(lat)}")
    if warm:
        print(f"warm  requests: {len(warm)}  mean client ms={statistics.mean(r['client_ms'] for r in warm):.0f}  "
              f"mean inference ms={statistics.mean(r['inference_ms'] for r in warm):.0f}")
    if cold:
        print(f"COLD  requests: {len(cold)}  mean client ms={statistics.mean(r['client_ms'] for r in cold):.0f}  "
              f"init seconds={sorted({r['init_seconds'] for r in cold})}")
    errs = [r for r in rows if r["status"] != 200]
    for r in errs[:5]:
        print(f"  error #{r['i']}: {r['status']} {r['error']}")
    print(f"per-request rows written to {a.csv}")


if __name__ == "__main__":
    main()
