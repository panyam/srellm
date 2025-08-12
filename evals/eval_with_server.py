# evals/eval_with_server.py
import pandas as pd, requests

CSV = "data/incidents_gold.csv"
URL = "http://127.0.0.1:8000/generate"

def generate(summary: str) -> str:
    r = requests.post(URL, json={"prompt": summary, "top_k": 2, "temperature": 0.0}, timeout=60)
    r.raise_for_status()
    return r.json()["output"]

def contains_all(pred: str, keys):
    t = pred.lower()
    return all(k.strip().lower() in t for k in keys if k.strip())

def score_row(pred: str, expected: str) -> float:
    keys = [k.strip() for k in expected.split(";")]
    return 1.0 if contains_all(pred, keys) else 0.0

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    rows = []
    for _, r in df.iterrows():
        out = generate(r["summary"])
        s = score_row(out, r["expected_key_actions"])
        rows.append({"incident_id": r["incident_id"], "pred": out, "score": s})
        print(f"id={r['incident_id']} score={s} pred={out}")

    out_df = pd.DataFrame(rows)
    print("\nAvg score:", out_df["score"].mean())

