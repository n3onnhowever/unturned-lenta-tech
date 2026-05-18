"""Compare two parsed-field JSON files (match by crop filename)."""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


def similarity(a: str, b: str) -> float:
    a = str(a or "").lower().strip()
    b = str(b or "").lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_parsed(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for item in data:
        name = Path(item["image_path"]).name
        out[name] = item.get("parsed", {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/compare_vs_baseline.json"))
    args = ap.parse_args()

    base = load_parsed(args.baseline)
    cand = load_parsed(args.candidate)
    keys = sorted(set(base) & set(cand))

    name_sims = []
    pd_match = pc_match = 0
    for k in keys:
        b, c = base[k], cand[k]
        name_sims.append(similarity(b.get("product_name"), c.get("product_name")))
        if (b.get("price_default") or "") == (c.get("price_default") or ""):
            pd_match += 1
        if (b.get("price_card") or "") == (c.get("price_card") or ""):
            pc_match += 1

    n = len(keys)
    results = {
        "images_compared": n,
        "baseline_only": len(set(base) - set(cand)),
        "candidate_only": len(set(cand) - set(base)),
        "product_name_avg_sim_vs_baseline": round(sum(name_sims) / n, 4) if n else 0.0,
        "price_default_same_as_baseline_pct": round(100.0 * pd_match / n, 1) if n else 0.0,
        "price_card_same_as_baseline_pct": round(100.0 * pc_match / n, 1) if n else 0.0,
        "candidate_better_name_count": sum(
            1 for k in keys if similarity(cand[k].get("product_name"), base[k].get("product_name")) > 0.99
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
