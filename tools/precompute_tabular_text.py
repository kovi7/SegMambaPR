"""Precompute sentence embeddings of the clinical metadata (tabular.mode=text).

A fixed template turns age, sex and scanner type of each case into one sentence, and a frozen
sentence-transformer embeds it. Training only reads the resulting JSON cache. Cases already in
--out are kept, so a rerun embeds only new cases. Needs `pip install sentence-transformers`.

    python tools/precompute_tabular_text.py \
        --manifest data/panorama_manifest_seed42.json \
        --out data/tabular_text_embeddings.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

MISSING_TEXT = "Clinical information is not available for this patient."

SEX_WORDS = {"M": "male", "MALE": "male", "F": "female", "FEMALE": "female"}


def row_to_text(row: Dict[str, Any]) -> str:
    age = row.get("age")
    sex = row.get("sex") or row.get("gender")
    scanner = row.get("scanner_type")

    age_part = f"{int(age)}-year-old" if age not in (None, "") else "age unknown"
    sex_part = SEX_WORDS.get(str(sex).strip().upper(), "sex unknown") if sex else "sex unknown"
    scanner_part = f"CT scan acquired on a {scanner} scanner." if scanner else "Scanner type unknown."

    return f"{age_part} {sex_part} patient. {scanner_part}"


def load_manifest_rows(manifest_path: str) -> List[Dict[str, Any]]:
    rows = json.load(open(manifest_path, encoding="utf-8"))
    seen = {}
    for r in rows:
        cid = r.get("case_id")
        if cid is not None:
            seen[cid] = r  # last row per case_id wins
    return list(seen.values())


def build_texts(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return {r["case_id"]: row_to_text(r) for r in rows}


def embed_texts(
    texts: Dict[str, str], embed_model: str, existing_embeddings: Dict[str, List[float]]
) -> tuple[Dict[str, List[float]], List[float], int]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    todo_ids = [cid for cid in texts if cid not in existing_embeddings]
    print(f"[precompute_tabular_text] {len(todo_ids)}/{len(texts)} cases need embedding "
          f"({len(texts) - len(todo_ids)} already cached).")

    embeddings = dict(existing_embeddings)
    if todo_ids:
        sentences = [texts[cid] for cid in todo_ids]
        vectors = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
        for cid, vec in zip(todo_ids, vectors):
            embeddings[cid] = vec.tolist()

    missing_vec = model.encode([MISSING_TEXT], normalize_embeddings=True)[0].tolist()
    dim = len(missing_vec)
    return embeddings, missing_vec, dim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/panorama_manifest_seed42.json", help="Manifest JSON.")
    ap.add_argument("--out", default="data/tabular_text_embeddings.json")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    rows = load_manifest_rows(args.manifest)
    texts = build_texts(rows)

    existing: Dict[str, Any] = {}
    out_path = Path(args.out)
    if out_path.exists():
        existing = json.load(open(out_path, encoding="utf-8"))
    existing_embeddings = dict(existing.get("embeddings", {}))

    embeddings, missing_vec, dim = embed_texts(texts, args.embed_model, existing_embeddings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "embeddings": embeddings,
            "missing": missing_vec,
            "texts": texts,
            "embed_model": args.embed_model,
            "dim": dim,
        },
        open(out_path, "w", encoding="utf-8"),
        indent=2,
    )
    print(f"[precompute_tabular_text] wrote {len(embeddings)} embeddings (dim={dim}) to {out_path}")


if __name__ == "__main__":
    main()
