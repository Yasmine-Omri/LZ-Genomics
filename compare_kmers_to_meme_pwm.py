#!/usr/bin/env python3
"""
compare_kmers_to_meme_pwm.py

Unified tool to:
  1) Compare a single k-mers CSV against a target motif in a MEME file.
  2) Batch-process one or many k-mers CSVs (via --input_csv or --input_dir),
     inferring motifs from filenames when not explicitly provided,
     writing per-motif .txt result files, and a summary CSV.

This merges the functionality of:
 - compare_kmers_to_meme_pwm.py  (core comparison, PWM build, CLI pieces)
 - compare_kmers_batch.py        (batch mode, motif inference, summaries)

CSV format expected: columns: kmer,count

Examples
--------
Single CSV, explicit motif:
  python kmers_compare_unified.py \
    --meme HOCOMOCOv12.meme \
    --input_csv AHR.H12CORE.0.P.B_true_k-mers.csv \
    --motif AHR.H12CORE.0.P.B \
    --metric pearson --rc_merge \
    --out_dir motif_results_txt --summary_csv motif_results_summary.csv

Batch over a directory, infer motif per file:
  python kmers_compare_unified.py \
    --meme HOCOMOCOv12.meme \
    --input_dir kmers_csv \
    --metric pearson --rc_merge \
    --out_dir motif_results_txt --summary_csv motif_results_summary.csv

Optionally write each built PWM to MEME format (for inspection):
  --write_tree_pwms_dir tree_pwms

Notes
-----
- Motif inference from filename: robust to extra suffixes like "_true_k-mers", "_false_k-mers".
- If you supply --motif, inference is skipped and that pattern is used for all files.
- Auto Top-N plateau selection can be disabled with --no_autoN and a fixed --topN.
"""

import os
import re
import sys
import csv
import math
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np

# -------------------- DNA helpers --------------------
_RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(s: str) -> str:
    return s.translate(_RC_TRANS)[::-1]

def canonical_rc(s: str) -> str:
    rc = revcomp(s)
    return s if s <= rc else rc

# -------------------- MEME / GimmeMotifs helpers --------------------
def load_motifs_from_meme(meme_path: str):
    from gimmemotifs.motif import read_motifs
    return read_motifs(meme_path)

def _matches(text: str, pat: str) -> bool:
    # treat strings containing regex metachars as regex; else substring
    if re.search(r"[.^$*+?{}\[\]|()\\]", pat):
        return re.search(pat, text) is not None
    return pat in text

def get_motif_by_pattern(motifs, pattern: str):
    cand = [m for m in motifs if _matches(f"{m.id} {getattr(m,'name','')}", pattern)]
    if not cand:
        raise ValueError(f"No motif matching '{pattern}' found in MEME file.")
    # prefer exact id match if multiple
    exact = [m for m in cand if getattr(m, "id", None) == pattern]
    return exact[0] if exact else cand[0]

# -------------------- PWM from k-mers --------------------
def build_pwm_from_kmers(kmers_counts: List[Tuple[str,int]], pseudocount: float = 0.5):
    """
    kmers_counts: list of (kmer, count), fixed-length k.
    Returns PWM as list[dict] per position: {'A':pA,'C':pC,'G':pG,'T':pT}.
    """
    if not kmers_counts:
        return []
    L = len(kmers_counts[0][0])
    bases = "ACGT"
    cols = [Counter() for _ in range(L)]
    for s, c in kmers_counts:
        if len(s) != L:  # ignore bad rows
            continue
        for i, b in enumerate(s):
            if b in bases:
                cols[i][b] += int(c)
    pwm = []
    for col in cols:
        total = sum(col[b] for b in bases) + 4 * pseudocount
        pwm.append({b: (col[b] + pseudocount) / total for b in bases})
    return pwm

def pwm_list_to_ppm_array(pwm_list):
    """Convert list[dict] -> 4 x L numpy array (A,C,G,T rows), columns sum to 1."""
    if not pwm_list:
        return np.zeros((4, 0), dtype=float)
    arr = np.array([[col['A'], col['C'], col['G'], col['T']] for col in pwm_list]).T
    colsum = arr.sum(axis=0, keepdims=True)
    colsum[colsum == 0] = 1.0
    return arr / colsum

# -------------------- Optional: choose Top-N via plateau on PWM hits --------------------
def pwm_logodds_score(kmer: str, pwm_cols: List[Dict[str,float]],
                      bg: Optional[Dict[str,float]] = None, logbase=2) -> float:
    if bg is None:
        bg = {'A':0.25,'C':0.25,'G':0.25,'T':0.25}
    ln = (lambda x: math.log(x, 2)) if logbase == 2 else math.log
    s = 0.0
    for i, b in enumerate(kmer):
        p = max(1e-12, pwm_cols[i][b])
        q = max(1e-12, bg[b])
        s += ln(p / q)
    return s

def choose_topN_plateau(kmer_counts: List[Tuple[str,int]],
                        motif_pwm_cols: List[Dict[str,float]],
                        pwm_bg: Optional[Dict[str,float]] = None,
                        Ns: Optional[List[int]] = None,
                        hit_thresh: float = 0.0,
                        plateau_tol: float = 0.01,
                        plateau_patience: int = 1) -> int:
    """Pick N where weighted hit-rate plateaus."""
    if not kmer_counts:
        return 0
    max_items = len(kmer_counts)
    if Ns is None:
        grid = [25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
        Ns = [n for n in grid if n <= max_items] or [max_items]
        if max_items not in Ns:
            Ns.append(max_items)
    prev = None
    patience = 0
    chosen = Ns[-1]
    for N in Ns:
        top = kmer_counts[:N]
        tot = sum(c for _, c in top) or 1
        hits = 0
        for s, c in top:
            if pwm_logodds_score(s, motif_pwm_cols, bg=pwm_bg) >= hit_thresh:
                hits += c
        hr = hits / tot
        if prev is not None and (hr - prev) <= plateau_tol:
            patience += 1
            if patience >= plateau_patience and chosen == Ns[-1]:
                chosen = N
        else:
            patience = 0
        prev = hr
    return chosen

# -------------------- Core comparison function --------------------
def compare_kmers_to_meme_pwm(
    meme_path: str,
    motif_pattern: str,
    kmer_counts: List[Tuple[str,int]],
    metric: str = "pearson",
    rc_merge_kmers: bool = True,
    pseudocount: float = 0.5,
    auto_choose_topN: bool = True,
    hit_thresh: float = 0.0,
    plateau_tol: float = 0.01,
    plateau_patience: int = 1,
    pwm_bg: Optional[Dict[str,float]] = None,
    topN: Optional[int] = None,
    write_tree_pwm_meme: Optional[str] = None,
):
    """
    Build a PWM from your k-mers (matching k = motif width) and compare to target PWM from MEME.

    Returns dict with: score, metric, chosen_k, chosen_N, target_id, etc.
    """
    # 1) Load motif
    motifs = load_motifs_from_meme(meme_path)
    target = get_motif_by_pattern(motifs, motif_pattern)
    k = int(target.length)

    # 2) Filter to length k and (optionally) RC-merge
    merged = defaultdict(int)
    for s, c in kmer_counts:
        s = s.upper()
        if len(s) != k or any(ch not in "ACGT" for ch in s):
            continue
        key = canonical_rc(s) if rc_merge_kmers else s
        merged[key] += int(c)
    kmer_k = sorted(merged.items(), key=lambda t: t[1], reverse=True)
    if not kmer_k:
        return {"ok": False, "reason": f"No k-mers of length {k} in input."}

    # 3) Optionally choose Top-N by plateau vs target PWM
    #    Build a dict form of target for the hit-rate calc
    target_ppm = target.ppm  # 4 x k
    target_cols = [{'A': float(target_ppm[0, j]),
                    'C': float(target_ppm[1, j]),
                    'G': float(target_ppm[2, j]),
                    'T': float(target_ppm[3, j])} for j in range(k)]

    if auto_choose_topN:
        chosen_N = choose_topN_plateau(
            kmer_k, target_cols, pwm_bg=pwm_bg,
            hit_thresh=hit_thresh, plateau_tol=plateau_tol, plateau_patience=plateau_patience
        ) or min(200, len(kmer_k))
    else:
        chosen_N = topN or min(200, len(kmer_k))

    top = kmer_k[:chosen_N]

    # 4) Build tree PWM from selected k-mers
    tree_pwm_list = build_pwm_from_kmers(top, pseudocount=pseudocount)

    # 5) Compare with GimmeMotifs
    from gimmemotifs.motif import Motif, write_motifs
    from gimmemotifs.comparison import MotifComparer

    tree_ppm = pwm_list_to_ppm_array(tree_pwm_list)  # 4 x k
    tree_motif = Motif(ppm=tree_ppm)
    tree_motif.id = "TREE_PWM"

    mc = MotifComparer()
    score = mc.compare_motifs(tree_motif, target, metric=metric, rc=True)

    # Optional: write your PWM to a MEME file for inspection
    if write_tree_pwm_meme:
        write_motifs([tree_motif], write_tree_pwm_meme, fmt="meme")

    return {
        "ok": True,
        "metric": metric,
        "score": float(score),
        "chosen_k": k,
        "chosen_N": int(chosen_N),
        "num_kmers_available": len(kmer_k),
        "target_id": target.id,
        "wrote_tree_pwm_to": write_tree_pwm_meme or "",
    }

# -------------------- CSV I/O --------------------
def read_kmers_csv(path: str) -> List[Tuple[str,int]]:
    out = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if "kmer" not in r.fieldnames or "count" not in r.fieldnames:
            raise SystemExit(f"CSV must have columns: kmer,count (got {r.fieldnames})")
        for row in r:
            s = row["kmer"].strip().upper()
            c = int(row["count"])
            out.append((s, c))
    return out

# -------------------- Motif inference from filename --------------------
def _variants(s: str):
    items = {s, s.replace(".", "_"), s.replace("_", "."), re.sub(r"\s+", " ", s.strip())}
    items |= {x.lower() for x in list(items)}
    return list(items)

def infer_motif_id_from_filename(filename: str, motifs) -> Optional[str]:
    """
    Heuristically choose the best motif id for this filename.
    Strategy:
      1) Exact id substring match (case-sensitive), prefer longer ids.
      2) Case-insensitive id substring match.
      3) Variant-based (._ swaps), case-insensitive.
      4) Try the motif 'name' field similarly.
    """
    stem = Path(filename).stem
    # remove common suffixes added by upstream pipeline
    stem_norm = re.sub(r"(_true|_false)?_k-?mers$", "", stem, flags=re.IGNORECASE)
    cands = []
    for m in motifs:
        mid = getattr(m, "id", "")
        mname = getattr(m, "name", "") or ""

        def score_if(ok: bool, base: int):
            if ok:
                cands.append((base + len(mid), mid))

        score_if(mid and mid in stem_norm, 1000)
        score_if(mid and mid.lower() in stem_norm.lower(), 800)
        for v in _variants(mid):
            score_if(v and v in stem_norm.lower(), 600)

        if mname:
            score_if(mname and mname in stem_norm, 400)
            score_if(mname and mname.lower() in stem_norm.lower(), 300)
            for v in _variants(mname):
                score_if(v and v in stem_norm.lower(), 200)

    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]

# -------------------- Batch utilities --------------------
def iter_csv_files(input_csv: Optional[str], input_dir: Optional[str]):
    files = []
    if input_csv:
        files.append(input_csv)
    if input_dir:
        for f in os.listdir(input_dir):
            if f.lower().endswith(".csv"):
                files.append(os.path.join(input_dir, f))
    # dedupe preserve order
    seen = set()
    out = []
    for p in files:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def write_txt_result(out_dir: str, motif_id: str, src_csv: str, result: Dict):
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", motif_id) or "UNKNOWN"
    out_path = os.path.join(out_dir, f"{safe}.txt")
    payload = {
        "source_csv": os.path.basename(src_csv),
        "motif_id": motif_id,
        **result
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path

def append_summary_row(summary_csv: str, row: Dict, header: List[str]):
    # Ensure parent directory exists if a path like logs/summary.csv was given
    parent = os.path.dirname(summary_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Unified single/batch comparator of k-mer CSVs to MEME motifs.")
    ap.add_argument("--meme", required=True, help="Path to MEME motif database")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_csv", help="Single k-mer CSV file (kmer,count)")
    group.add_argument("--input_dir", help="Directory of k-mer CSV files")

    ap.add_argument("--motif", default=None, help="Explicit motif id/pattern (skip inference)")
    ap.add_argument("--metric", default="pearson", help="MotifComparer metric (default: pearson)")
    ap.add_argument("--rc_merge", action="store_true", help="Merge reverse complements before PWM build")
    ap.add_argument("--no_autoN", action="store_true", help="Disable auto Top-N plateau; use --topN")
    ap.add_argument("--topN", type=int, default=None, help="Top N k-mers to use when --no_autoN")
    ap.add_argument("--pseudocount", type=float, default=0.5, help="PWM pseudocount")

    ap.add_argument("--out_dir", default="motif_results_txt", help="Directory to write per-motif .txt results")
    ap.add_argument("--summary_csv", default="motif_results_summary.csv", help="Path for summary CSV")

    ap.add_argument("--write_tree_pwms_dir", default=None, help="Optional dir to write each built PWM in MEME format")

    args = ap.parse_args()

    motifs = load_motifs_from_meme(args.meme)

    files = iter_csv_files(args.input_csv, args.input_dir)
    if not files:
        print("No CSV files found.")
        sys.exit(2)

    header = [
        "source_csv", "motif_id", "metric", "score",
        "chosen_k", "chosen_N", "num_kmers_available",
        "rc_merge", "auto_choose_topN", "tree_pwm_meme_path"
    ]

    wrote = 0
    for csv_path in files:
        try:
            kmer_counts = read_kmers_csv(csv_path)

            if args.motif:
                motif_pat = args.motif
            else:
                motif_id = infer_motif_id_from_filename(os.path.basename(csv_path), motifs)
                if not motif_id:
                    print(f"[WARN] Could not infer motif for {csv_path}; skipping.")
                    continue
                motif_pat = motif_id

            write_tree_pwm_meme = None
            if args.write_tree_pwms_dir:
                os.makedirs(args.write_tree_pwms_dir, exist_ok=True)
                safe_csv = re.sub(r"[^A-Za-z0-9._+-]+", "_", os.path.basename(csv_path))
                write_tree_pwm_meme = os.path.join(args.write_tree_pwms_dir, f"{safe_csv}.meme")

            res = compare_kmers_to_meme_pwm(
                meme_path=args.meme,
                motif_pattern=motif_pat,
                kmer_counts=kmer_counts,
                metric=args.metric,
                rc_merge_kmers=args.rc_merge,
                pseudocount=args.pseudocount,
                auto_choose_topN=(not args.no_autoN),
                topN=args.topN,
                write_tree_pwm_meme=write_tree_pwm_meme
            )

            if not res.get("ok"):
                print(f"[WARN] {csv_path}: {res.get('reason','Unknown error')}")
                continue

            out_txt = write_txt_result(args.out_dir, res.get("target_id", motif_pat), csv_path, res)

            row = {
                "source_csv": os.path.basename(csv_path),
                "motif_id": res.get("target_id", motif_pat),
                "metric": res.get("metric", args.metric),
                "score": res.get("score"),
                "chosen_k": res.get("chosen_k"),
                "chosen_N": res.get("chosen_N"),
                "num_kmers_available": res.get("num_kmers_available"),
                "rc_merge": bool(args.rc_merge),
                "auto_choose_topN": (not args.no_autoN),
                "tree_pwm_meme_path": res.get("wrote_tree_pwm_to", ""),
            }
            append_summary_row(args.summary_csv, row, header)
            wrote += 1
            print(f"[OK] {csv_path} -> {out_txt} (score={row['score']:.4f})")

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    if wrote == 0:
        print("No results written. Check warnings above.")
        sys.exit(1)
    else:
        print(f"Done. Wrote {wrote} result(s). Summary: {args.summary_csv}")

if __name__ == "__main__":
    main()

