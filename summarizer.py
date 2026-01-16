#!/usr/bin/env python3
"""
Summarize k-mer baseline reports into one spreadsheet.

Parses report files produced by KmerBaseline_external.py (the version that prints):
- grid rows: "k,alpha,feature_mode,test_acc,test_mcc,train_wall_s,test_wall_s,tool,canonical"
- "BEST ON TEST: {...}"
- "VERIFY ACC: X.XXXX"
- "VERIFY MCC: X.XXXX"
- TIME PROFILING+ fields

Outputs:
- Excel workbook (default): <output>.xlsx with sheets: grid, summary
- CSV fallbacks: <output>_grid.csv and <output>_summary.csv if openpyxl not found.

Usage:
  python summarize_kmer_reports.py --input_dir ./kmer_train_reports --output summary

Tip:
  The dataset name is inferred from the filename; customize `infer_dataset()` if needed.
"""
import argparse
import os
import re
import ast
import glob
from typing import Dict, Any, List
import pandas as pd

GRID_HEADER = (
    "k,alpha,feature_mode,test_acc,test_mcc,train_wall_s,test_wall_s,tool,canonical"
)

re_best = re.compile(r"^BEST ON TEST:\s*(\{.*\})\s*$")
re_verify_acc = re.compile(r"^VERIFY ACC:\s*([0-9.]+)")
re_verify_mcc = re.compile(r"^VERIFY MCC:\s*([0-9.]+)")

# TIME PROFILING+ lines
re_read_train = re.compile(r"^Read train \+ val data time:\s*([0-9.eE+\- ]+)")
re_nb_train_symbols = re.compile(r"^Number of training symbols:\s*([0-9]+)")
re_train_seq_len = re.compile(r"^Length of one training sequence:\s*([0-9]+)")
re_total_train = re.compile(r"^Total training time:\s*([0-9.]+)\s*seconds")

re_nb_test = re.compile(r"^Number of test sequences:\s*([0-9]+)")
re_test_seq_len = re.compile(r"^Length of test sequence:\s*([0-9]+)")
re_read_test = re.compile(r"^Read test data time:\s*([0-9.eE+\- ]+)")
re_total_inf = re.compile(r"^Total inference time:\s*([0-9.]+)\s*seconds")
re_inf_per_symbol = re.compile(r"^Inference time/symbol:\s*([0-9.eE+\-]+)\s*seconds")

def infer_dataset(fname: str) -> str:
    """Infer dataset label from filename (e.g., 'mouse0.txt' -> 'mouse0')."""
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    return name

def parse_grid_line(line: str) -> Dict[str, Any]:
    # CSV with 9 columns matching GRID_HEADER
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 9:
        raise ValueError("grid line wrong column count")
    k, alpha, fmode, acc, mcc, train_s, test_s, tool, canonical = parts
    return {
        "k": int(k),
        "alpha": float(alpha),
        "feature_mode": fmode,
        "test_acc": float(acc),
        "test_mcc": float(mcc),
        "train_wall_s": float(train_s),
        "test_wall_s": float(test_s),
        "tool": tool,
        "canonical": True if canonical == "True" else False if canonical == "False" else canonical,
    }

def safe_literal_eval(s: str) -> Dict[str, Any]:
    # Convert single quotes/True/False/None safely
    try:
        return ast.literal_eval(s)
    except Exception:
        # last resort: replace booleans with lowercase to help literal_eval
        s2 = s.replace("True", "True").replace("False", "False").replace("None", "None")
        return ast.literal_eval(s2)

def parse_report(path: str) -> Dict[str, Any]:
    grid_rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}
    verify_acc = None
    verify_mcc = None
    timing: Dict[str, Any] = {}

    in_grid = False

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()

            if not in_grid and line.startswith(GRID_HEADER):
                in_grid = True
                continue

            if in_grid:
                # grid ends when we hit an empty line or non-CSV ‘BEST’ line
                if not line or line.startswith("BEST ON TEST:"):
                    in_grid = False
                else:
                    # try parse row; skip if malformed
                    try:
                        row = parse_grid_line(line)
                        grid_rows.append(row)
                        continue
                    except Exception:
                        # if it's not a valid grid row, treat as non-grid from now on
                        in_grid = False
                        # fall-through to parse other stuff

            m = re_best.match(line)
            if m:
                try:
                    best = safe_literal_eval(m.group(1))
                except Exception:
                    best = {}
                continue

            m = re_verify_acc.match(line)
            if m:
                try:
                    verify_acc = float(m.group(1))
                except Exception:
                    verify_acc = None
                continue

            m = re_verify_mcc.match(line)
            if m:
                try:
                    verify_mcc = float(m.group(1))
                except Exception:
                    verify_mcc = None
                continue

            # timing fields
            m = re_read_train.match(line)
            if m: timing["read_train_val_s"] = float(m.group(1))
            m = re_nb_train_symbols.match(line)
            if m: timing["nb_train_symbols"] = int(m.group(1))
            m = re_train_seq_len.match(line)
            if m: timing["train_seq_len"] = int(m.group(1))
            m = re_total_train.match(line)
            if m: timing["total_training_s"] = float(m.group(1))
            m = re_nb_test.match(line)
            if m: timing["nb_test_seqs"] = int(m.group(1))
            m = re_test_seq_len.match(line)
            if m: timing["test_seq_len"] = int(m.group(1))
            m = re_read_test.match(line)
            if m: timing["read_test_s"] = float(m.group(1))
            m = re_total_inf.match(line)
            if m: timing["total_inference_s"] = float(m.group(1))
            m = re_inf_per_symbol.match(line)
            if m: timing["inference_per_symbol_s"] = float(m.group(1))

    return {
        "grid": grid_rows,
        "best": best,
        "verify_acc": verify_acc,
        "verify_mcc": verify_mcc,
        "timing": timing,
    }

def main():
    ap = argparse.ArgumentParser(description="Summarize k-mer baseline report files into one spreadsheet.")
    ap.add_argument("--input_dir", required=True, help="Folder containing .txt reports (e.g., ./kmer_train_reports)")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern for report files (default: *.txt)")
    ap.add_argument("--output", required=True, help="Output file stem (e.g., summary -> summary.xlsx / CSVs)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    if not paths:
        print(f"No files found at {args.input_dir}/{args.glob}")
        return

    grid_records = []
    summary_records = []

    for p in paths:
        parsed = parse_report(p)
        dataset = infer_dataset(p)

        # Add dataset/file to grid rows
        for row in parsed["grid"]:
            rec = {"dataset": dataset, "report_file": os.path.basename(p)}
            rec.update(row)
            grid_records.append(rec)

        # Build summary row
        best = parsed["best"] or {}
        timing = parsed["timing"] or {}
        summary = {
            "dataset": dataset,
            "report_file": os.path.basename(p),
            # best hyperparams (if present)
            "best_k": best.get("k"),
            "best_alpha": best.get("alpha"),
            "best_feature_mode": best.get("fmode") or best.get("feature_mode"),
            "best_tool": best.get("tool"),
            "best_canonical": best.get("canonical"),
            "best_test_acc": best.get("acc"),
            "best_test_mcc": best.get("mcc"),
            # verify
            "verify_acc": parsed["verify_acc"],
            "verify_mcc": parsed["verify_mcc"],
            # timing
            "read_train_val_s": timing.get("read_train_val_s"),
            "nb_train_symbols": timing.get("nb_train_symbols"),
            "train_seq_len": timing.get("train_seq_len"),
            "total_training_s": timing.get("total_training_s"),
            "nb_test_seqs": timing.get("nb_test_seqs"),
            "test_seq_len": timing.get("test_seq_len"),
            "read_test_s": timing.get("read_test_s"),
            "total_inference_s": timing.get("total_inference_s"),
            "inference_per_symbol_s": timing.get("inference_per_symbol_s"),
        }
        summary_records.append(summary)

    df_grid = pd.DataFrame(grid_records)
    df_summary = pd.DataFrame(summary_records)

    # Write Excel if possible; otherwise CSVs
    xlsx_path = f"{args.output}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_grid.to_excel(writer, index=False, sheet_name="grid")
            df_summary.to_excel(writer, index=False, sheet_name="summary")
        print(f"Wrote Excel: {xlsx_path}")
    except Exception as e:
        print(f"Could not write Excel ({e}); writing CSVs instead.")
        df_grid.to_csv(f"{args.output}_grid.csv", index=False)
        df_summary.to_csv(f"{args.output}_summary.csv", index=False)
        print(f"Wrote CSVs: {args.output}_grid.csv, {args.output}_summary.csv")

if __name__ == "__main__":
    main()
