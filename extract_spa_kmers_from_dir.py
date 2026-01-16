#!/usr/bin/env python3
"""
Extract sorted k-mers from a directory of SPA .bin files using motif-specific k from a MEME file.

File naming convention expected:
  "<motif_name>_0.bin"  -> false case
  "<motif_name>_1.bin"  -> true case

Output files (CSV) per SPA:
  "<motif_name>_[true|false]_k-mers.csv"

Each CSV contains two columns: kmer,count (sorted by descending count).

This script uses functions demonstrated in the attached template:
 - spa_from_file
 - get_top_counts_at_depths
 - CharacterMap
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
from typing import Dict, Tuple, Optional

from lz78 import spa_from_file, get_top_counts_at_depths, CharacterMap


def parse_meme_widths(meme_path: str) -> Dict[str, int]:
    """
    Parse a MEME-format file to build a mapping from motif name -> width (k).

    We support standard MEME blocks like:
      MOTIF <name> [altname...]
      letter-probability matrix: alength= 4 w= 10 ...

    We store multiple normalized keys for robust matching:
      - original motif name
      - motif with '.' replaced by '_' and vice versa
      - lowercased variants of the above
      - motif with internal whitespace collapsed to single spaces
    """
    name_to_w: Dict[str, int] = {}

    current_motif: Optional[str] = None
    with open(meme_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Capture motif name
            if line.startswith("MOTIF "):
                # Everything after 'MOTIF ' up to end-of-line is the identifier line.
                # The first token after MOTIF is typically the primary name.
                tokens = line.split()
                # tokens[0] == "MOTIF"; there may be 1+ following tokens; we take the first as the motif name
                current_motif = tokens[1] if len(tokens) > 1 else None

            # Capture width 'w=' on the letter-probability matrix line
            if "letter-probability matrix" in line and "w=" in line and current_motif:
                # Extract w=<int>
                m = re.search(r"\bw\s*=\s*(\d+)", line)
                if m:
                    w = int(m.group(1))
                    # Store multiple keys for the same motif for easier matching
                    variants = set()
                    variants.add(current_motif)
                    variants.add(current_motif.replace(".", "_"))
                    variants.add(current_motif.replace("_", "."))
                    variants.add(re.sub(r"\s+", " ", current_motif.strip()))
                    # lowercased
                    variants |= {v.lower() for v in list(variants)}
                    for key in variants:
                        name_to_w[key] = w
                current_motif = None  # reset until the next MOTIF
    return name_to_w


def derive_motif_and_label(bin_filename: str) -> Tuple[str, Optional[int]]:
    """
    Given a filename like '<motif>_0.bin' or '<motif>_1.bin', return (motif, label_int).
    label_int is 0 or 1 if parseable, else None.
    """
    base = os.path.basename(bin_filename)
    if not base.endswith(".bin"):
        return base, None
    stem = base[:-4]  # drop .bin
    # Split off the final underscore segment for the label
    if "_" not in stem:
        return stem, None
    motif_part, label_part = stem.rsplit("_", 1)
    try:
        label = int(label_part)
        if label not in (0, 1):
            label = None
    except ValueError:
        label = None
    return motif_part, label


def normalize_for_lookup(name: str) -> Tuple[str, ...]:
    """
    Produce a tuple of possible keys to use for lookup in the MEME mapping.
    """
    variants = {
        name,
        name.replace(".", "_"),
        name.replace("_", "."),
        re.sub(r"\s+", " ", name.strip()),
    }
    variants |= {v.lower() for v in list(variants)}
    return tuple(variants)


def extract_kmers_for_spa(
    spa_path: str,
    k: int,
    topk: Optional[int],
    alphabet: str = "ACGT",
) -> Dict[str, int]:
    """
    Load SPA and extract k-mer counts at exactly depth k.
    Returns a dict {kmer: count}.
    """
    spa = spa_from_file(spa_path)
    # Use min_depth=max_depth=k to get only the target depth
    depth_counts = get_top_counts_at_depths(
        spa,
        min_depth=k,
        max_depth=k,
        charmap=CharacterMap(alphabet),
        topk=topk,
    )
    # depth_counts is a dict: {k: {kmer: count}}
    return depth_counts.get(k, {})


def write_sorted_csv(
    out_path: str,
    counts: Dict[str, int],
) -> None:
    """
    Write CSV with columns: kmer,count sorted by descending count then kmer.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kmer", "count"])
        for kmer, c in sorted_items:
            w.writerow([kmer, c])


def main():
    p = argparse.ArgumentParser(description="Extract sorted k-mers for each SPA .bin using motif-specific k from a MEME file.")
    p.add_argument("--spa_dir", required=True, help="Directory containing SPA .bin files named '<motif>_[0|1].bin'")
    p.add_argument("--meme_path", required=True, help="Path to the .meme file specifying per-motif k (width)")
    p.add_argument("--output_dir", default="kmers_csv", help="Directory to write per-file CSV outputs")
    p.add_argument("--alphabet", default="ACGT", help="Alphabet for CharacterMap (default: ACGT)")
    p.add_argument("--topk", type=int, default=None, help="Optionally limit to top-K k-mers by count at depth k")
    args = p.parse_args()

    name_to_w = parse_meme_widths(args.meme_path)

    # Walk the directory for .bin files
    for fname in os.listdir(args.spa_dir):
        if not fname.endswith(".bin"):
            continue
        spa_path = os.path.join(args.spa_dir, fname)
        motif, label = derive_motif_and_label(fname)

        # Determine label string for output naming
        label_str = "true" if label == 1 else "false" if label == 0 else "unknown"

        # Find k for this motif
        k = None
        for key in normalize_for_lookup(motif):
                k = name_to_w[key.replace(".csv","").replace("_csv","")]
                break

        if k is None:
            print(f"[WARN] Could not find k (width) for motif '{motif}' in MEME file. Skipping: {fname}")
            continue

        print(f"[INFO] Processing: motif='{motif}' label={label_str} k={k} file='{fname}'")

        # Extract k-mers
        try:
            counts = extract_kmers_for_spa(spa_path, k=k, topk=args.topk, alphabet=args.alphabet)
        except Exception as e:
            print(f"[ERROR] Failed to extract k-mers for '{fname}': {e}")
            continue

        # Output CSV path
        out_name = f"{motif}_{label_str}_k-mers.csv"
        out_path = os.path.join(args.output_dir, out_name)
        try:
            write_sorted_csv(out_path, counts)
            print(f"[OK] Wrote: {out_path} ({len(counts)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV for '{fname}': {e}")


if __name__ == "__main__":
    main()
