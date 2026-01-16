from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd


def to_canonical(kmer: str) -> str:
    """Return lexicographically smaller of kmer and its reverse complement."""
    complement = str.maketrans("ACGT", "TGCA")
    revcomp = kmer.translate(complement)[::-1]
    return min(kmer, revcomp)


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
        s = to_canonical(s)
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


def main(
    csv_path: str,
    k: int,
    start_idxs: list[int]
):
    df = pd.read_csv(csv_path)

    dfs_per_class = {}
    for class_label in sorted(df["label"].unique()):
        dfs_per_class[class_label] = df[df["label"] == class_label]
    
    for start_idx in start_idxs:
        for class_label, class_df in dfs_per_class.items():
            snapshots = class_df["sequence"].apply(lambda s: s[start_idx:start_idx+k])
            counts = snapshots.value_counts().sort_values(ascending=False)

            pwm = pwm_list_to_ppm_array(
                build_pwm_from_kmers([(kmer, count) for kmer, count in counts.items()])
            )
            print(f"Class: {class_label}, Start idx: {start_idx}")
            print(pwm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--start_idxs", type=int, nargs="+", required=True)
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        k=args.k,
        start_idxs=args.start_idxs
    )