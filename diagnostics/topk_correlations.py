from collections import Counter
import os
from typing import List, Tuple
from gimmemotifs.motif import Motif
from gimmemotifs.comparison import MotifComparer
from lz78 import spa_from_file, get_top_counts_at_depths, CharacterMap
import numpy as np


ACGT = "ACGT"


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
        # print(s, to_canonical(s))
        # s = to_canonical(s)
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
    spa_per_class: list[str],
    k: int
):
    spas = [spa_from_file(spa_path) for spa_path in spa_per_class]
    res = {
        i: get_top_counts_at_depths(
            spa,
            min_depth=k,
            max_depth=k,
            charmap=CharacterMap("ACGT"),
            topk=None,
        ) for i, spa in enumerate(spas)
    }

    motifs = {
        i: pwm_list_to_ppm_array(build_pwm_from_kmers(
            [(kmer, count) for kmer, count in res[i][k].items()]
        )) for i in range(len(spas))
    }

    for i, pwm in motifs.items():
        print(f">Class{i}")
        for (j, row) in enumerate(pwm):
            print(ACGT[j] + "\t" + "\t".join(f"{v:.4f}" for v in row))
        print()
    # for (i, motif) in motifs.items():
    #     motif.id = f"Class{i}"

    # mc = MotifComparer()

    # # compare all pairs of motifs
    # scores = {}
    # for i in range(len(spas)):
    #     for j in range(i + 1, len(spas)):
    #         motif1 = motifs[i]
    #         motif2 = motifs[j]
    #         scores[(i, j)] = mc.compare_motifs(motif1, motif2)
    
    # print(scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get top counts at depths from multiple SPA files and compare motifs.")
    
    # Define command-line arguments
    parser.add_argument("--spas_dir", type=str, required=True, help="Directory containing the .bin spa files for each class")
    parser.add_argument("--task", type=str, required=True, help="Name of the task")
    parser.add_argument("--k", type=int, required=True, help="Length of k-mers to extract")
    
    # Parse arguments
    args = parser.parse_args()

    spa_paths = []
    while True:
        if os.path.exists(f"{args.spas_dir}/{args.task}_{len(spa_paths)}.bin"):
            spa_paths.append(f"{args.spas_dir}/{args.task}_{len(spa_paths)}.bin")
        else:
            break
    
    k = args.k

    main(
        spa_per_class=spa_paths,
        k=k
    )
    
    