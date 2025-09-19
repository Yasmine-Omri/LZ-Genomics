# save as h5_to_fasta_like.py and run with Python 3.8+
import h5py
import numpy as np
from typing import List, Tuple, Optional
import argparse
import pandas as pd
import os


BASES = np.array(list("ACGT"))  # indices 0->A,1->C,2->G,3->T

def onehot_array_to_seqs(arr: np.ndarray) -> List[str]:
    """
    Convert one-hot array -> list of nucleotide strings.
    Accepts arr shape (n, L, 4) or (L, 4). Treats all-zero positions as 'N'.
    """
    a = np.asarray(arr)
    # handle single-sequence input
    if a.ndim == 2 and a.shape[1] == 4:
        a = a[np.newaxis, ...]
    if not (a.ndim == 3 and a.shape[2] == 4):
        raise ValueError(f"Expected array shape (n, L, 4) or (L,4). Got {a.shape}")

    # index of max channel per position
    idx = a.argmax(axis=2)          # shape (n, L), values 0..3
    chars = BASES[idx]              # shape (n, L), dtype='<U1'
    # positions that are all-zero -> mark as 'N'
    zero_mask = (a.sum(axis=2) == 0)
    if zero_mask.any():
        chars[zero_mask] = 'N'
    # join rows to strings
    seqs = [''.join(row.tolist()) for row in chars]
    return seqs


def get_category_seqs(
    h5_path: str,
    cell_type: str = 'GM12878',
    split: str = 'train',
    category: str = 'peaks'
) -> list[str]:
    """
    Return (seq_list, idxs_array_or_None) for the requested cell/split/category.
    Example category values: 'peaks', 'nonpeaks', 'idr'
    """
    with h5py.File(h5_path, 'r') as f:
        try:
            grp = f[cell_type][split][category]
        except KeyError as e:
            raise KeyError(f"Could not find path {cell_type}/{split}/{category} in {h5_path}") from e

        seqs_ds = grp['seqs']      # h5py dataset -> shape (N, L, 4)
        arr = seqs_ds[...]
        seqs = onehot_array_to_seqs(arr)
        return seqs


# ----------------- example usage -----------------
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description="Extract sequences from H5 file for task4.")
    parser.add_argument('--h5_path', type=str, default="data.h5", help="Path to the input H5 file.")
    parser.add_argument('--cell_type', type=str, default='all',
                        choices=['GM12878', 'H1ESC', 'HEPG2', 'IMR90', 'K562', 'all'],
                        help="Cell type to extract (default: all).")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save output CSV files.")
    args = parser.parse_args()

    H5 = args.h5_path
    # peaks

    cell_types = ['GM12878', 'H1ESC', 'HEPG2', 'IMR90', 'K562'] if args.cell_type == 'all' else [args.cell_type]
    for args.cell_type in cell_types:
        for split in ['train', 'val', 'test']:
            dir = f"{args.output_dir}/{args.cell_type}"
            os.makedirs(dir, exist_ok=True)
            pd.DataFrame([
                {"sequence": seq, "label": 1} for seq in get_category_seqs(H5, cell_type=args.cell_type, split=split, category='idr_peaks')
            ] + [
                {"sequence": seq, "label": 0} for seq in get_category_seqs(H5, cell_type=args.cell_type, split=split, category='nonpeaks')
                
            ]).to_csv(f"{dir}/{split if split != 'val' else 'dev'}.csv", index=False)
