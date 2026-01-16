import re
import argparse, os, math, shutil, subprocess, tempfile, itertools, time, random, sys
from collections import Counter
from typing import Dict, Optional
import pandas as pd
from memory_profiler import profile
from sklearn.metrics import f1_score, matthews_corrcoef
from lz78 import BackgroundPriors, KmerMultinomial, Sequences

ALPHABET = "ACGT"
ALPHABET_SIZE = 4


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


def normalize_for_lookup(name: str) -> tuple[str, ...]:
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


def check_bin(path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name if os.path.exists(path_or_name) else None
    return shutil.which(path_or_name)

# ============================ external counters ============================

def write_class_fastas(df_train, outdir):
    class_to_path = {}
    for c, sub in df_train.groupby("label"):
        p = os.path.join(outdir, f"class_{int(c)}.fa")
        with open(p, "w") as fh:
            for i, seq in enumerate(sub["sequence"].values):
                fh.write(f">{c}_{i}\n{seq}\n")
        class_to_path[int(c)] = p
    return class_to_path

# Jellyfish (canonical via -C; contiguous k-mers only)
def count_with_jellyfish(class_fastas, k, threads=8, hashsize="200M", canonical=True,
                         jellyfish_bin="jellyfish", tdir=None):
    jf = check_bin(jellyfish_bin)
    if not jf:
        raise RuntimeError("jellyfish binary not found. Set --jellyfish_bin.")
    files = []
    for c, fa in class_fastas.items():
        db = os.path.join(tdir, f"class_{c}.jf")
        cmd = [jf, "count", "-m", str(k), "-s", str(hashsize), "-t", str(threads), "-o", db]
        if canonical: cmd.append("-C")
        cmd.append(fa)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Dump kmer\tcount
        tsv = os.path.join(tdir, f"class_{c}.tsv")
        with open(tsv, "w") as outfh:
            subprocess.run([jf, "dump", "-c", db], check=True, stdout=outfh, stderr=subprocess.PIPE)
        files.append(tsv)
    return files


def main(
    motif_name,
    meme_path,
    output_dir,
    alpha=0.5,
    use_canonical=True,
    threads=32,
    hashsize="200M",
    jellyfish_bin="jellyfish",
):
    os.makedirs(output_dir, exist_ok=True)

    meme_widths = parse_meme_widths(meme_path)

    k = None
    for key in normalize_for_lookup(motif_name):
        k = meme_widths[key]
        break

    assert k is not None, \
        f"[ERROR] Could not find k (width) for motif '{motif_name}' in MEME file."


    train_path = f"motif_training_csv/{motif_name}.csv"
    train_data = pd.read_csv(train_path)
    # remove N from all sequences
    train_data["sequence"] = train_data["sequence"].str.replace("N", "", regex=False)

    with tempfile.TemporaryDirectory() as tmp:
        class_fastas = write_class_fastas(train_data, tmp)

        files = count_with_jellyfish(
            class_fastas, k,
            threads=threads, hashsize=hashsize,
            canonical=use_canonical, jellyfish_bin=jellyfish_bin, tdir=tmp
        )
        
        model = KmerMultinomial(
            k=k, alpha=alpha, feature_mode="count",
            canonical=use_canonical,
            background_priors=BackgroundPriors(canonical=use_canonical)
        )

        freqs = {
            cls: pd.DataFrame(counts, columns=["kmer", "count"]) \
                for cls, counts in enumerate(model.files_to_sorted_counts_per_class(files))
        }
    
    cls_to_key = {
        0: "false",
        1: "true",
    }
    for (key, df) in freqs.items():
        df.to_csv(f"{output_dir}/{motif_name}_{cls_to_key[key]}_k-mers.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train motif-specific k-mer frequency tables from SPA .bin files using Jellyfish.")
    ap.add_argument("--motif_name", required=True, help="Motif name (used to find training CSV)")
    ap.add_argument("--meme_path", required=True, help="Path to the .meme file specifying per-motif k (width)")
    ap.add_argument("--output_dir", required=True, help="Directory to save output files")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet prior alpha (default: 0.5)")
    ap.add_argument("--use_canonical", action="store_true", help="Use canonical k-mers (default: False)")
    ap.add_argument("--threads", type=int, default=32, help="Number of threads for Jellyfish (default: 32)")
    ap.add_argument("--hashsize", default="200M", help="Hash size for Jellyfish (default: 200M)")
    ap.add_argument("--jellyfish_bin", default="jellyfish", help="Path to jellyfish binary (default: jellyfish)")
    args = ap.parse_args()
    main(**vars(args))
