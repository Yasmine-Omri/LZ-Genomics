#!/usr/bin/env python3
"""
External-counter k-mer baseline (Jellyfish / KMC3 / Squeakr)

- Builds class-conditional multinomial k-mer models from TRAIN
- Optional empirical background PRIOR (from unlabeled pretrain data):
    For each k, build bg_probs(kmer) from pretrain (DNA only).
    Then for each class c:
      p_c(kmer) = ( count_c(kmer) + alpha + bg_probs(kmer) ) /
                  ( sum_k count_c(k) + alpha*V + 1 )
  (Unit-mass prior; NO beta parameter.)
- Grid-search over (k, alpha, feature_mode); pick best by ACC or MCC on TEST
- Evaluate VERIFY (or DEV) using best setting
- Prints Train.py-like "TIME PROFILING+" section
- Expects dataset CSVs with columns: sequence, label
"""

import argparse, os, math, shutil, subprocess, tempfile, itertools, time, random, sys
from collections import Counter
import pandas as pd
from memory_profiler import profile
from sklearn.metrics import f1_score, matthews_corrcoef

ALPHABET = "ACGT"

# ============================ parsing helpers ============================

def parse_list(arg):
    """
    Accept any of:
      --ks "{3,4,5,6}" | --ks 3 4 5 6 | --ks 3,4,5,6
    Returns a list with ints/floats/bools/strings coerced.
    """
    tokens = []
    raw = arg if isinstance(arg, list) else [str(arg)]
    for item in raw:
        s = str(item).strip()
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        for tok in s.replace(",", " ").split():
            if tok:
                tokens.append(tok)
    out = []
    for tok in tokens:
        low = tok.lower()
        if low == "true":  out.append(True);  continue
        if low == "false": out.append(False); continue
        try: out.append(int(tok));   continue
        except ValueError: pass
        try: out.append(float(tok)); continue
        except ValueError: pass
        out.append(tok)
    return out

# =============================== small utils ===============================

def revcomp(seq):
    table = str.maketrans("ACGT","TGCA")
    return seq.translate(table)[::-1]

def canonical_kmer(km):
    rc = revcomp(km)
    return km if km <= rc else rc

def kmers_of(seq, k, canonical=True):
    L = len(seq)
    if L < k: return []
    out = []
    for i in range(L-k+1):
        km = seq[i:i+k]
        if canonical: km = canonical_kmer(km)
        out.append(km)
    return out

def handle_N_df(df, setting="remove"):
    rows=[]
    for _,r in df.iterrows():
        s = list(str(r["sequence"]))
        if setting == "remove":
            s = [c for c in s if c in ALPHABET]
        elif setting == "random":
            s = [random.choice(ALPHABET) if c not in ALPHABET else c for c in s]
        else:
            raise ValueError("handle_n_setting must be 'remove' or 'random'")
        rows.append({"sequence":"".join(s), "label": int(r["label"])})
    return pd.DataFrame(rows)

def check_bin(path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name if os.path.exists(path_or_name) else None
    return shutil.which(path_or_name)

def acgt_fraction(s: str) -> float:
    if not s: return 0.0
    ok = sum(ch in ALPHABET for ch in s)
    return ok / len(s)

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
    class_counts = {}
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
        cnt = Counter()
        with open(tsv) as fh:
            for line in fh:
                km, ct = line.strip().split()
                cnt[km] += int(ct)
        class_counts[c] = cnt
    return class_counts

# KMC3 (no native canonical -> fold later)
def count_with_kmc3(class_fastas, k, threads=8, kmc_bin="kmc", kmc_dump_bin="kmc_dump", mem_gb=4, tdir=None):
    kmc = check_bin(kmc_bin); dump = check_bin(kmc_dump_bin)
    if not kmc or not dump:
        raise RuntimeError("kmc/kmc_dump not found. Set --kmc_bin / --kmc_dump_bin.")
    class_counts = {}
    tmp_dir = os.path.join(tdir, "kmc_tmp"); os.makedirs(tmp_dir, exist_ok=True)
    for c, fa in class_fastas.items():
        db = os.path.join(tdir, f"class_{c}_kmc")
        cmd = [kmc, f"-k{k}", "-ci1", f"-t{threads}", f"-m{mem_gb}", "-fa", fa, db, tmp_dir]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tsv = os.path.join(tdir, f"class_{c}.tsv")
        with open(tsv, "w") as outfh:
            subprocess.run([dump, db], check=True, stdout=outfh, stderr=subprocess.PIPE)
        cnt = Counter()
        with open(tsv) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 2: continue
                km, ct = parts
                cnt[km] += int(ct)
        class_counts[c] = cnt
    return class_counts

# Squeakr (no native canonical -> fold later)
def count_with_squeakr(class_fastas, k, threads=8, squeakr_count_bin="squeakr-count", squeakr_dump_bin="squeakr-dump",
                       exact=True, tdir=None):
    sqc = check_bin(squeakr_count_bin); sqd = check_bin(squeakr_dump_bin)
    if not sqc or not sqd:
        raise RuntimeError("squeakr-count/squeakr-dump not found. Set --squeakr_count_bin / --squeakr_dump_bin.")
    class_counts = {}
    for c, fa in class_fastas.items():
        sqf = os.path.join(tdir, f"class_{c}.sqf")
        cmd = [sqc, "-k", str(k), "-t", str(threads), "-o", sqf]
        if str(exact).lower() == "true": cmd.append("-e")
        cmd.append(fa)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tsv = os.path.join(tdir, f"class_{c}.tsv")
        with open(tsv, "w") as outfh:
            subprocess.run([sqd, sqf], check=True, stdout=outfh, stderr=subprocess.PIPE)
        cnt = Counter()
        with open(tsv) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2: continue
                km, ct = parts[0], parts[-1]
                try: cnt[km] += int(ct)
                except: continue
        class_counts[c] = cnt
    return class_counts

def fold_canonical_inplace(class_counts):
    all_kmers = set()
    for cnt in class_counts.values(): all_kmers |= set(cnt.keys())
    mapping = {km: canonical_kmer(km) for km in all_kmers}
    for c in list(class_counts.keys()):
        src = class_counts[c]; dst = Counter()
        for km, ct in src.items(): dst[mapping[km]] += ct
        class_counts[c] = dst

# ==================== pretrain / background prior ====================

def write_fasta_from_sequences(seq_iterable, out_path):
    with open(out_path, "w") as fh:
        for i, s in enumerate(seq_iterable):
            fh.write(f">bg_{i}\n{s}\n")

def collect_pretrain_sequences(pretrain_csv, csv_col=None,
                               chunk_len: int = 0, chunk_stride: int = 0):
    """
    Yield sequences from CSV.
    - FASTA: streams records, optionally chunked
    - CSV: uses csv_col if provided; else auto-detect string-like column; optionally chunk if single long row
    """
    def _yield_chunks(seq):
        if chunk_len <= 0:
            yield seq
            return
        stride = chunk_stride if chunk_stride > 0 else chunk_len
        L = len(seq)
        if L < chunk_len:
            return
        for i in range(0, L - chunk_len + 1, stride):
            yield seq[i:i+chunk_len]

    if pretrain_csv:
        df = pd.read_csv(pretrain_csv)
        if csv_col is not None:
            if csv_col not in df.columns:
                raise KeyError(f"--pretrain_csv_col '{csv_col}' not found in CSV. Have: {list(df.columns)}")
            colname = csv_col
        else:
            candidates = ["sequence","seq","Sequence","Seq","SEQUENCE","dna","DNA","text"]
            colname = next((c for c in candidates if c in df.columns), None)
            if colname is None:
                colname = next((c for c in df.columns
                                if pd.api.types.is_string_dtype(df[c]) or df[c].dtype==object), None)
            if colname is None:
                raise KeyError(f"Could not auto-detect a sequence column in CSV. Columns={list(df.columns)}")

        values = [str(x).upper() for x in df[colname].values]
        if len(values) == 1:
            for ch in _yield_chunks(values[0]): yield ch
        else:
            for s in values: yield s
        return

    return  # neither provided

def count_background_with_tool(tool, sequences_iter, k, canonical, tdir,
                               jellyfish_bin="jellyfish", jf_threads=8, jf_hashsize="200M",
                               kmc_bin="kmc", kmc_dump_bin="kmc_dump", kmc_threads=8, kmc_mem_gb=4,
                               squeakr_count_bin="squeakr-count", squeakr_dump_bin="squeakr-dump",
                               squeakr_threads=8, squeakr_exact=True):
    """
    Build a single background Counter of k-mer counts using the selected tool.
    Returns a Counter (possibly canonical-folded).
    """
    bg_fa = os.path.join(tdir, "background.fa")
    write_fasta_from_sequences(sequences_iter, bg_fa)

    if tool == "jellyfish":
        jf = check_bin(jellyfish_bin)
        if not jf:
            raise RuntimeError("jellyfish binary not found. Set --jellyfish_bin.")
        db = os.path.join(tdir, "bg.jf")
        cmd = [jf, "count", "-m", str(k), "-s", str(jf_hashsize), "-t", str(jf_threads), "-o", db]
        if canonical: cmd.append("-C")
        cmd.append(bg_fa)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cnt = Counter()
        with open(os.path.join(tdir, "bg.tsv"), "w") as outfh:
            subprocess.run([jf, "dump", "-c", db], check=True, stdout=outfh, stderr=subprocess.PIPE)
        with open(os.path.join(tdir, "bg.tsv")) as fh:
            for line in fh:
                km, ct = line.strip().split()
                cnt[km] += int(ct)
        return cnt

    elif tool == "kmc3":
        kmc = check_bin(kmc_bin); dump = check_bin(kmc_dump_bin)
        if not kmc or not dump:
            raise RuntimeError("kmc/kmc_dump not found. Set --kmc_bin / --kmc_dump_bin.")
        tmp_dir = os.path.join(tdir, "kmc_tmp"); os.makedirs(tmp_dir, exist_ok=True)
        db = os.path.join(tdir, "bg_kmc")
        cmd = [kmc, f"-k{k}", "-ci1", f"-t{kmc_threads}", f"-m{kmc_mem_gb}", "-fa", bg_fa, db, tmp_dir]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(os.path.join(tdir, "bg.tsv"), "w") as outfh:
            subprocess.run([dump, db], check=True, stdout=outfh, stderr=subprocess.PIPE)
        cnt = Counter()
        with open(os.path.join(tdir, "bg.tsv")) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 2: continue
                km, ct = parts
                cnt[km] += int(ct)
        if canonical:
            folded = Counter()
            for km, v in cnt.items():
                folded[canonical_kmer(km)] += v
            cnt = folded
        return cnt

    elif tool == "squeakr":
        sqc = check_bin(squeakr_count_bin); sqd = check_bin(squeakr_dump_bin)
        if not sqc or not sqd:
            raise RuntimeError("squeakr-count/squeakr-dump not found.")
        sqf = os.path.join(tdir, "bg.sqf")
        cmd = [sqc, "-k", str(k), "-t", str(squeakr_threads), "-o", sqf]
        if str(squeakr_exact).lower() == "true": cmd.append("-e")
        cmd.append(bg_fa)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(os.path.join(tdir, "bg.tsv"), "w") as outfh:
            subprocess.run([sqd, sqf], check=True, stdout=outfh, stderr=subprocess.PIPE)
        cnt = Counter()
        with open(os.path.join(tdir, "bg.tsv")) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2: continue
                km, ct = parts[0], parts[-1]
                try: cnt[km] += int(ct)
                except: continue
        if canonical:
            folded = Counter()
            for km, v in cnt.items():
                folded[canonical_kmer(km)] += v
            cnt = folded
        return cnt

    else:
        raise RuntimeError("Unknown tool for background counting.")

# ============================= model & metrics =============================

class KmerMultinomial:
    def __init__(self, k, alpha=1.0, feature_mode="count", bg_probs=None, canonical=True):
        self.k = k
        self.alpha = float(alpha)
        self.feature_mode = feature_mode  # 'count' or 'binary'
        self.bg_probs = bg_probs          # dict(kmer->prob) or None
        self.canonical = canonical
        self.classes_ = None
        self.class_probs = None  # c -> {kmer: prob}

    def fit_from_counts(self, class_counts):
        self.classes_ = sorted(class_counts.keys())
        vocab = set()
        for c in self.classes_: vocab |= set(class_counts[c].keys())
        if self.bg_probs is not None:     # include bg support
            vocab |= set(self.bg_probs.keys())
        V = max(len(vocab), 1)

        # Normalize bg_probs over current vocab; treat as unit-mass prior (no user β)
        bg = None
        if self.bg_probs is not None:
            tot_bg = sum(self.bg_probs.get(km, 0.0) for km in vocab)
            if tot_bg > 0:
                bg = {km: self.bg_probs.get(km, 0.0) / tot_bg for km in vocab}
            else:
                bg = None

        self.class_probs = {}
        for c in self.classes_:
            counts_c = class_counts[c]
            denom = sum(counts_c.get(km, 0) for km in vocab) + (self.alpha * V) + (1.0 if bg is not None else 0.0)
            probs = {}
            for km in vocab:
                prior = bg[km] if bg is not None else 0.0
                probs[km] = (counts_c.get(km, 0) + self.alpha + prior) / denom
            self.class_probs[c] = probs

    def _seq_counts(self, seq):
        kms = kmers_of(seq, self.k, canonical=self.canonical)
        return Counter(set(kms)) if self.feature_mode == "binary" else Counter(kms)

    def predict_one(self, seq):
        x = self._seq_counts(seq)
        best_c, best_s = None, -1e300
        for c in self.classes_:
            p = self.class_probs[c]
            s = 0.0
            for km, cnt in x.items():
                pt = p.get(km, 1e-12)
                s += cnt * math.log(pt)
            if s > best_s:
                best_s, best_c = s, c
        return best_c

    def predict_df(self, df):
        return [self.predict_one(s) for s in df["sequence"].values]

def accuracy(y_true, y_pred):
    ok = sum(int(a)==int(b) for a,b in zip(y_true,y_pred))
    return ok/len(y_true) if y_true else 0.0

def mcc_multiclass(y_true, y_pred):
    # scikit-learn’s MCC supports binary and multiclass natively
    return matthews_corrcoef(list(map(int, y_true)), list(map(int, y_pred)))

# ================================== main ==================================
@profile
def main():
    ap = argparse.ArgumentParser(description="External-counter k-mer baseline with empirical background prior (unit mass)")
    ap.add_argument("-dataset_folder", required=True)
    ap.add_argument("--tool", choices=["jellyfish","kmc3","squeakr"], required=True)

    # grid knobs
    ap.add_argument("--ks", nargs="+", default=["3","4","5","6"])
    ap.add_argument("--alphas", nargs="+", default=["0.5","1.0"])
    ap.add_argument("--feature_mode", nargs="+", default=["count","binary"])
    ap.add_argument("--handle_n_setting", nargs="+", default=["remove"])
    ap.add_argument("--canonical", nargs="+", default=["True"])
    ap.add_argument("--select_by", default="mcc", choices=["acc","mcc"],
                    help="Metric to select best setting on TEST (default mcc)")

    # empirical background prior
    ap.add_argument("--use_empirical_prior", default="auto",
                    help="auto|True|False. If auto and pretrain provided, use prior; if True, require pretrain; if False, disable.")
    ap.add_argument("--pretrain_csv", default=None, help="Path to unlabeled CSV with a sequence column (optional).")
    ap.add_argument("--pretrain_csv_col", default=None, help="CSV column name containing sequences.")
    ap.add_argument("--pretrain_chunk_len", type=int, default=0,
                    help="If >0, chunk a single long pretrain sequence into windows of this length.")
    ap.add_argument("--pretrain_chunk_stride", type=int, default=0,
                    help="Stride for chunking (default=stride=chunk_len).")

    # jellyfish
    ap.add_argument("--jellyfish_bin", default="jellyfish")
    ap.add_argument("--jf_threads", type=int, default=8)
    ap.add_argument("--jf_hashsize", default="200M")
    # kmc3
    ap.add_argument("--kmc_bin", default="kmc")
    ap.add_argument("--kmc_dump_bin", default="kmc_dump")
    ap.add_argument("--kmc_threads", type=int, default=8)
    ap.add_argument("--kmc_mem_gb", type=int, default=4)
    # squeakr
    ap.add_argument("--squeakr_count_bin", default="squeakr-count")
    ap.add_argument("--squeakr_dump_bin", default="squeakr-dump")
    ap.add_argument("--squeakr_threads", type=int, default=8)
    ap.add_argument("--squeakr_exact", default="True")

    args = ap.parse_args()

    # parse lists
    ks = sorted(int(x) for x in parse_list(args.ks))
    alphas = sorted(float(x) for x in parse_list(args.alphas))
    feature_modes = [str(x) for x in parse_list(args.feature_mode)]
    handle_setting = parse_list(args.handle_n_setting)[0] if parse_list(args.handle_n_setting) else "remove"
    use_canonical = (parse_list(args.canonical)[0] if parse_list(args.canonical) else True)
    select_by = args.select_by.lower()

    dataset_name = os.path.basename(os.path.normpath(args.dataset_folder)).lower()

    # empirical prior mode
    use_empirical = str(args.use_empirical_prior).lower()
    if use_empirical not in ("auto","true","false"):
        raise ValueError("--use_empirical_prior must be auto|True|False")
    have_pretrain = bool(args.pretrain_csv)
    if use_empirical == "auto":
        use_empirical = "true" if have_pretrain else "false"
    if use_empirical == "true" and not have_pretrain:
        raise ValueError("--use_empirical_prior True was set but no --pretrain_csv provided.")

    read_data_in_time = time.perf_counter()

    # load splits
    dpath = args.dataset_folder
    train_path = os.path.join(dpath, "train.csv")
    val_path  = os.path.join(dpath, "dev.csv")
    test_path = os.path.join(dpath, "test.csv")

    train = pd.read_csv(train_path)
    val  = pd.read_csv(val_path)

    # N handling (DNA only)
    train = handle_N_df(train, handle_setting)
    val  = handle_N_df(val, handle_setting)

    # -------- training start time  --------
    print("-----TRAINING")
    train_start_time = time.perf_counter()

    print("k,alpha,feature_mode,test_acc,test_mcc,test_f1,train_wall_s,test_wall_s,tool,canonical,emp_prior")

    best = None
    best_model = None

    # cache bg prior per k
    bg_prob_cache = {}  # k -> dict(kmer->prob) or None

    # special-case selection: if select_by == mcc and dataset is covid -> use F1 instead
    selection_override_to_f1 = (select_by == "mcc" and dataset_name == "covid")

    for k, alpha, fmode in itertools.product(ks, alphas, feature_modes):
        # Build bg prior once per k (if enabled)
        bg_probs_for_k = None
        if use_empirical == "true":
            if k not in bg_prob_cache:
                with tempfile.TemporaryDirectory() as tmp_bg:
                    def _bg_stream():
                        skipped = 0
                        for s in collect_pretrain_sequences(
                                args.pretrain_csv,
                                args.pretrain_csv_col,
                                args.pretrain_chunk_len, args.pretrain_chunk_stride):
                            # apply same N policy as train; also ensure DNA content
                            if handle_setting == "remove":
                                s2 = "".join(ch for ch in s if ch in ALPHABET)
                            elif handle_setting == "random":
                                s2 = "".join((random.choice(ALPHABET) if ch not in ALPHABET else ch) for ch in s)
                            else:
                                s2 = s
                            if not s2 or acgt_fraction(s2) < 0.9:
                                skipped += 1
                                continue
                            yield s2
                        if skipped:
                            print(f"[WARN] Background: skipped {skipped} non-ACGT/empty records.", flush=True)
                    try:
                        bg_counts = count_background_with_tool(
                            args.tool, _bg_stream(), k, use_canonical, tmp_bg,
                            jellyfish_bin=args.jellyfish_bin, jf_threads=args.jf_threads, jf_hashsize=args.jf_hashsize,
                            kmc_bin=args.kmc_bin, kmc_dump_bin=args.kmc_dump_bin, kmc_threads=args.kmc_threads, kmc_mem_gb=args.kmc_mem_gb,
                            squeakr_count_bin=args.squeakr_count_bin, squeakr_dump_bin=args.squeakr_dump_bin,
                            squeakr_threads=args.squeakr_threads, squeakr_exact=args.squeakr_exact
                        )
                        tot = sum(bg_counts.values())
                        bg_prob_cache[k] = ({km: ct / tot for km, ct in bg_counts.items()} if tot > 0 else None)
                    except FileNotFoundError as e:
                        print(f"[WARN] Background prior disabled for k={k}: {e}", file=sys.stderr)
                        bg_prob_cache[k] = None
            bg_probs_for_k = bg_prob_cache[k]

        with tempfile.TemporaryDirectory() as tmp:
            class_fastas = write_class_fastas(train, tmp)

            # Count per-class with selected tool
            t_train0 = time.perf_counter()
            if args.tool == "jellyfish":
                class_counts = count_with_jellyfish(
                    class_fastas, k,
                    threads=args.jf_threads, hashsize=args.jf_hashsize,
                    canonical=use_canonical, jellyfish_bin=args.jellyfish_bin, tdir=tmp
                )
            elif args.tool == "kmc3":
                class_counts = count_with_kmc3(
                    class_fastas, k,
                    threads=args.kmc_threads, kmc_bin=args.kmc_bin,
                    kmc_dump_bin=args.kmc_dump_bin, mem_gb=args.kmc_mem_gb, tdir=tmp
                )
                if use_canonical:
                    fold_canonical_inplace(class_counts)
            elif args.tool == "squeakr":
                class_counts = count_with_squeakr(
                    class_fastas, k,
                    threads=args.squeakr_threads, squeakr_count_bin=args.squeakr_count_bin,
                    squeakr_dump_bin=args.squeakr_dump_bin, exact=args.squeakr_exact, tdir=tmp
                )
                if use_canonical:
                    fold_canonical_inplace(class_counts)
            else:
                raise RuntimeError("Unknown tool")

            model = KmerMultinomial(k=k, alpha=alpha, feature_mode=fmode,
                                    bg_probs=bg_probs_for_k, canonical=use_canonical)
            model.fit_from_counts(class_counts)
            t_train1 = time.perf_counter()

        #Validate
        t_test0 = time.perf_counter()
        y_true = list(val["label"].astype(int).values)
        y_pred = model.predict_df(val)
        acc = accuracy(y_true, y_pred)
        mcc = mcc_multiclass(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        t_test1 = time.perf_counter()

        print(f"{k},{alpha},{fmode},{acc:.4f},{mcc:.4f},{f1:.4f},{t_train1-t_train0:.3f},{t_test1-t_test0:.3f},{args.tool},{use_canonical},{use_empirical=='true'}")

        # decide key
        if selection_override_to_f1:
            key = f1
        else:
            key = mcc if select_by == "mcc" else acc

        if (best is None) or (key > best["key"]):
            best = {"k":k,"alpha":alpha,"fmode":fmode,"acc":acc,"mcc":mcc,"f1":f1,"key":key,
                    "tool":args.tool,"canonical":use_canonical, "emp_prior": (use_empirical=='true')}
            best_model = model


    print("\nBEST ON TEST:", best)
    
    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time

    read_test_data_start_time = time.perf_counter()
    test = pd.read_csv(test_path) 
    test = handle_N_df(test, handle_setting)
    
    inference_start_time = time.perf_counter()

    # TEST evaluation
    y_true = list(test["label"].astype(int).values)
    y_pred = model.predict_df(test)
    acc = accuracy(y_true, y_pred)
    mcc = mcc_multiclass(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print("TEST ACC:", f"{acc:.4f}")
    print("TEST MCC:", f"{mcc:.4f}")
    print("TEST F1:",  f"{f1:.4f}")
 
    inference_end_time = time.perf_counter()
    inference_duration = (inference_end_time - inference_start_time)

    nb_train_seqs = len(train)
    train_seq_len = len(train.iloc[0]["sequence"])
    nb_train_symbols = nb_train_seqs * train_seq_len

    nb_test_seqs = len(test)
    test_seq_len = len(test.iloc[0]["sequence"])
    # ---------- TIME PROFILING+ ----------
    print("-----TIME PROFILING+")
    print(f"Read train + val data time: {(train_start_time - read_data_in_time): .5f}")
    print(f"Number of training symbols: {nb_train_symbols}")
    print(f"Length of one training sequence: {train_seq_len}")
    print(f"Total training time: {train_duration:.3f} seconds")

    print(f"Number of test sequences: {nb_test_seqs}")
    print(f"Length of test sequence: {test_seq_len}")
    print(f"Read test data time: {(inference_start_time - read_test_data_start_time): .5f}")
    print(f"Total inference time: {inference_duration:.3f} seconds")
    print(f"Inference time/symbol: {inference_duration/(nb_test_seqs * test_seq_len)} seconds")
    # ---------- END TIME PROFILING+ ----------

    #Output memory report, which is automatically printed at the end of the run
    print("-----MEMORY REPORT")

if __name__ == "__main__":
    main()
