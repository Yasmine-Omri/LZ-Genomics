#!/usr/bin/env python3
"""
LZ78-based EPI (Enhancer–Promoter Interaction) classifier for pairwise inputs.

- INPUT CSVs (train.csv, dev.csv, test.csv) must have columns: enhancer,promoter,label
- NO pretraining (per your constraint).
- Feature extractor (per pair):
    * c_E         = avg_log_loss(E)                             [bits/base]
    * c_P         = avg_log_loss(P)                             [bits/base]
    * H(P||E)     = avg_log_loss(P) after train_on_block(E) + reset_state() [bits/base]
    * H(E||P)     = avg_log_loss(E) after train_on_block(P) + reset_state() [bits/base]
    * NCD_sym     = 0.5*(NCD(E,P) + NCD(P,E)),
                    with joint approximated via chain rule:
                    C(E,P) ≈ C(E) + C(P|E) and C(P,E) ≈ C(P) + C(E|P)
                    where C(·) are TOTALS in bits: avg_log_loss * length

- Classifier: Logistic Regression (CPU), with L2 (C swept), features standardized.
- Validation metric: MCC, with exact threshold search over unique validation probabilities.
- Test metric: MCC at the chosen threshold (plus AUROC/AP as extra diagnostics).

ASSUMPTIONS (based on your clarifications):
- compute_test_loss_parallel([...]) returns dict with "avg_log_loss" (bits/base).
- compute_test_loss_parallel is read-only (does not update/mutate dictionary).
- train_on_block() updates dictionary.
- reset_state() resets current context to root, without flushing the dictionary (exact boundary rule you requested).
- We parallelize ACROSS pairs with ThreadPoolExecutor(max_workers=num_threads) and use num_threads=1 INSIDE
  compute_test_loss_parallel to avoid oversubscription (you run n_threads=64 on your system).
"""

import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from memory_profiler import profile
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

# --- Your LZ78 API ---
from lz78 import Sequence, CharacterMap, LZ78SPA


# ----------------------------
# Utilities
# ----------------------------

def parse_set(s: str):
    """
    Convert "{a, b, c}" to a Python set {a, b, c}.
    Numbers are parsed as float when possible; else left as strings.
    """
    s = s.strip().strip("{}").strip()
    if not s:
        return set()
    out = set()
    for item in s.split(","):
        item = item.strip()
        try:
            out.add(float(item))
        except ValueError:
            out.add(item)
    return out


# ----------------------------
# I/O and preprocessing
# ----------------------------

def load_pair_csv(path: str) -> pd.DataFrame:
    """
    Expect CSV columns: enhancer,promoter,label
    label must be 0/1
    """
    df = pd.read_csv(path)
    required = {"enhancer", "promoter", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}; expected {required}")
    # sanitize
    df["enhancer"] = df["enhancer"].astype(str).str.upper()
    df["promoter"] = df["promoter"].astype(str).str.upper()
    df["label"] = df["label"].astype(int)
    return df


def process_sequence_fast(seq: str) -> str:
    """
    Fast path: remove non-ACGT characters.
    """
    return "".join(ch for ch in seq if ch in "ACGT")


def handle_N_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fast processing to BOTH enhancer and promoter columns.
    """
    out = df.copy()
    out["enhancer"] = out["enhancer"].apply(process_sequence_fast)
    out["promoter"] = out["promoter"].apply(process_sequence_fast)
    return out


# ----------------------------
# LZ78 configuration + helpers
# ----------------------------

@dataclass
class LZConfig:
    gamma: float
    max_depth: Optional[int]  # None = unlimited
    # Keep defaults aligned with your original inference config
    lb: float = 1e-5
    ensemble_type: str = "entropy"
    ensemble_n: int = 10
    backshift_parsing: bool = True
    backshift_ctx_len: int = 20
    backshift_break_at_phrase: bool = True


def build_spa(cfg: LZConfig) -> LZ78SPA:
    """
    Create and configure an LZ78SPA instance with the given config.
    """
    spa = LZ78SPA(
        alphabet_size=4,
        gamma=cfg.gamma,
        compute_training_loss=False,
        max_depth=cfg.max_depth if cfg.max_depth is not None else None
    )
    spa.set_inference_config(
        lb=cfg.lb,
        ensemble_type=cfg.ensemble_type,
        ensemble_n=cfg.ensemble_n,
        backshift_parsing=cfg.backshift_parsing,
        backshift_ctx_len=cfg.backshift_ctx_len,
        backshift_break_at_phrase=cfg.backshift_break_at_phrase
    )
    return spa


def seq_obj(s: str) -> Sequence:
    return Sequence(s, charmap=CharacterMap("ACGT"))


def avg_loss_on_seq(spa: LZ78SPA, s: str) -> float:
    """
    Compute avg_log_loss (bits/base) on sequence s using current SPA state.
    Assumes compute_test_loss_parallel is read-only and returns a list of dicts
    with "avg_log_loss".
    """
    res_list = spa.compute_test_loss_parallel([seq_obj(s)], num_threads=1)  # inner=1 to avoid oversubscription
    if not res_list or not isinstance(res_list, list):
        raise RuntimeError("compute_test_loss_parallel did not return a list")
    res = res_list[0]
    if "avg_log_loss" not in res:
        raise KeyError(f"Result dict missing 'avg_log_loss': keys={list(res.keys())}")
    return float(res["avg_log_loss"])  # bits/base


# ----------------------------
# Feature extraction per pair
# ----------------------------

def features_for_pair(cfg: LZConfig, E: str, P: str) -> Tuple[float, float, float, float, float]:
    """
    Compute 5 features for (E, P):

      c_E          = avg_log_loss(E)                             [bits/base]
      c_P          = avg_log_loss(P)                             [bits/base]
      H(P||E)      = avg_log_loss(P) AFTER train_on_block(E) and reset_state() [bits/base]
      H(E||P)      = avg_log_loss(E) AFTER train_on_block(P) and reset_state() [bits/base]
      NCD_sym      = 0.5 * (NCD(E,P) + NCD(P,E))                 [unitless]

    NCD uses TOTALS:
      C(E)           = c_E * |E|
      C(P)           = c_P * |P|
      C(P|E) total   = H(P||E) * |P|
      C(E|P) total   = H(E||P) * |E|
      Joint approx:  C(E,P) ≈ C(E) + C(P|E),   C(P,E) ≈ C(P) + C(E|P)
      NCD(E,P) = [C(E,P) - min{C(E), C(P)}] / max{C(E), C(P)}
      NCD(P,E) = [C(P,E) - min{C(E), C(P)}] / max{C(E), C(P)}
      NCD_sym  = average of the two.
    """
    lenE = max(1, len(E))
    lenP = max(1, len(P))

    # Self complexities (avg bits/base)
    spaE = build_spa(cfg)
    c_E = avg_loss_on_seq(spaE, E)

    spaP = build_spa(cfg)
    c_P = avg_loss_on_seq(spaP, P)

    # Directed cross-entropies (avg bits/base):
    # H(P||E): train on E, reset to root, test on P
    spa_EP = build_spa(cfg)
    spa_EP.train_on_block(seq_obj(E))   # learn E
    spa_EP.reset_state()                # boundary (do NOT clear dictionary)
    H_P_given_E = avg_loss_on_seq(spa_EP, P)

    # H(E||P): train on P, reset to root, test on E
    spa_PE = build_spa(cfg)
    spa_PE.train_on_block(seq_obj(P))   # learn P
    spa_PE.reset_state()
    H_E_given_P = avg_loss_on_seq(spa_PE, E)

    # NCD (using totals)
    C_E = c_E * lenE
    C_P = c_P * lenP
    C_P_given_E = H_P_given_E * lenP
    C_E_given_P = H_E_given_P * lenE

    C_E_then_P = C_E + C_P_given_E   # approx joint C(E,P)
    C_P_then_E = C_P + C_E_given_P   # approx joint C(P,E)
    denom = max(C_E, C_P) if max(C_E, C_P) > 0 else 1.0

    NCD_E_P = (C_E_then_P - min(C_E, C_P)) / denom
    NCD_P_E = (C_P_then_E - min(C_E, C_P)) / denom
    NCD_sym = 0.5 * (NCD_E_P + NCD_P_E)

    return H_P_given_E, H_E_given_P, c_E, c_P, NCD_sym


def compute_features_parallel(cfg: LZConfig,
                              df: pd.DataFrame,
                              num_threads: int) -> np.ndarray:
    """
    Parallel feature extraction across pairs.
    - We parallelize ACROSS pairs with ThreadPoolExecutor(max_workers=num_threads).
    - INSIDE each feature computation we set num_threads=1 for the SPA,
      to avoid nested oversubscription (matches your system practice).
    Returns X of shape (N, 5) in the order:
      [H(P|E), H(E|P), c_E, c_P, NCD_sym]
    """
    X = np.zeros((len(df), 5), dtype=np.float64)

    def _work(row_idx: int, e: str, p: str):
        return row_idx, features_for_pair(cfg, e, p)

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = []
        for idx, row in df.iterrows():
            futures.append(ex.submit(_work, idx, row["enhancer"], row["promoter"]))

        for fut in as_completed(futures):
            idx, feats = fut.result()
            X[idx, :] = feats

    return X


# ----------------------------
# Threshold selection (MCC)
# ----------------------------

def best_threshold_mcc(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Exact threshold search over the unique predicted probabilities (plus midpoints).
    Returns (best_threshold, best_mcc).
    """
    uniq = np.unique(p)
    if len(uniq) == 1:
        # degenerate: all probabilities equal → MCC is undefined for different thresholds
        t = float(uniq[0])
        return t, matthews_corrcoef(y_true, (p >= t).astype(int))

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.unique(np.concatenate([uniq, mids, np.array([0.5])]))
    best_t, best_m = 0.5, -2.0
    for t in candidates:
        yhat = (p >= t).astype(int)
        mcc = matthews_corrcoef(y_true, yhat)
        if mcc > best_m:
            best_m, best_t = mcc, t
    return float(best_t), float(best_m)


# ----------------------------
# Main training / validation / test
# ----------------------------
@profile
def main():
    parser = argparse.ArgumentParser(description="Pairwise LZ78-based EPI classifier (no pretraining).")
    parser.add_argument("-dataset_folder", type=str, required=True,
                        help="Folder with train.csv, dev.csv, test.csv (enhancer,promoter,label).")
    parser.add_argument("--gamma", type=str, required=True,
                        help="Set of gamma values, e.g., '{0.1, 0.33, 0.5, 0.75, 1, 3, 5}'.")
    parser.add_argument("--max_depth", type=str, default="{}",
                        help="Set of max depths for the LZ78 tree, e.g., '{4, 8, 12}'. Empty means only None.")
    parser.add_argument("--clf_C", type=str, default="{1.0}",
                        help="Set of C (inverse L2 strength) for logistic regression, e.g., '{0.1, 0.3, 1, 3, 10}'.")
    parser.add_argument("--num_threads", type=int, required=True,
                        help="Threads for parallel pair feature extraction (outer level).")
    args = parser.parse_args()

    # Paths
    train_path = os.path.join(args.dataset_folder, "train.csv")
    dev_path   = os.path.join(args.dataset_folder, "dev.csv")
    test_path  = os.path.join(args.dataset_folder, "test.csv")

    # Load data
    t0 = time.perf_counter()
    train_df = load_pair_csv(train_path)
    dev_df   = load_pair_csv(dev_path)
    test_df  = load_pair_csv(test_path)

    # Handle Ns (fast removal of non-ACGT)
    train_df = handle_N_pairs(train_df)
    dev_df   = handle_N_pairs(dev_df)
    test_df  = handle_N_pairs(test_df)

    y_train = train_df["label"].values.astype(int)
    y_dev   = dev_df["label"].values.astype(int)
    y_test  = test_df["label"].values.astype(int)

    # Hyperparam grids
    gammas     = sorted(float(x) for x in parse_set(args.gamma))
    depths_raw = parse_set(args.max_depth)
    max_depths = [None] + [int(x) for x in depths_raw] if depths_raw else [None]
    Cs         = sorted(float(x) for x in parse_set(args.clf_C)) or [1.0]

    print("----- TRAIN/VAL (MCC with threshold search) -----", flush=True)
    print("gamma,max_depth,C,num_threads, "
          "build_train_feats_s,fit_s,build_dev_feats_s, val_mcc, best_t", flush=True)

    best = None  # (val_mcc, cfg, scaler, clf, t_star)
    results = []

    # Grid over gamma, depth, C
    for gamma in gammas:
        for depth in max_depths:
            cfg = LZConfig(gamma=gamma, max_depth=depth)

            # Build TRAIN features (parallel across pairs)
            s1 = time.perf_counter()
            X_train = compute_features_parallel(cfg, train_df, args.num_threads)
            e1 = time.perf_counter()

            # Standardize features
            scaler = StandardScaler().fit(X_train)
            Xtr = scaler.transform(X_train)

            for Cval in Cs:
                # Fit logistic regression (CPU, robust)
                clf = LogisticRegression(solver="lbfgs", C=Cval, max_iter=1000)
                s2 = time.perf_counter()
                clf.fit(Xtr, y_train)
                e2 = time.perf_counter()

                # Build DEV features and evaluate MCC across thresholds
                s3 = time.perf_counter()
                X_dev = compute_features_parallel(cfg, dev_df, args.num_threads)
                Xdv   = scaler.transform(X_dev)
                p_dev = clf.predict_proba(Xdv)[:, 1]
                t_star, mcc_dev = best_threshold_mcc(y_dev, p_dev)
                e3 = time.perf_counter()

                print(f"{gamma},{depth},{Cval},{args.num_threads}, "
                      f"{(e1-s1):.3f},{(e2-s2):.3f},{(e3-s3):.3f}, "
                      f"{mcc_dev:.4f},{t_star:.4f}", flush=True)

                results.append({
                    "gamma": gamma,
                    "max_depth": depth,
                    "C": Cval,
                    "val_mcc": mcc_dev,
                    "t_star": t_star,
                    "build_train_feats_s": (e1 - s1),
                    "fit_s": (e2 - s2),
                    "build_dev_feats_s": (e3 - s3),
                })

                if (best is None) or (mcc_dev > best[0]):
                    best = (mcc_dev, cfg, scaler, clf, t_star)

    if best is None:
        raise RuntimeError("No model trained (empty grid or failure during training).")

    val_mcc_best, cfg_best, scaler_best, clf_best, t_star_best = best
    print("\n--- BEST ON DEV ---")
    print({
        "gamma": cfg_best.gamma,
        "max_depth": cfg_best.max_depth,
        "C": getattr(clf_best, "C", None),
        "val_mcc": val_mcc_best,
        "t_star": t_star_best
    })

    # TEST set evaluation at fixed threshold
    print("\n----- TEST -----", flush=True)
    sT = time.perf_counter()
    X_test = compute_features_parallel(cfg_best, test_df, args.num_threads)
    Xte    = scaler_best.transform(X_test)
    p_test = clf_best.predict_proba(Xte)[:, 1]
    yhat   = (p_test >= t_star_best).astype(int)
    mcc    = matthews_corrcoef(y_test, yhat)
    try:
        auroc = roc_auc_score(y_test, p_test)
    except Exception:
        auroc = float("nan")
    try:
        ap = average_precision_score(y_test, p_test)
    except Exception:
        ap = float("nan")
    eT = time.perf_counter()

    print(f"TEST_MCC:   {mcc:.4f}")
    print(f"TEST_AUROC: {auroc:.4f}")
    print(f"TEST_AP:    {ap:.4f}")
    print(f"TEST_time_s:{(eT - sT):.3f}")

    t1 = time.perf_counter()
    print("\n----- TIME PROFILE -----")
    print(f"Total wall time: {(t1 - t0):.3f} s")

    # Optional: save artifacts if desired.
    # from joblib import dump
    # dump({"cfg": cfg_best, "scaler": scaler_best, "clf": clf_best, "t_star": t_star_best},
    #      "epi_lz78_model.joblib")


if __name__ == "__main__":
    main()
