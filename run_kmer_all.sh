#!/usr/bin/env bash
set -euo pipefail
set +o braceexpand   # avoid accidental {a,b} expansion

# ----------- USER KNOBS -----------
OUTPUT_DIR="${OUTPUT_DIR:-./kmer_train_reports_acc_pretrain_32threads}"  # output folder
TOOL="${TOOL:-jellyfish}"                  # jellyfish | kmc3 | squeakr
JOBS="${JOBS:-1}"                          # parallel datasets
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-0}"      # 1=skip if report exists, 0=overwrite

# Hyperparameter sweeps (space-separated; Python accepts both styles too)
KS="${KS:-3 6 9 12 15 19}"
ALPHAS="${ALPHAS:-0.5 1.0}"
FEATURE_MODE="${FEATURE_MODE:-count binary}"
CANONICAL="${CANONICAL:-True}"
HANDLE_N="${HANDLE_N:-remove}"
SELECT_BY="${SELECT_BY:-acc}"              # acc | mcc

# Empirical background prior (optional)
USE_EMPIRICAL_PRIOR="${USE_EMPIRICAL_PRIOR:-False}"  # auto|True|False
PRETRAIN_CSV="${PRETRAIN_CSV:-}"                    # path to unlabeled CSV (or empty)
PRETRAIN_CSV_COL="${PRETRAIN_CSV_COL:-}"            # column name in CSV
PRETRAIN_CHUNK_LEN="${PRETRAIN_CHUNK_LEN:-0}"       # chunk windows if single long sequence
PRETRAIN_CHUNK_STRIDE="${PRETRAIN_CHUNK_STRIDE:-0}" # stride for chunking

# Tool-specific settings
JELLYFISH_BIN="${JELLYFISH_BIN:-jellyfish}"
JF_THREADS="${JF_THREADS:-32}"
JF_HASHSIZE="${JF_HASHSIZE:-200M}"
KMC_BIN="${KMC_BIN:-kmc}"
KMC_DUMP_BIN="${KMC_DUMP_BIN:-kmc_dump}"
KMC_THREADS="${KMC_THREADS:-32}"
KMC_MEM_GB="${KMC_MEM_GB:-4}"
SQUEAKR_COUNT_BIN="${SQUEAKR_COUNT_BIN:-squeakr-count}"
SQUEAKR_DUMP_BIN="${SQUEAKR_DUMP_BIN:-squeakr-dump}"
SQUEAKR_THREADS="${SQUEAKR_THREADS:-32}"
SQUEAKR_exact="${SQUEAKR_exact:-True}"
INF_THREADS="${INF_THREADS:-32}"  # threads for inference in KmerBaseline_external.py

mkdir -p "$OUTPUT_DIR"

# Map: dataset_folder -> output filename (uncomment what you want)
declare -A DATASET_OUTPUTS=(
    #Mouse
    ["./GUE/mouse/0"]="mouse0.txt"
    ["./GUE/mouse/1"]="mouse1.txt"
    ["./GUE/mouse/2"]="mouse2.txt"
    ["./GUE/mouse/3"]="mouse3.txt"
    ["./GUE/mouse/4"]="mouse4.txt"

    #TF
    ["./GUE/tf/0"]="tf0.txt"
    ["./GUE/tf/1"]="tf1.txt"
    ["./GUE/tf/2"]="tf2.txt"
    ["./GUE/tf/3"]="tf3.txt"
    ["./GUE/tf/4"]="tf4.txt"

    #Splice
    ["./GUE/splice/reconstructed"]="splice.txt"

    #Prom
    ["./GUE/prom/prom_300_all"]="prom_300_all.txt"
    ["./GUE/prom/prom_300_notata"]="prom_300_notata.txt"
    ["./GUE/prom/prom_300_tata"]="prom_300_tata.txt"
    ["./GUE/prom/prom_core_all"]="prom_core_all.txt"
    ["./GUE/prom/prom_core_notata"]="prom_core_notata.txt"
    ["./GUE/prom/prom_core_tata"]="prom_core_tata.txt"

    #EMP
    ["./GUE/EMP/H3"]="H3.txt"
    ["./GUE/EMP/H3K4me1"]="H3K4me1.txt"
    ["./GUE/EMP/H3K4me2"]="H3K4me2.txt"
    ["./GUE/EMP/H3K4me3"]="H3K4me3.txt"
    ["./GUE/EMP/H3K9ac"]="H3K9ac.txt"
    ["./GUE/EMP/H3K14ac"]="H3K14ac.txt"
    ["./GUE/EMP/H3K36me3"]="H3K36me3.txt"
    ["./GUE/EMP/H3K79me3"]="H3K79me3.txt"
    ["./GUE/EMP/H4"]="H4.txt"
    ["./GUE/EMP/H4ac"]="H4ac.txt"

    #Covid
    ["./GUE/virus/covid"]="covid.txt"

)

# KMER_SCRIPT="${KMER_SCRIPT:-./KmerBaseline_external.py}"
KMER_SCRIPT="${KMER_SCRIPT:-./KmerBaseline_rust.py}"


# lightweight semaphore
sem() {
  local max_jobs="$1"; shift
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$max_jobs" ]; do sleep 0.2; done
  "$@" &
}

for DATASET_FOLDER in "${!DATASET_OUTPUTS[@]}"; do
  OUTFILE="${OUTPUT_DIR}/${DATASET_OUTPUTS[$DATASET_FOLDER]}"

  if [[ "$SKIP_IF_EXISTS" == "1" && -f "$OUTFILE" ]]; then
    echo "Skipping $DATASET_FOLDER (exists: $OUTFILE)"
    continue
  fi

  CMD=( python "$KMER_SCRIPT"
        -dataset_folder "$DATASET_FOLDER"
        --tool "$TOOL"
        --ks $KS
        --alphas $ALPHAS
        --feature_mode $FEATURE_MODE
        --handle_n_setting $HANDLE_N
        --canonical $CANONICAL
        --select_by "$SELECT_BY"
        --use_empirical_prior "$USE_EMPIRICAL_PRIOR"
        --inf_threads "$INF_THREADS"
      )

  # prior inputs (optional)
  if [[ -n "${PRETRAIN_CSV}" ]]; then CMD+=( --pretrain_csv "$PRETRAIN_CSV" ); fi
  if [[ -n "${PRETRAIN_CSV_COL}" ]]; then CMD+=( --pretrain_csv_col "$PRETRAIN_CSV_COL" ); fi
  CMD+=( --pretrain_chunk_len "$PRETRAIN_CHUNK_LEN" --pretrain_chunk_stride "$PRETRAIN_CHUNK_STRIDE" )

  # Tool-specific flags
  case "$TOOL" in
    jellyfish)
      CMD+=( --jellyfish_bin "$JELLYFISH_BIN" --jf_threads "$JF_THREADS" --jf_hashsize "$JF_HASHSIZE" )
      ;;
    kmc3)
      CMD+=( --kmc_bin "$KMC_BIN" --kmc_dump_bin "$KMC_DUMP_BIN" --kmc_threads "$KMC_THREADS" --kmc_mem_gb "$KMC_MEM_GB" )
      ;;
    squeakr)
      CMD+=( --squeakr_count_bin "$SQUEAKR_COUNT_BIN" --squeakr_dump_bin "$SQUEAKR_DUMP_BIN" --squeakr_threads "$SQUEAKR_THREADS" --squeakr_exact "$SQUEAKR_exact" )
      ;;
    *)
      echo "Unknown TOOL: $TOOL" >&2; exit 1;;
  esac

  printf 'RUN:'; printf ' %q' "${CMD[@]}"; printf ' > %q\n' "$OUTFILE"
  sem "$JOBS" "${CMD[@]}" > "$OUTFILE" 2>&1
  # # FIXME: debug
  # ${CMD[@]}
done

wait
echo "All tasks complete. Reports in: $OUTPUT_DIR"
