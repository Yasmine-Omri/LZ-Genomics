'''
New benchmarks
[
  ["human_nontata_promoters", "benchmarks/genomic_benchmarks/data/human_nontata_promoters"],
  ["human_enhancers_cohn", "benchmarks/genomic_benchmarks/data/human_enhancers_cohn"],
  ["human_enhancers_ensembl", "benchmarks/genomic_benchmarks/data/human_enhancers_ensembl"],
  ["human_ocr_ensembl", "benchmarks/genomic_benchmarks/data/human_ocr_ensembl"],
  ["human_ensembl_regulatory", "benchmarks/genomic_benchmarks/data/human_ensembl_regulatory"],
  ["dummy_mouse_enhancers_ensembl", "benchmarks/genomic_benchmarks/data/dummy_mouse_enhancers_ensembl"],
  ["demo_coding_vs_intergenomic_seqs", "benchmarks/genomic_benchmarks/data/demo_coding_vs_intergenomic_seqs"],
  ["demo_human_or_worm", "benchmarks/genomic_benchmarks/data/demo_human_or_worm"],

  ["gue_plus_fungi", "benchmarks/GUE_v2/fungi/species_20"],
  ["gue_plus_virus", "benchmarks/GUE_v2/virus/species_40"],
  ["bend_cpg", "benchmarks/BEND/data/cpg"],
  ["bend_histone_modification", "benchmarks/BEND/data/histone_modification"],
  ["bend_chromatin_accessibility", "benchmarks/BEND/data/chromatin_accessibility"],
  ["dart_task1", "benchmarks/DART-EVAL/data/task1"]
]

'''



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
# Map: dataset_folder -> output filename
declare -A DATASET_OUTPUTS=(
    ["benchmarks/genomic_benchmarks/data/human_nontata_promoters"]="human_nontata_promoters.txt"
    ["benchmarks/genomic_benchmarks/data/human_enhancers_cohn"]="human_enhancers_cohn.txt"
    ["benchmarks/genomic_benchmarks/data/human_enhancers_ensembl"]="human_enhancers_ensembl.txt"
    ["benchmarks/genomic_benchmarks/data/human_ocr_ensembl"]="human_ocr_ensembl.txt"
    ["benchmarks/genomic_benchmarks/data/human_ensembl_regulatory"]="human_ensembl_regulatory.txt"
    ["benchmarks/genomic_benchmarks/data/dummy_mouse_enhancers_ensembl"]="dummy_mouse_enhancers_ensembl.txt"
    ["benchmarks/genomic_benchmarks/data/demo_coding_vs_intergenomic_seqs"]="demo_coding_vs_intergenomic_seqs.txt"
    ["benchmarks/genomic_benchmarks/data/demo_human_or_worm"]="demo_human_or_worm.txt"

    ["benchmarks/GUE_v2/fungi/species_20"]="gue_plus_fungi.txt"
    ["benchmarks/GUE_v2/virus/species_40"]="gue_plus_virus.txt"
    ["benchmarks/BEND/data/cpg"]="bend_cpg.txt"
    ["benchmarks/BEND/data/histone_modification"]="bend_histone_modification.txt"
    ["benchmarks/BEND/data/chromatin_accessibility"]="bend_chromatin_accessibility.txt"
    ["benchmarks/DART-EVAL/data/task1"]="dart_task1.txt"
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
