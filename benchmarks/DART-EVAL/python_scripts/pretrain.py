'''
NUMBER OF GBs to consume
MAX TREE DEPTH
Hyperparams (entropy ensemble heuristic, N handling)
?? efficient way of consuming the "5GB" while staying memory efficient, should we fetch it one shorter seq at a time?

'''
#!/usr/bin/env python3
import argparse, io, os, re, sys, time, random, tracemalloc, resource

from lz78 import Sequence, CharacterMap, LZ78SPA

ALPHABET = "ACGT"
CHARMAP = CharacterMap(ALPHABET)

def parse_bytes(s: str) -> int:
    """Parse byte strings like 5G, 500M, 2500000000 -> int(bytes). Uses binary units."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?)b?", s)
    if not m:
        raise ValueError(f"Invalid bytes spec: {s}")
    val, unit = m.groups()
    mult = {"":1, "k":1024, "m":1024**2, "g":1024**3, "t":1024**4}[unit]
    return int(float(val) * mult)

def limit_label(bytes_limit: int, original: str) -> str:
    """
    Create a friendly label for filename, preferring integer GiB if exact.
    Examples: 5GB, 512MB, 2500000000B.
    """
    if bytes_limit % (1024**3) == 0:
        return f"{bytes_limit // (1024**3)}GB"
    if bytes_limit % (1024**2) == 0:
        return f"{bytes_limit // (1024**2)}MB"
    return f"{bytes_limit}B"

def clean_chunk(text: str, handle_n: str) -> str:
    if handle_n == "remove":
        return "".join(c for c in text if c in ALPHABET)
    elif handle_n == "random":
        return "".join(c if c in ALPHABET else random.choice(ALPHABET) for c in text)
    else:
        return text  # not expected

def stream_pretrain(
    path: str,
    byte_limit: int,
    spa: LZ78SPA,
    include_prev_context: bool,
    handle_n: str,
    read_chunk_bytes: int = 64 * 1024 * 1024,  # 64 MiB I/O
    train_block_len: int = 4 * 1024 * 1024,     # ~4 Mi symbols per train call
) -> int:
    """
    Stream the first `byte_limit` bytes from file, train incrementally, and
    return the actual number of bytes read from disk.
    """
    consumed = 0
    with open(path, "rb", buffering=0) as f:
        reader = io.BufferedReader(f, buffer_size=read_chunk_bytes)
        while consumed < byte_limit:
            to_read = min(read_chunk_bytes, byte_limit - consumed)
            raw = reader.read(to_read)
            if not raw:
                break
            consumed += len(raw)
            text = raw.decode("utf-8", errors="ignore").upper()
            text = clean_chunk(text, handle_n)

            # Slice into manageable training blocks (for memory/cache efficiency).
            for i in range(0, len(text), train_block_len):
                block = text[i:i+train_block_len]
                if not block:
                    continue
                if not include_prev_context:
                    spa.reset_state()
                spa.train_on_block(Sequence(block, charmap=CHARMAP))
    return consumed

def main():
    ap = argparse.ArgumentParser(
        description="Pretrain ONE LZ78 tree on the first X bytes of an unlabeled DNA text file."
    )
    ap.add_argument("--pretrain_file", required=True, help="Path to unlabeled DNA text (continuous A/C/G/T/N…).")
    ap.add_argument("--limit", required=True, help="Byte budget like 5G, 500M, 2500000000.")
    ap.add_argument("--max_depth", type=int, default=None, help="Max tree depth (None = unlimited).")
    ap.add_argument(
        "--include_prev_context",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        required=True,
        help="Set to True to keep context across chunks, False to reset between chunks."
    )
    ap.add_argument("--handle_n_setting", choices=["remove","random"], default="remove",
                    help="How to handle non-ACGT chars (default: remove).")
    ap.add_argument("--output_dir", default="best_spas", help="Directory to write the .bin (default: best_spas)")
    args = ap.parse_args()

    if not os.path.exists(args.pretrain_file):
        print(f"File not found: {args.pretrain_file}", file=sys.stderr)
        sys.exit(1)

    byte_limit = parse_bytes(args.limit)
    label = limit_label(byte_limit, args.limit)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"pretrain_file_{label}.bin")

    # Build the SPA (single tree)
    spa = LZ78SPA(
        alphabet_size=len(ALPHABET),
        compute_training_loss=False,
        max_depth=args.max_depth if args.max_depth is not None else None,
    )
    # (No ensemble/threading here; those are inference/testing concerns.)

    # ---- Time + memory profiling ----
    overall_start = time.perf_counter()
    tracemalloc.start()

    bytes_consumed = stream_pretrain(
        path=args.pretrain_file,
        byte_limit=byte_limit,
        spa=spa,
        include_prev_context=args.include_prev_context,
        handle_n=args.handle_n_setting,
    )

    # Save model
    spa_bytes = bytes(spa.to_bytes())
    with open(out_path, "wb") as fp:
        fp.write(spa_bytes)

    overall_end = time.perf_counter()
    elapsed = overall_end - overall_start

    # Memory stats
    current, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ru_maxrss is in kilobytes on Linux, bytes on macOS. We normalize to MiB.
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss_kb = ru.ru_maxrss
    # Heuristic: if value is very large, assume KB (Linux); else if macOS bytes, convert to KB then MiB.
    # But better: detect platform.
    import platform
    if platform.system() == "Darwin":
        peak_rss_mib = ru_maxrss_kb / (1024 * 1024)  # bytes -> MiB
    else:
        peak_rss_mib = ru_maxrss_kb / 1024           # kilobytes -> MiB

    spa_size_mib = len(spa_bytes) / (1024 * 1024)
    peak_tracemalloc_mib = peak_tracemalloc / (1024 * 1024)

    # ---- Report ----
    print("----- PRETRAIN SUMMARY -----")
    print(f"Pretrain file            : {args.pretrain_file}")
    print(f"Byte limit (requested)   : {byte_limit} bytes ({label})")
    print(f"Bytes actually consumed  : {bytes_consumed} bytes")
    print(f"Max depth                : {args.max_depth if args.max_depth is not None else 'unlimited'}")
    print(f"Include prev context     : {bool(args.include_prev_context)}")
    print(f"N handling               : {args.handle_n_setting}")
    print(f"Output path              : {out_path}")
    print("----- TIME PROFILING -----")
    print(f"Total pretrain time      : {elapsed:.3f} s")
    if bytes_consumed:
        print(f"Throughput               : {bytes_consumed/elapsed/1e6:.2f} MB/s")
    print("----- MEMORY PROFILING -----")
    print(f"SPA size on disk         : {spa_size_mib:.2f} MiB")
    print(f"Peak RSS (OS-reported)   : {peak_rss_mib:.2f} MiB")
    print(f"Peak tracemalloc         : {peak_tracemalloc_mib:.2f} MiB")

if __name__ == "__main__":
    main()
