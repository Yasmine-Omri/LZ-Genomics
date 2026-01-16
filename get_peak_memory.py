import re

def str_to_memory_in_mib(mem_str: str) -> float:
    """
    Convert memory string like '195.5 MiB' or '2.3 GiB' to MiB float.
    """
    parts = mem_str.split()
    if len(parts) != 2:
        raise ValueError(f"Unexpected memory format: {mem_str}")
    
    value, unit = parts
    value = float(value)
    
    if unit == "MiB":
        return value
    elif unit == "GiB":
        return value * 1024
    elif unit == "KiB":
        return value / 1024
    else:
        raise ValueError(f"Unknown memory unit: {unit}")


def main(
    input_file: str,
):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    peak_memory = None
    memory_start = False
    for line in lines:
        if "-----MEMORY REPORT" in line:
            memory_start = True
            continue
        if not memory_start:
            continue
        
        #    264    195.5 MiB    195.5 MiB           1   @profile
        match = re.match(r"^\s*\d+\s+([\d\.]+\s+\w+)\s+([\d\.]+\s+\w+)\s+\d+\s+.*", line)
        if match:
            current_memory = str_to_memory_in_mib(match.group(1))
            if peak_memory is None or current_memory > peak_memory:
                peak_memory = current_memory

    print(peak_memory)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get peak memory usage from memory profiler log")
    parser.add_argument("input_file", type=str, help="Path to memory profiler log file")
    args = parser.parse_args()
    main(args.input_file)