from lz78 import spa_from_file, get_top_counts_at_depths, CharacterMap
import json


def main(
    spa_path: str,
    min_depth: int,
    max_depth: int,
    topk: int
):
    spa = spa_from_file(spa_path)
    print(f"Top counts at depths for SPA file: {spa_path}")

    res = get_top_counts_at_depths(
        spa,
        min_depth=min_depth,
        max_depth=max_depth,
        charmap=CharacterMap("ACGT"),
        topk=topk,
    )
    # sort by depth and then by count
    sorted_res = {}
    for depth in range(min_depth, max_depth + 1):
        sorted_res[depth] = {k: v for (k, v) in sorted(res[depth].items(), key=lambda x: x[1], reverse=True)}

    print(json.dumps(sorted_res, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get top counts at depths from a SPA file.")
    
    # Define command-line arguments
    parser.add_argument("--spa_path", type=str, required=True, help="Path to the .bin spa file")
    parser.add_argument("--min_depth", type=int, required=True, help="Minimum depth")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth")
    parser.add_argument("--topk", type=int, required=False, default=None, help="Top K sequences to retrieve")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Access arguments
    spa_path = args.spa_path
    min_depth = args.min_depth
    max_depth = args.max_depth
    topk = args.topk

    main(
        spa_path=spa_path,
        min_depth=min_depth,
        max_depth=max_depth,
        topk=topk
    )

"""
Usage example:

python get_top_counts_at_depths.py --spa_path results/best_spas/prom_prom_300_tata_0.bin --min_depth 4 --max_depth 10 --topk 25
"""