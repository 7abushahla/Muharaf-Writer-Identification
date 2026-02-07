import csv
import os


def load_split_map(split_dir, seed, disjoint_mode="page", writer_policy=None):
    """
    Load page-disjoint or document-disjoint split map for a given seed.

    Args:
        split_dir: Directory containing split CSV files
        seed: Random seed used to generate splits
        disjoint_mode: Either "page" or "document"
        writer_policy: Optional writer policy suffix (e.g., "require_3way", "drop_if_lt3")

    Returns:
        dict: {page_id: split}, where split in {"train","val","test"}
    """
    split_dir = split_dir or "."
    seed_str = str(seed)
    
    # Build filename based on mode and policy
    disjoint_tag = "_document" if disjoint_mode == "document" else ""
    policy_tag = f"_{writer_policy}" if writer_policy else ""
    
    filename = f"page_disjoint{disjoint_tag}{policy_tag}_seed_{seed_str}.csv"
    path = os.path.join(split_dir, filename)
    
    # Try with the specified pattern first
    if not os.path.exists(path):
        # Fallback to simpler filename without policy
        filename = f"page_disjoint{disjoint_tag}_seed_{seed_str}.csv"
        path = os.path.join(split_dir, filename)
        
        if not os.path.exists(path):
            # List available split files to help user
            available = []
            if os.path.isdir(split_dir):
                available = [f for f in os.listdir(split_dir) if f.endswith(f"_seed_{seed_str}.csv")]
            
            error_msg = f"Split file not found: {path}"
            if available:
                error_msg += f"\n\nAvailable split files for seed {seed_str}:\n  " + "\n  ".join(available)
            else:
                error_msg += f"\n\nNo split files found for seed {seed_str} in {split_dir}"
                error_msg += f"\n\nGenerate splits first using: python page_disjoint_splits.py --seeds {seed_str} --disjoint-mode {disjoint_mode}"
            
            raise FileNotFoundError(error_msg)

    split_map = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"page_id", "writer", "split", "line_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Split file missing required columns {required}. Found: {reader.fieldnames}"
            )
        for row in reader:
            page_id = row["page_id"]
            split = row["split"]
            split_map[page_id] = split
    
    print(f"Loaded split file: {filename}")
    return split_map


def validate_split_map(split_map, page_ids):
    """
    Validate that all page_ids are present and split values are valid.
    """
    valid_splits = {"train", "val", "test"}
    missing = [pid for pid in page_ids if pid not in split_map]
    if missing:
        raise ValueError(
            f"Missing page_ids in split map (showing up to 10): {missing[:10]}"
        )
    invalid = [pid for pid, split in split_map.items() if split not in valid_splits]
    if invalid:
        raise ValueError(
            f"Invalid split labels found for page_ids (showing up to 10): {invalid[:10]}"
        )
