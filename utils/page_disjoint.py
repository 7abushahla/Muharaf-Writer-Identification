import csv
import os


def load_split_map(split_dir, seed):
    """
    Load page-disjoint split map for a given seed.

    Returns:
        dict: {page_id: split}, where split in {"train","val","test"}
    """
    split_dir = split_dir or "."
    seed_str = str(seed)
    filename = f"page_disjoint_seed_{seed_str}.csv"
    path = os.path.join(split_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

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
