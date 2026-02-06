#!/usr/bin/env python3
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def insert_utils_import(text):
    if "utils.page_disjoint" in text:
        return text, False

    block_lines = []
    if "import os" not in text:
        block_lines.append("import os")
    if "import sys" not in text:
        block_lines.append("import sys")
    block_lines.append("")
    block_lines.append("# Add repo root to sys.path for shared utilities")
    block_lines.append(
        "REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\"))"
    )
    block_lines.append("if REPO_ROOT not in sys.path:")
    block_lines.append("    sys.path.insert(0, REPO_ROOT)")
    block_lines.append("")
    block_lines.append("from utils.page_disjoint import load_split_map, validate_split_map")
    block = "\n".join(block_lines) + "\n"

    pattern = re.compile(r"^import os\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match:
        insert_pos = match.end()
        new_text = text[:insert_pos] + "\n" + block + text[insert_pos + 1 :]
        return new_text, True

    # fallback: insert after shebang or at top
    if text.startswith("#!"):
        first_nl = text.find("\n")
        if first_nl != -1:
            new_text = text[: first_nl + 1] + block + text[first_nl + 1 :]
            return new_text, True
    return block + text, True


def insert_split_envs(text):
    if re.search(r"^SPLIT_MODE\s*=", text, flags=re.MULTILINE):
        return text, False

    text_before = text

    text = re.sub(
        r"^main_dir\s*=\s*['\"]\.?/Lines['\"]\s*$",
        "main_dir = os.environ.get(\"LINES_DIR\", \"./Lines\")",
        text,
        flags=re.MULTILINE,
    )

    csv_block = (
        "csv_file = os.environ.get(\"MERGED_WRITER_CSV\", \"manual_labeling/merged_writer.csv\")\n"
        "if not os.path.exists(csv_file) and os.path.exists(\"merged_writer.csv\"):\n"
        "    csv_file = \"merged_writer.csv\"\n"
        "SPLIT_MODE = os.environ.get(\"SPLIT_MODE\", \"line\")\n"
        "SPLIT_DIR = os.environ.get(\"SPLIT_DIR\", \"./splits\")\n"
        "OUTPUT_PREFIX = \"PD_\" if SPLIT_MODE == \"page_disjoint\" else \"\"\n"
    )

    text = re.sub(
        r"^csv_file\s*=.*$",
        csv_block.rstrip(),
        text,
        count=1,
        flags=re.MULTILINE,
    )

    return text, text != text_before


def insert_attn_prefix(text):
    if "ATTN = OUTPUT_PREFIX + ATTN" in text:
        return text, False

    pattern = re.compile(r"^ATTN\s*=.*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return text, False

    insert_pos = match.end()
    new_text = text[:insert_pos] + "\nATTN = OUTPUT_PREFIX + ATTN" + text[insert_pos:]
    return new_text, True


def insert_page_ids_list(text):
    if "page_ids = []" in text:
        return text, False

    new_text = re.sub(
        r"images\s*=\s*\[\]\s*\nlabels\s*=\s*\[\]",
        "images = []\nlabels = []\npage_ids = []",
        text,
    )
    return new_text, new_text != text


def insert_page_ids_append(text):
    if "page_ids.append(" in text:
        return text, False

    pattern = re.compile(
        r"^(\s*)labels\.append\(writer_to_label\[writer_name\]\)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return text, False
    indent = match.group(1)
    replacement = (
        f"{indent}labels.append(writer_to_label[writer_name])\n"
        f"{indent}page_ids.append(image_filename)"
    )
    new_text = pattern.sub(replacement, text, count=1)
    return new_text, True


def replace_split_block(text):
    if "SPLIT_MODE == \"page_disjoint\"" in text:
        return text, False

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "# First split:" in line:
            start = i
            break
    if start is None:
        return text, False

    # find first train_test_split call
    first_call = None
    for i in range(start, len(lines)):
        if "train_test_split(" in lines[i]:
            first_call = i
            break
    if first_call is None:
        return text, False

    def find_call_end(idx):
        paren = 0
        found = False
        for j in range(idx, len(lines)):
            if not found and "train_test_split(" in lines[j]:
                found = True
            if found:
                paren += lines[j].count("(") - lines[j].count(")")
                if paren == 0:
                    return j
        return None

    first_end = find_call_end(first_call)
    if first_end is None:
        return text, False

    second_call = None
    for i in range(first_end + 1, len(lines)):
        if "train_test_split(" in lines[i]:
            second_call = i
            break
    if second_call is None:
        return text, False

    second_end = find_call_end(second_call)
    if second_end is None:
        return text, False

    new_block = [
        "# Split data (line-level or page-disjoint)",
        "page_ids = np.array(page_ids)",
        "if SPLIT_MODE == \"page_disjoint\":",
        "    split_map = load_split_map(SPLIT_DIR, SEED)",
        "    validate_split_map(split_map, page_ids)",
        "    split_labels = np.array([split_map[pid] for pid in page_ids])",
        "    train_mask = split_labels == \"train\"",
        "    val_mask = split_labels == \"val\"",
        "    test_mask = split_labels == \"test\"",
        "",
        "    train_images, train_labels = images[train_mask], labels[train_mask]",
        "    val_images, val_labels = images[val_mask], labels[val_mask]",
        "    test_images, test_labels = images[test_mask], labels[test_mask]",
        "",
        "    train_pages = len(set(page_ids[train_mask]))",
        "    val_pages = len(set(page_ids[val_mask]))",
        "    test_pages = len(set(page_ids[test_mask]))",
        "    print(f\"Page-disjoint split pages: train={train_pages}, val={val_pages}, test={test_pages}\")",
        "    print(f\"Page-disjoint split lines: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}\")",
        "else:",
        "    # First split: 70% training and 30% (validation + test)",
        "    train_images, temp_images, train_labels, temp_labels = train_test_split(",
        "        images, labels, test_size=0.3, random_state=SEED, stratify=labels",
        "    )",
        "",
        "    # Second split: Split the temporary set into 50% validation and 50% test",
        "    # Since the temporary set is 30% of the original data,",
        "    # this results in 15% validation and 15% test",
        "    val_images, test_images, val_labels, test_labels = train_test_split(",
        "        temp_images, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels",
        "    )",
    ]

    new_lines = lines[:start] + new_block + lines[second_end + 1 :]
    return "\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), True


def patch_delete_block(text):
    pattern = re.compile(
        r"^del\s+images,\s+labels,\s+writer_data,\s+csv_file,\s+temp_images,\s+temp_labels\s*$",
        re.MULTILINE,
    )
    if not pattern.search(text):
        return text, False

    block = (
        "del images, labels, writer_data, csv_file\n"
        "if 'temp_images' in locals():\n"
        "    del temp_images, temp_labels\n"
        "if 'page_ids' in locals():\n"
        "    del page_ids"
    )
    new_text = pattern.sub(block, text, count=1)
    return new_text, True


def patch_file(path):
    text = path.read_text(encoding="utf-8")
    original = text

    text, _ = insert_utils_import(text)
    text, _ = insert_split_envs(text)
    text, _ = insert_attn_prefix(text)
    text, _ = insert_page_ids_list(text)
    text, _ = insert_page_ids_append(text)
    text, _ = replace_split_block(text)
    text, _ = patch_delete_block(text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    patched = []
    skipped = []
    failed = []

    for path in MODELS_DIR.rglob("*.py"):
        try:
            changed = patch_file(path)
            if changed:
                patched.append(path)
            else:
                skipped.append(path)
        except Exception as e:
            failed.append((path, str(e)))

    print(f"Patched: {len(patched)}")
    print(f"Unchanged: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failures:")
        for path, err in failed:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
