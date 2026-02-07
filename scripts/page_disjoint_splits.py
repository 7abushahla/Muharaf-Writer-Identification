#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import statistics
from collections import defaultdict

import pandas as pd
import xml.etree.ElementTree as ET


IMAGE_EXTS = (".png", ".jpg", ".jpeg")
WRITER_POLICY_CHOICES = (
    "allow_train_only",
    "drop_if_lt2",
    "drop_if_lt3",
    "require_3way",
    "allow_train_test_only",
)

DISJOINT_MODE_CHOICES = (
    "page",
    "document",
)


def parse_seeds(seeds_str):
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def load_public_pages(public_dir):
    if not public_dir or not os.path.isdir(public_dir):
        return None
    public_pages = set()
    for fn in os.listdir(public_dir):
        base, ext = os.path.splitext(fn)
        if ext.lower() in IMAGE_EXTS:
            public_pages.add(base)
    return public_pages


def extract_page_id_from_line_filename(filename):
    base, _ext = os.path.splitext(filename)
    m = re.match(r"^(.*)-\d+$", base)
    return m.group(1) if m else base


def extract_document_id(page_id, doc_id_re):
    m = doc_id_re.match(page_id)
    return m.group(1) if m else page_id


def build_doc_map_from_dir(documents_dir):
    """
    Build a mapping of page_id -> document_id from a documents directory
    where each subfolder corresponds to one multi-page document and contains page images.
    Any image files directly under the documents directory are treated as
    single-page documents (document_id == page_id).
    """
    if not documents_dir or not os.path.isdir(documents_dir):
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    page_to_doc = {}
    # first, handle single-page documents in the root
    for fn in os.listdir(documents_dir):
        if fn.startswith("."):
            continue
        path = os.path.join(documents_dir, fn)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(fn)
        if ext.lower() not in IMAGE_EXTS:
            continue
        page_id = base
        doc_id = base
        if page_id in page_to_doc and page_to_doc[page_id] != doc_id:
            raise ValueError(
                f"Page '{page_id}' appears in multiple document folders: "
                f"{page_to_doc[page_id]} and {doc_id}"
            )
        page_to_doc[page_id] = doc_id

    # then, handle multi-page documents in subfolders
    for doc_name in os.listdir(documents_dir):
        doc_path = os.path.join(documents_dir, doc_name)
        if not os.path.isdir(doc_path):
            continue
        for fn in os.listdir(doc_path):
            if fn.startswith("."):
                continue
            path = os.path.join(doc_path, fn)
            if not os.path.isfile(path):
                continue
            base, ext = os.path.splitext(fn)
            if ext.lower() not in IMAGE_EXTS:
                continue
            page_id = base
            if page_id in page_to_doc and page_to_doc[page_id] != doc_name:
                raise ValueError(
                    f"Page '{page_id}' appears in multiple document folders: "
                    f"{page_to_doc[page_id]} and {doc_name}"
                )
            page_to_doc[page_id] = doc_name

    if not page_to_doc:
        raise ValueError(f"No page images found in documents directory: {documents_dir}")

    return page_to_doc


def detect_lines_dir_mode(lines_dir):
    if not lines_dir or not os.path.isdir(lines_dir):
        return "missing"
    entries = os.listdir(lines_dir)
    has_root_images = any(fn.lower().endswith(IMAGE_EXTS) for fn in entries)
    has_root_txt = any(fn.lower().endswith(".txt") for fn in entries)
    if has_root_images or has_root_txt:
        return "flat"
    if any(os.path.isdir(os.path.join(lines_dir, e)) for e in entries):
        return "folder"
    return "flat"


def extract_arabic_handwritten_lines(xml_path):
    """Extract Arabic handwritten-cursive lines from a PAGE XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

    arabic_lines = []
    for text_region in root.findall(".//ns:TextRegion", ns):
        for text_line in text_region.findall(".//ns:TextLine", ns):
            primary_language = text_line.get("primaryLanguage")
            production = text_line.get("production")
            if primary_language == "Arabic" and production == "handwritten-cursive":
                text_equiv = text_line.find(".//ns:TextEquiv/ns:Unicode", ns)
                if text_equiv is not None and text_equiv.text:
                    arabic_lines.append(text_equiv.text.strip())
    return arabic_lines


def build_allowlist_from_xml(xml_dir, lines_dir, lines_dir_mode):
    if not xml_dir or not os.path.isdir(xml_dir):
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    allowlist = set()

    line_txt_index = None
    if lines_dir_mode == "flat":
        line_txt_index = defaultdict(list)
        for fn in os.listdir(lines_dir):
            if fn.lower().endswith(".txt"):
                page_id = extract_page_id_from_line_filename(fn)
                line_txt_index[page_id].append(fn)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.lower().endswith(".xml"):
            continue
        page_id = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(xml_dir, xml_file)
        arabic_lines = extract_arabic_handwritten_lines(xml_path)
        if not arabic_lines:
            continue
        arabic_set = set(arabic_lines)

        txt_files = []
        if lines_dir_mode == "folder":
            page_folder = os.path.join(lines_dir, page_id)
            if not os.path.isdir(page_folder):
                continue
            txt_files = [f for f in os.listdir(page_folder) if f.lower().endswith(".txt")]
            txt_files = [os.path.join(page_folder, f) for f in txt_files]
        else:
            txt_files = line_txt_index.get(page_id, [])
            txt_files = [os.path.join(lines_dir, f) for f in txt_files]

        for txt_path in txt_files:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read().strip()
            except Exception:
                continue
            if txt_content in arabic_set:
                base = os.path.splitext(os.path.basename(txt_path))[0]
                allowlist.add(base)
                allowlist.add(base + ".png")
                allowlist.add(base + ".jpg")
                allowlist.add(base + ".jpeg")
                allowlist.add(os.path.basename(txt_path))
    return allowlist


def build_line_counts(lines_dir, page_ids, allow_missing_lines, allowlist=None, lines_dir_mode=None):
    if not os.path.isdir(lines_dir):
        if not allow_missing_lines:
            raise FileNotFoundError(f"Lines directory not found: {lines_dir}")
        # fallback: treat each page as 1 line
        line_counts = {pid: 1 for pid in page_ids}
        return line_counts, [], [], True, "missing"

    entries = os.listdir(lines_dir)
    page_id_set = set(page_ids)
    mode = lines_dir_mode or detect_lines_dir_mode(lines_dir)

    line_counts = {pid: 0 for pid in page_ids}
    extra_pages = []

    if mode == "folder":
        # Folder-per-page structure
        for pid in page_ids:
            folder = os.path.join(lines_dir, pid)
            if not os.path.isdir(folder):
                continue
            count = 0
            for fn in os.listdir(folder):
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                if allowlist is not None:
                    base = os.path.splitext(fn)[0]
                    if fn not in allowlist and base not in allowlist:
                        continue
                count += 1
            line_counts[pid] = count

        # extra pages that exist as folders but not in CSV
        for e in entries:
            if os.path.isdir(os.path.join(lines_dir, e)) and e not in page_id_set:
                extra_pages.append(e)

        missing_pages = [pid for pid in page_ids if line_counts.get(pid, 0) == 0]
        return line_counts, missing_pages, sorted(extra_pages), False, "folder"

    # Flat file structure
    line_pages = set()
    for fn in entries:
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        if allowlist is not None:
            base = os.path.splitext(fn)[0]
            if fn not in allowlist and base not in allowlist:
                continue
        page_id = extract_page_id_from_line_filename(fn)
        line_pages.add(page_id)
        if page_id in line_counts:
            line_counts[page_id] += 1

    missing_pages = [pid for pid in page_ids if line_counts.get(pid, 0) == 0]
    extra_pages = sorted(line_pages - page_id_set)
    return line_counts, missing_pages, extra_pages, False, "flat"


def load_writer_pages(
    csv_path,
    lines_dir,
    allow_missing_lines=False,
    allowlist=None,
    lines_dir_mode=None,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Image Filename", "Writer Name (English)"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}. Found: {set(df.columns)}")

    page_to_writer = {}
    inconsistent_pages = []
    for _, row in df.iterrows():
        page_id = str(row["Image Filename"])
        writer = str(row["Writer Name (English)"])
        if page_id in page_to_writer and page_to_writer[page_id] != writer:
            inconsistent_pages.append(page_id)
        else:
            page_to_writer[page_id] = writer

    page_line_counts, missing_in_lines, extra_in_lines, lines_dir_missing, lines_dir_mode = (
        build_line_counts(
            lines_dir,
            page_to_writer.keys(),
            allow_missing_lines,
            allowlist=allowlist,
            lines_dir_mode=lines_dir_mode,
        )
    )

    writer_pages = defaultdict(list)
    for page_id, writer in page_to_writer.items():
        writer_pages[writer].append((page_id, page_line_counts.get(page_id, 0)))

    return (
        writer_pages,
        page_to_writer,
        page_line_counts,
        inconsistent_pages,
        missing_in_lines,
        extra_in_lines,
        lines_dir_missing,
        lines_dir_mode,
    )


def best_permutation_assign(pages, deficits):
    """
    Assign 3 pages to train/val/test minimizing deficit mismatch.
    pages: list of (page_id, line_count) length 3
    deficits: dict split -> remaining lines target
    """
    splits = ["train", "val", "test"]
    best = None
    best_score = None
    from itertools import permutations

    for perm in permutations(splits, 3):
        score = 0.0
        for (page_id, lc), split in zip(pages, perm):
            score += abs(deficits[split] - lc)
        if best_score is None or score < best_score:
            best_score = score
            best = list(zip(pages, perm))
    return best


def choose_split_by_deficit(deficits, allowed_splits):
    # pick the split with largest positive deficit (or least negative)
    ordered = sorted(
        allowed_splits, key=lambda s: (deficits[s], s), reverse=True
    )
    return ordered[0]


def split_writer_pages(
    writer_pages,
    seed,
    ratios,
    writer_policy="allow_train_only",
    disjoint_mode="page",
    doc_id_re=None,
    doc_map=None,
):
    rng = random.Random(seed)

    if disjoint_mode == "document":
        if doc_map is None and doc_id_re is None:
            raise ValueError(
                "doc_id_re or doc_map must be provided for document-disjoint mode."
            )
        doc_pages = defaultdict(list)
        doc_lines = defaultdict(int)
        writer_docs = defaultdict(set)
        for writer, pages in writer_pages.items():
            for page_id, lc in pages:
                if doc_map is not None:
                    if page_id not in doc_map:
                        raise ValueError(
                            f"Page '{page_id}' missing from documents mapping."
                        )
                    doc_id = doc_map[page_id]
                else:
                    doc_id = extract_document_id(page_id, doc_id_re)
                doc_pages[doc_id].append(page_id)
                doc_lines[doc_id] += lc
                writer_docs[writer].add(doc_id)

        total_lines = sum(doc_lines.values())
        targets = {
            "train": total_lines * ratios["train"],
            "val": total_lines * ratios["val"],
            "test": total_lines * ratios["test"],
        }
        current = {"train": 0, "val": 0, "test": 0}
        doc_split = {}

        def assign_doc(doc_id, split):
            if doc_id in doc_split:
                return
            doc_split[doc_id] = split
            current[split] += doc_lines[doc_id]

        writers = list(writer_docs.keys())
        rng.shuffle(writers)

        for writer in writers:
            docs = list(writer_docs[writer])
            rng.shuffle(docs)
            assigned_splits = {doc_split[d] for d in docs if d in doc_split}
            unassigned = [d for d in docs if d not in doc_split]
            if not unassigned:
                continue

            n_docs = len(docs)

            if n_docs == 1:
                assign_doc(unassigned[0], "train")
                continue

            if n_docs == 2:
                deficits = {k: targets[k] - current[k] for k in current}
                if writer_policy == "allow_train_test_only":
                    preferred = choose_split_by_deficit(deficits, ["test"])
                else:
                    preferred = choose_split_by_deficit(deficits, ["val", "test"])

                best_doc = min(
                    unassigned, key=lambda d: abs(deficits[preferred] - doc_lines[d])
                )
                assign_doc(best_doc, preferred)
                unassigned = [d for d in unassigned if d != best_doc]

                if unassigned:
                    deficits = {k: targets[k] - current[k] for k in current}
                    best_doc = min(
                        unassigned, key=lambda d: abs(deficits["train"] - doc_lines[d])
                    )
                    assign_doc(best_doc, "train")
                continue

            # n_docs >= 3
            missing = [s for s in ["train", "val", "test"] if s not in assigned_splits]
            for split in missing:
                if not unassigned:
                    break
                deficits = {k: targets[k] - current[k] for k in current}
                best_doc = min(
                    unassigned, key=lambda d: abs(deficits[split] - doc_lines[d])
                )
                assign_doc(best_doc, split)
                unassigned = [d for d in unassigned if d != best_doc]

            for doc_id in list(unassigned):
                deficits = {k: targets[k] - current[k] for k in current}
                chosen = choose_split_by_deficit(deficits, ["train", "val", "test"])
                assign_doc(doc_id, chosen)

        # assign any remaining docs (e.g., writers filtered out but docs still present)
        for doc_id in doc_pages:
            if doc_id not in doc_split:
                deficits = {k: targets[k] - current[k] for k in current}
                chosen = choose_split_by_deficit(deficits, ["train", "val", "test"])
                assign_doc(doc_id, chosen)

        split_map = {}
        for doc_id, split in doc_split.items():
            for pid in doc_pages[doc_id]:
                split_map[pid] = split

        return split_map, current, targets

    # page-disjoint (default)
    writer_units = {
        writer: [(page_id, lc, [page_id]) for page_id, lc in pages]
        for writer, pages in writer_pages.items()
    }

    total_lines = sum(sum(lc for _, lc, _ in units) for units in writer_units.values())
    targets = {
        "train": total_lines * ratios["train"],
        "val": total_lines * ratios["val"],
        "test": total_lines * ratios["test"],
    }
    current = {"train": 0, "val": 0, "test": 0}

    split_map = {}

    writers = list(writer_units.keys())
    rng.shuffle(writers)

    for writer in writers:
        units = list(writer_units[writer])
        rng.shuffle(units)
        n_units = len(units)

        if n_units == 1:
            _unit_id, lc, page_ids = units[0]
            for pid in page_ids:
                split_map[pid] = "train"
            current["train"] += lc
            continue

        if n_units == 2:
            deficits = {k: targets[k] - current[k] for k in current}
            if writer_policy == "allow_train_test_only":
                preferred = choose_split_by_deficit(deficits, ["test"])
            else:
                preferred = choose_split_by_deficit(deficits, ["val", "test"])
            # pick the page that best fits the preferred split
            unit_a, unit_b = units
            best_page = min(
                [unit_a, unit_b],
                key=lambda p: abs(deficits[preferred] - p[1]),
            )
            other_page = unit_b if best_page == unit_a else unit_a

            for pid in best_page[2]:
                split_map[pid] = preferred
            current[preferred] += best_page[1]
            for pid in other_page[2]:
                split_map[pid] = "train"
            current["train"] += other_page[1]
            continue

        # n_units >= 3
        deficits = {k: targets[k] - current[k] for k in current}
        seed_pages = units[:3]
        assigned = best_permutation_assign(
            [(pid, lc) for pid, lc, _ in seed_pages], deficits
        )
        used_pages = set()
        for (unit_id, lc), split in assigned:
            unit = next(u for u in seed_pages if u[0] == unit_id)
            for pid in unit[2]:
                split_map[pid] = split
            current[split] += lc
            used_pages.add(unit_id)

        remaining = [u for u in units if u[0] not in used_pages]
        for _unit_id, lc, page_ids in remaining:
            deficits = {k: targets[k] - current[k] for k in current}
            chosen = choose_split_by_deficit(deficits, ["train", "val", "test"])
            for pid in page_ids:
                split_map[pid] = chosen
            current[chosen] += lc

    return split_map, current, targets


def validate_splits(split_map, page_to_writer, writer_policy=None):
    # no duplicate pages by construction; check writer presence
    writer_to_splits = defaultdict(set)
    for page_id, split in split_map.items():
        writer = page_to_writer.get(page_id)
        if writer is None:
            continue
        writer_to_splits[writer].add(split)

    missing_train = [w for w, splits in writer_to_splits.items() if "train" not in splits]
    if missing_train:
        raise ValueError(
            f"Writers present in val/test but missing from train (showing up to 10): {missing_train[:10]}"
        )

    if writer_policy == "require_3way":
        missing_val = [w for w, splits in writer_to_splits.items() if "val" not in splits]
        missing_test = [w for w, splits in writer_to_splits.items() if "test" not in splits]
        if missing_val or missing_test:
            raise ValueError(
                "Writers missing required splits for 3-way policy. "
                f"Missing val (up to 10): {missing_val[:10]} "
                f"Missing test (up to 10): {missing_test[:10]}"
            )


def write_split_csv(path, split_map, page_to_writer, page_line_counts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_id", "writer", "split", "line_count"])
        for page_id, split in split_map.items():
            writer.writerow(
                [
                    page_id,
                    page_to_writer.get(page_id, ""),
                    split,
                    page_line_counts.get(page_id, 0),
                ]
            )


def write_writer_stats(path, rows):
    fieldnames = [
        "seed",
        "writer",
        "pages_total",
        "lines_total",
        "pages_train",
        "pages_val",
        "pages_test",
        "lines_train",
        "lines_val",
        "lines_test",
        "min_lines_page",
        "max_lines_page",
        "splittable_3_way",
        "has_val",
        "has_test",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_seed(seed, split_map, page_to_writer, page_line_counts, ratios):
    total_pages = len(split_map)
    total_lines = sum(page_line_counts.get(pid, 0) for pid in split_map)
    lines_by_split = {"train": 0, "val": 0, "test": 0}
    pages_by_split = {"train": 0, "val": 0, "test": 0}
    writers_by_split = {"train": set(), "val": set(), "test": set()}

    for page_id, split in split_map.items():
        lines_by_split[split] += page_line_counts.get(page_id, 0)
        pages_by_split[split] += 1
        writer = page_to_writer.get(page_id)
        if writer:
            writers_by_split[split].add(writer)

    writer_to_splits = defaultdict(set)
    for page_id, split in split_map.items():
        writer = page_to_writer.get(page_id)
        if writer:
            writer_to_splits[writer].add(split)

    coverage_counts = {
        "train_only": 0,
        "train_val": 0,
        "train_test": 0,
        "train_val_test": 0,
    }
    for splits in writer_to_splits.values():
        if splits == {"train"}:
            coverage_counts["train_only"] += 1
        elif splits == {"train", "val"}:
            coverage_counts["train_val"] += 1
        elif splits == {"train", "test"}:
            coverage_counts["train_test"] += 1
        elif splits == {"train", "val", "test"}:
            coverage_counts["train_val_test"] += 1

    writer_page_counts = defaultdict(int)
    for page_id, writer in page_to_writer.items():
        writer_page_counts[writer] += 1

    w1 = sum(1 for c in writer_page_counts.values() if c == 1)
    w2 = sum(1 for c in writer_page_counts.values() if c == 2)
    w3 = sum(1 for c in writer_page_counts.values() if c >= 3)

    summary = {
        "seed": seed,
        "total_pages": total_pages,
        "total_lines": total_lines,
        "pages_by_split": pages_by_split,
        "lines_by_split": lines_by_split,
        "writers_by_split": {k: len(v) for k, v in writers_by_split.items()},
        "writer_coverage": coverage_counts,
        "writer_page_counts": {"1": w1, "2": w2, "3+": w3},
        "ratios": ratios,
    }
    return summary


def apply_writer_policy(writer_pages, policy, unit_counts=None, unit_label="pages"):
    filtered = {}
    rows = []
    for writer, pages in writer_pages.items():
        pages_total = len(pages)
        lines_total = sum(lc for _, lc in pages)
        units_total = (
            unit_counts[writer]
            if unit_counts is not None and writer in unit_counts
            else pages_total
        )
        effective_policy = "drop_if_lt3" if policy == "require_3way" else policy
        if units_total >= 3:
            include = True
            eligibility = "train_val_test"
            reason = "has >=3 pages"
        elif units_total == 2:
            if effective_policy in {"drop_if_lt2", "drop_if_lt3"}:
                include = False
                eligibility = "insufficient_2_pages"
                reason = "requires >=3 pages for 3-way split"
            elif policy == "allow_train_test_only":
                include = True
                eligibility = "train_test_only"
                reason = "2 pages; val not possible"
            else:
                include = True
                eligibility = "train_plus_val_or_test"
                reason = "2 pages; only one of val/test possible"
        else:
            if effective_policy in {"drop_if_lt2", "drop_if_lt3"}:
                include = False
                eligibility = "insufficient_1_page"
                reason = "requires >=3 pages for 3-way split"
            else:
                include = True
                eligibility = "train_only"
                reason = "single page"

        if include:
            filtered[writer] = pages
        rows.append(
            {
                "writer": writer,
                "pages_total": pages_total,
                "units_total": units_total,
                "units_label": unit_label,
                "lines_total": lines_total,
                "eligibility": eligibility,
                "included": include,
                "reason": reason,
                "policy": policy,
            }
        )
    return filtered, rows


def write_writer_policy_csv(path, rows):
    fieldnames = [
        "writer",
        "pages_total",
        "units_total",
        "units_label",
        "lines_total",
        "eligibility",
        "included",
        "reason",
        "policy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_writer_coverage_csv(path, seed, split_map, page_to_writer, writer_pages):
    writer_to_splits = defaultdict(set)
    for page_id, split in split_map.items():
        writer = page_to_writer.get(page_id)
        if writer:
            writer_to_splits[writer].add(split)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "writer", "splits", "pages_total", "lines_total"])
        for writer_name, splits in sorted(writer_to_splits.items()):
            pages = writer_pages.get(writer_name, [])
            pages_total = len(pages)
            lines_total = sum(lc for _, lc in pages)
            writer.writerow(
                [seed, writer_name, ",".join(sorted(splits)), pages_total, lines_total]
            )


def build_line_txt_index(lines_dir):
    index = defaultdict(list)
    for fn in os.listdir(lines_dir):
        if fn.lower().endswith(".txt"):
            page_id = extract_page_id_from_line_filename(fn)
            index[page_id].append(fn)
    return index


def report_xml_mismatches(xml_dir, lines_dir, page_ids, lines_dir_mode, out_csv, out_md):
    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    if not os.path.isdir(lines_dir):
        raise FileNotFoundError(f"Lines directory not found: {lines_dir}")

    line_txt_index = None
    if lines_dir_mode == "flat":
        line_txt_index = build_line_txt_index(lines_dir)

    rows = []
    counts = defaultdict(int)
    for page_id in sorted(page_ids):
        xml_path = os.path.join(xml_dir, f"{page_id}.xml")
        if not os.path.exists(xml_path):
            # no XML; mark all lines for this page as unmatched
            if lines_dir_mode == "folder":
                page_folder = os.path.join(lines_dir, page_id)
                txt_files = []
                if os.path.isdir(page_folder):
                    txt_files = [
                        os.path.join(page_folder, f)
                        for f in os.listdir(page_folder)
                        if f.lower().endswith(".txt")
                    ]
            else:
                txt_files = [
                    os.path.join(lines_dir, f)
                    for f in line_txt_index.get(page_id, [])
                ]
            for txt_path in txt_files:
                rows.append(
                    {
                        "page_id": page_id,
                        "txt_file": os.path.basename(txt_path),
                        "reason": "missing_xml",
                    }
                )
                counts["missing_xml"] += 1
            continue

        arabic_lines = extract_arabic_handwritten_lines(xml_path)
        if not arabic_lines:
            # XML present but no Arabic handwritten-cursive lines
            if lines_dir_mode == "folder":
                page_folder = os.path.join(lines_dir, page_id)
                txt_files = []
                if os.path.isdir(page_folder):
                    txt_files = [
                        os.path.join(page_folder, f)
                        for f in os.listdir(page_folder)
                        if f.lower().endswith(".txt")
                    ]
            else:
                txt_files = [
                    os.path.join(lines_dir, f)
                    for f in line_txt_index.get(page_id, [])
                ]
            for txt_path in txt_files:
                rows.append(
                    {
                        "page_id": page_id,
                        "txt_file": os.path.basename(txt_path),
                        "reason": "no_arabic_handwritten_lines",
                    }
                )
                counts["no_arabic_handwritten_lines"] += 1
            continue

        arabic_set = set(arabic_lines)
        if lines_dir_mode == "folder":
            page_folder = os.path.join(lines_dir, page_id)
            txt_files = []
            if os.path.isdir(page_folder):
                txt_files = [
                    os.path.join(page_folder, f)
                    for f in os.listdir(page_folder)
                    if f.lower().endswith(".txt")
                ]
        else:
            txt_files = [
                os.path.join(lines_dir, f) for f in line_txt_index.get(page_id, [])
            ]

        for txt_path in txt_files:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read().strip()
            except Exception:
                txt_content = None
            if txt_content in arabic_set:
                continue
            rows.append(
                {
                    "page_id": page_id,
                    "txt_file": os.path.basename(txt_path),
                    "reason": "text_not_matched",
                }
            )
            counts["text_not_matched"] += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_id", "txt_file", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# XML Mismatch Report\n")
        f.write(f"- Total unmatched lines: {len(rows)}\n")
        for reason in sorted(counts.keys()):
            f.write(f"- {reason}: {counts[reason]}\n")


def write_line_stats(
    path_csv,
    path_md,
    writer_pages,
    page_line_counts,
    policy_tag,
    xml_filter_enabled,
    doc_id_re=None,
    doc_map=None,
):
    writer_lines = {}
    writer_page_counts = {}
    for writer, pages in writer_pages.items():
        writer_lines[writer] = sum(lc for _, lc in pages)
        writer_page_counts[writer] = len(pages)

    line_values = list(writer_lines.values())
    page_values = list(writer_page_counts.values())
    total_lines = sum(line_values)
    total_writers = len(writer_lines)

    if total_writers:
        mean_lines = statistics.mean(line_values)
        std_lines = statistics.stdev(line_values) if total_writers > 1 else 0.0
        min_lines = min(line_values)
        max_lines = max(line_values)
        mean_pages = statistics.mean(page_values)
        std_pages = statistics.stdev(page_values) if total_writers > 1 else 0.0
        min_pages = min(page_values)
        max_pages = max(page_values)
    else:
        mean_lines = 0.0
        std_lines = 0.0
        min_lines = 0
        max_lines = 0
        mean_pages = 0.0
        std_pages = 0.0
        min_pages = 0
        max_pages = 0

    min_writer = None
    max_writer = None
    if total_writers:
        min_writer = min(writer_lines, key=writer_lines.get)
        max_writer = max(writer_lines, key=writer_lines.get)

    min_pages_writer = None
    max_pages_writer = None
    if total_writers:
        min_pages_writer = min(writer_page_counts, key=writer_page_counts.get)
        max_pages_writer = max(writer_page_counts, key=writer_page_counts.get)

    writer_doc_counts = None
    doc_stats = None
    if doc_map is not None or doc_id_re is not None:
        writer_doc_counts = {}
        for writer, pages in writer_pages.items():
            doc_ids = set()
            for page_id, _lc in pages:
                if doc_map is not None:
                    if page_id not in doc_map:
                        raise ValueError(
                            f"Page '{page_id}' missing from documents mapping."
                        )
                    doc_ids.add(doc_map[page_id])
                else:
                    doc_ids.add(extract_document_id(page_id, doc_id_re))
            writer_doc_counts[writer] = len(doc_ids)

        doc_values = list(writer_doc_counts.values())
        if total_writers:
            mean_docs = statistics.mean(doc_values)
            std_docs = statistics.stdev(doc_values) if total_writers > 1 else 0.0
            min_docs = min(doc_values)
            max_docs = max(doc_values)
            min_docs_writer = min(writer_doc_counts, key=writer_doc_counts.get)
            max_docs_writer = max(writer_doc_counts, key=writer_doc_counts.get)
        else:
            mean_docs = 0.0
            std_docs = 0.0
            min_docs = 0
            max_docs = 0
            min_docs_writer = None
            max_docs_writer = None

        doc_stats = {
            "mean_docs": mean_docs,
            "std_docs": std_docs,
            "min_docs": min_docs,
            "max_docs": max_docs,
            "min_docs_writer": min_docs_writer,
            "max_docs_writer": max_docs_writer,
        }

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_writers", total_writers])
        writer.writerow(["total_lines", total_lines])
        writer.writerow(["mean_lines_per_writer", f"{mean_lines:.4f}"])
        writer.writerow(["std_lines_per_writer", f"{std_lines:.4f}"])
        writer.writerow(["min_lines_per_writer", int(min_lines)])
        writer.writerow(["max_lines_per_writer", int(max_lines)])
        if min_writer is not None:
            writer.writerow(["min_writer", min_writer])
        if max_writer is not None:
            writer.writerow(["max_writer", max_writer])
        writer.writerow(["mean_pages_per_writer", f"{mean_pages:.4f}"])
        writer.writerow(["std_pages_per_writer", f"{std_pages:.4f}"])
        writer.writerow(["min_pages_per_writer", int(min_pages)])
        writer.writerow(["max_pages_per_writer", int(max_pages)])
        if min_pages_writer is not None:
            writer.writerow(["min_pages_writer", min_pages_writer])
        if max_pages_writer is not None:
            writer.writerow(["max_pages_writer", max_pages_writer])
        if doc_stats is not None:
            writer.writerow(["mean_documents_per_writer", f"{doc_stats['mean_docs']:.4f}"])
            writer.writerow(["std_documents_per_writer", f"{doc_stats['std_docs']:.4f}"])
            writer.writerow(["min_documents_per_writer", int(doc_stats["min_docs"])])
            writer.writerow(["max_documents_per_writer", int(doc_stats["max_docs"])])
            if doc_stats["min_docs_writer"] is not None:
                writer.writerow(["min_documents_writer", doc_stats["min_docs_writer"]])
            if doc_stats["max_docs_writer"] is not None:
                writer.writerow(["max_documents_writer", doc_stats["max_docs_writer"]])

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("# Line Summary Statistics\n")
        f.write(f"- Policy: {policy_tag}\n")
        f.write(f"- XML filter enabled: {xml_filter_enabled}\n")
        f.write(f"- Total writers: {total_writers}\n")
        f.write(f"- Total lines: {total_lines}\n")
        f.write(f"- Mean lines/writer: {mean_lines:.4f}\n")
        f.write(f"- Std lines/writer: {std_lines:.4f}\n")
        f.write(f"- Min lines/writer: {int(min_lines)} ({min_writer})\n")
        f.write(f"- Max lines/writer: {int(max_lines)} ({max_writer})\n")
        f.write(f"- Mean pages/writer: {mean_pages:.4f}\n")
        f.write(f"- Std pages/writer: {std_pages:.4f}\n")
        f.write(f"- Min pages/writer: {int(min_pages)} ({min_pages_writer})\n")
        f.write(f"- Max pages/writer: {int(max_pages)} ({max_pages_writer})\n")
        if doc_stats is not None:
            f.write(f"- Mean documents/writer: {doc_stats['mean_docs']:.4f}\n")
            f.write(f"- Std documents/writer: {doc_stats['std_docs']:.4f}\n")
            f.write(
                f"- Min documents/writer: {int(doc_stats['min_docs'])} ({doc_stats['min_docs_writer']})\n"
            )
            f.write(
                f"- Max documents/writer: {int(doc_stats['max_docs'])} ({doc_stats['max_docs_writer']})\n"
            )


def write_public_vs_csv_csv(path, missing_in_public, extra_in_public):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_id", "status"])
        for page_id in missing_in_public:
            writer.writerow([page_id, "missing_in_public"])
        for page_id in extra_in_public:
            writer.writerow([page_id, "extra_in_public"])


def write_summary_md(
    path,
    summaries,
    missing_in_lines,
    extra_in_lines,
    inconsistent_pages,
    public_pages,
    csv_pages,
    missing_in_public,
    extra_in_public,
    lines_dir,
    lines_dir_missing,
    lines_dir_mode,
    allowlist_count,
    verify_only,
    xml_filter_enabled,
    xml_dir,
    writer_policy,
    writer_policy_rows,
    dropped_pages_count,
    dropped_lines_count,
    disjoint_mode,
    units_label,
):
    lines = []
    lines.append("# Page-Disjoint Split Summary\n")

    lines.append("## Public vs Merged CSV\n")
    if public_pages is None:
        lines.append("- Public directory not found or not provided.\n")
    else:
        lines.append(f"- Pages in merged_writer.csv: {len(csv_pages)}\n")
        lines.append(f"- Pages in public/ (images): {len(public_pages)}\n")
        lines.append(f"- Missing in public/: {len(missing_in_public)}\n")
        lines.append(f"- Extra in public/ (unlabeled): {len(extra_in_public)}\n")
        if missing_in_public:
            lines.append("  - Sample missing: " + ", ".join(missing_in_public[:10]) + "\n")
        if extra_in_public:
            lines.append("  - Sample extra: " + ", ".join(extra_in_public[:10]) + "\n")
    lines.append("\n")

    if verify_only:
        lines.append("**Verify-only mode:** splits were not generated.\n\n")

    lines.append(f"## Writer Eligibility (by {units_label} count)\n")
    total_writers = len(writer_policy_rows)
    included_writers = sum(1 for r in writer_policy_rows if r["included"])
    excluded_writers = total_writers - included_writers
    w_one_unit = sum(1 for r in writer_policy_rows if r.get("units_total", 0) == 1)
    w_two_units = sum(1 for r in writer_policy_rows if r.get("units_total", 0) == 2)
    w_three_plus = sum(1 for r in writer_policy_rows if r.get("units_total", 0) >= 3)
    w_insufficient_1 = w_one_unit
    w_insufficient_2 = w_two_units
    lines.append(f"- Writer policy: {writer_policy}\n")
    lines.append(f"- Disjoint mode: {disjoint_mode}\n")
    lines.append(f"- Total writers: {total_writers}\n")
    lines.append(f"- Included writers (policy): {included_writers}\n")
    lines.append(f"- Excluded writers (policy): {excluded_writers}\n")
    lines.append(f"- Writers with 1 {units_label}: {w_one_unit}\n")
    lines.append(f"- Writers with 2 {units_label}: {w_two_units}\n")
    lines.append(
        f"- Writers with >=3 {units_label} (train/val/test eligible): {w_three_plus}\n"
    )
    lines.append(
        f"- Writers insufficient for 3-way split (1 {units_label}): {w_insufficient_1}\n"
    )
    lines.append(
        f"- Writers insufficient for 3-way split (2 {units_label}): {w_insufficient_2}\n"
    )
    if writer_policy in {"drop_if_lt2", "drop_if_lt3", "require_3way"}:
        lines.append(f"- Dropped pages (policy): {dropped_pages_count}\n")
        lines.append(f"- Dropped lines (policy): {dropped_lines_count}\n")
    lines.append("\n")

    if inconsistent_pages:
        lines.append("## Inconsistent Page-to-Writer Mappings\n")
        lines.append(f"Found {len(inconsistent_pages)} pages with inconsistent writers.\n")
        lines.append("First 10 examples: " + ", ".join(inconsistent_pages[:10]) + "\n")

    lines.append("## Line Images vs Merged CSV\n")
    lines.append(f"- Lines directory: {lines_dir}\n")
    if xml_filter_enabled:
        lines.append(f"- XML filter enabled: {xml_dir}\n")
    if allowlist_count is not None:
        lines.append(f"- Line allowlist entries: {allowlist_count}\n")
    if lines_dir_missing:
        lines.append("Lines directory was not found. Line counts were set to 1 per page.\n")
        lines.append("Splits (if generated) are effectively page-count-based.\n\n")
    else:
        lines.append(f"- Lines directory mode: {lines_dir_mode}\n")
        lines.append(f"- Missing in lines dir: {len(missing_in_lines)}\n")
        lines.append(f"- Extra in lines dir (unlabeled): {len(extra_in_lines)}\n")
        if missing_in_lines:
            lines.append("  - Sample missing: " + ", ".join(missing_in_lines[:10]) + "\n")
        if extra_in_lines:
            lines.append("  - Sample extra: " + ", ".join(extra_in_lines[:10]) + "\n")
        lines.append("\n")

    for summary in summaries:
        seed = summary["seed"]
        lines.append(f"## Seed {seed}\n")
        lines.append(
            f"- Total pages: {summary['total_pages']}\n"
            f"- Total lines: {summary['total_lines']}\n"
        )
        lines.append(
            f"- Writers with 1 page: {summary['writer_page_counts']['1']}\n"
            f"- Writers with 2 pages: {summary['writer_page_counts']['2']}\n"
            f"- Writers with >=3 pages: {summary['writer_page_counts']['3+']}\n"
        )

        lines.append("### Split Counts (Pages / Lines / Ratio)\n")
        for split in ["train", "val", "test"]:
            pages = summary["pages_by_split"][split]
            lines_count = summary["lines_by_split"][split]
            ratio = (
                lines_count / summary["total_lines"]
                if summary["total_lines"] > 0
                else 0.0
            )
            target = summary["ratios"][split]
            lines.append(
                f"- {split}: pages={pages}, lines={lines_count}, ratio={ratio:.4f} (target {target:.2f})\n"
            )

        lines.append("### Writers per Split\n")
        for split in ["train", "val", "test"]:
            lines.append(
                f"- {split}: {summary['writers_by_split'][split]} writers\n"
            )
        lines.append("### Writer Coverage by Split\n")
        lines.append(
            f"- train only: {summary['writer_coverage']['train_only']}\n"
            f"- train+val: {summary['writer_coverage']['train_val']}\n"
            f"- train+test: {summary['writer_coverage']['train_test']}\n"
            f"- train+val+test: {summary['writer_coverage']['train_val_test']}\n"
        )
        lines.append("\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate page-disjoint splits and stats.")
    parser.add_argument(
        "--csv",
        default="manual_labeling/merged_writer.csv",
        help="Path to merged_writer.csv",
    )
    parser.add_argument(
        "--lines-dir",
        default="Lines",
        help="Directory containing page folders of line images",
    )
    parser.add_argument(
        "--line-allowlist",
        default=None,
        help="Optional text file listing allowed line image filenames (one per line).",
    )
    parser.add_argument(
        "--filter-xml",
        action="store_true",
        help="Filter line counts to Arabic handwritten-cursive lines based on PAGE XML.",
    )
    parser.add_argument(
        "--xml-dir",
        default="public",
        help="Directory containing PAGE XML files (used with --filter-xml).",
    )
    parser.add_argument(
        "--public-dir",
        default="public",
        help="Directory containing public page images for consistency checks",
    )
    parser.add_argument(
        "--writer-policy",
        choices=WRITER_POLICY_CHOICES,
        default="allow_train_only",
        help=(
            "Policy for writers with insufficient pages: "
            "allow_train_only (default), drop_if_lt2, drop_if_lt3, require_3way, "
            "allow_train_test_only"
        ),
    )
    parser.add_argument(
        "--disjoint-mode",
        choices=DISJOINT_MODE_CHOICES,
        default="page",
        help="Disjoint mode: page (default) or document.",
    )
    parser.add_argument(
        "--documents-dir",
        default=None,
        help=(
            "Directory containing document subfolders with page images. "
            "If provided with --disjoint-mode document, this mapping is used "
            "instead of the regex heuristic."
        ),
    )
    parser.add_argument(
        "--doc-id-regex",
        default=r"^(.*?)(?:[_\-\s]+\d+)$",
        help=(
            "Regex to derive document id from page id in document mode. "
            "The first capture group is used as the document id."
        ),
    )
    parser.add_argument(
        "--report-xml-mismatches",
        action="store_true",
        help="Write a report of line .txt files that do not match Arabic handwritten-cursive XML lines.",
    )
    parser.add_argument(
        "--allow-missing-lines",
        action="store_true",
        help="Allow missing Lines/ by using 1 line per page (page-count split).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify public vs CSV and write stats; do not generate splits.",
    )
    parser.add_argument(
        "--seeds",
        default="42,570,1073",
        help="Comma-separated list of seeds",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--out-splits", default="splits", help="Output splits directory")
    parser.add_argument("--out-stats", default="stats", help="Output stats directory")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path) and os.path.exists("merged_writer.csv"):
        csv_path = "merged_writer.csv"

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    lines_dir = args.lines_dir
    if not os.path.isdir(lines_dir) and os.path.isdir("public_line_images"):
        lines_dir = "public_line_images"

    os.makedirs(args.out_splits, exist_ok=True)
    os.makedirs(args.out_stats, exist_ok=True)

    public_pages = load_public_pages(args.public_dir)

    allow_missing_lines = args.allow_missing_lines or args.verify_only
    lines_dir_mode = detect_lines_dir_mode(lines_dir)
    allowlist = None

    if args.filter_xml:
        if lines_dir_mode == "missing":
            raise FileNotFoundError(
                f"Lines directory not found: {lines_dir}. Cannot apply XML filtering."
            )
        xml_dir = args.xml_dir
        if not os.path.isdir(xml_dir):
            raise FileNotFoundError(f"XML directory not found: {xml_dir}")
        allowlist = build_allowlist_from_xml(xml_dir, lines_dir, lines_dir_mode)

    if args.line_allowlist:
        file_allowlist = set()
        with open(args.line_allowlist, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                file_allowlist.add(name)
                # also store basename without extension for convenience
                file_allowlist.add(os.path.splitext(name)[0])
        if allowlist is None:
            allowlist = file_allowlist
        else:
            allowlist = allowlist & file_allowlist
    (
        writer_pages,
        page_to_writer,
        page_line_counts,
        inconsistent_pages,
        missing_in_lines,
        extra_in_lines,
        lines_dir_missing,
        lines_dir_mode,
    ) = load_writer_pages(
        csv_path,
        lines_dir,
        allow_missing_lines=allow_missing_lines,
        allowlist=allowlist,
        lines_dir_mode=lines_dir_mode if lines_dir_mode != "missing" else None,
    )

    doc_id_re = None
    doc_map = None
    unit_label = "pages"
    unit_counts = None
    if args.disjoint_mode == "document":
        unit_label = "documents"
        if args.documents_dir:
            doc_map = build_doc_map_from_dir(args.documents_dir)
        else:
            doc_id_re = re.compile(args.doc_id_regex)

        unit_counts = defaultdict(int)
        for writer, pages in writer_pages.items():
            doc_ids = set()
            for page_id, _ in pages:
                if doc_map is not None:
                    if page_id not in doc_map:
                        raise ValueError(
                            f"Page '{page_id}' missing from documents mapping. "
                            "Check --documents-dir."
                        )
                    doc_ids.add(doc_map[page_id])
                else:
                    doc_ids.add(extract_document_id(page_id, doc_id_re))
            unit_counts[writer] = len(doc_ids)

    writer_pages_policy, writer_policy_rows = apply_writer_policy(
        writer_pages,
        args.writer_policy,
        unit_counts=unit_counts,
        unit_label=unit_label,
    )
    included_pages = {
        page_id for pages in writer_pages_policy.values() for page_id, _ in pages
    }
    page_to_writer_policy = {
        pid: writer for pid, writer in page_to_writer.items() if pid in included_pages
    }
    page_line_counts_policy = {
        pid: lc for pid, lc in page_line_counts.items() if pid in included_pages
    }
    dropped_pages_count = len(page_to_writer) - len(page_to_writer_policy)
    dropped_lines_count = sum(
        lc for pid, lc in page_line_counts.items() if pid not in included_pages
    )

    csv_pages = set(page_to_writer.keys())
    if public_pages is not None:
        missing_in_public = sorted(csv_pages - public_pages)
        extra_in_public = sorted(public_pages - csv_pages)
        public_vs_csv_path = os.path.join(args.out_stats, "public_vs_merged_writer.csv")
        write_public_vs_csv_csv(public_vs_csv_path, missing_in_public, extra_in_public)
    else:
        missing_in_public = []
        extra_in_public = []
    lines_vs_csv_path = os.path.join(args.out_stats, "lines_vs_merged_writer.csv")
    write_public_vs_csv_csv(lines_vs_csv_path, missing_in_lines, extra_in_lines)

    writer_stats_rows = []
    summaries = []

    policy_tag = "" if args.writer_policy == "allow_train_only" else f"_{args.writer_policy}"
    disjoint_tag = ""
    if args.disjoint_mode == "document":
        disjoint_tag = "_document"

    if not args.verify_only:
        for seed in parse_seeds(args.seeds):
            split_map, current, targets = split_writer_pages(
                writer_pages_policy,
                seed,
                ratios,
                writer_policy=args.writer_policy,
                disjoint_mode=args.disjoint_mode,
                doc_id_re=doc_id_re,
                doc_map=doc_map,
            )
            validate_splits(split_map, page_to_writer_policy, writer_policy=args.writer_policy)

            split_path = os.path.join(
                args.out_splits,
                f"page_disjoint{disjoint_tag}{policy_tag}_seed_{seed}.csv",
            )
            write_split_csv(
                split_path, split_map, page_to_writer_policy, page_line_counts_policy
            )

            # writer stats
            for writer, pages in writer_pages_policy.items():
                pages_total = len(pages)
                lines_total = sum(lc for _, lc in pages)
                lines_per_page = [lc for _, lc in pages]
                min_lines = min(lines_per_page) if lines_per_page else 0
                max_lines = max(lines_per_page) if lines_per_page else 0

                pages_train = pages_val = pages_test = 0
                lines_train = lines_val = lines_test = 0
                for page_id, lc in pages:
                    split = split_map.get(page_id)
                    if split == "train":
                        pages_train += 1
                        lines_train += lc
                    elif split == "val":
                        pages_val += 1
                        lines_val += lc
                    elif split == "test":
                        pages_test += 1
                        lines_test += lc

                writer_stats_rows.append(
                    {
                        "seed": seed,
                        "writer": writer,
                        "pages_total": pages_total,
                        "lines_total": lines_total,
                        "pages_train": pages_train,
                        "pages_val": pages_val,
                        "pages_test": pages_test,
                        "lines_train": lines_train,
                        "lines_val": lines_val,
                        "lines_test": lines_test,
                        "min_lines_page": min_lines,
                        "max_lines_page": max_lines,
                        "splittable_3_way": pages_total >= 3,
                        "has_val": pages_val > 0,
                        "has_test": pages_test > 0,
                    }
                )

            summaries.append(
                summarize_seed(
                    seed, split_map, page_to_writer_policy, page_line_counts_policy, ratios
                )
            )

            coverage_path = os.path.join(
                args.out_stats,
                f"page_disjoint{disjoint_tag}{policy_tag}_writer_coverage_seed_{seed}.csv",
            )
            write_writer_coverage_csv(
                coverage_path, seed, split_map, page_to_writer_policy, writer_pages_policy
            )

    stats_csv = os.path.join(
        args.out_stats, f"page_disjoint{disjoint_tag}{policy_tag}_writer_stats.csv"
    )
    stats_md = os.path.join(
        args.out_stats, f"page_disjoint{disjoint_tag}{policy_tag}_summary.md"
    )
    policy_csv = os.path.join(
        args.out_stats, f"page_disjoint{disjoint_tag}{policy_tag}_writer_eligibility.csv"
    )
    line_stats_csv = os.path.join(
        args.out_stats, f"page_disjoint{disjoint_tag}{policy_tag}_line_stats.csv"
    )
    line_stats_md = os.path.join(
        args.out_stats, f"page_disjoint{disjoint_tag}{policy_tag}_line_stats.md"
    )
    wrote_stats_csv = False
    if writer_stats_rows:
        write_writer_stats(stats_csv, writer_stats_rows)
        wrote_stats_csv = True
    write_writer_policy_csv(policy_csv, writer_policy_rows)
    write_line_stats(
        line_stats_csv,
        line_stats_md,
        writer_pages_policy,
        page_line_counts_policy,
        args.writer_policy,
        args.filter_xml,
        doc_id_re=doc_id_re,
        doc_map=doc_map,
    )
    write_summary_md(
        stats_md,
        summaries,
        missing_in_lines,
        extra_in_lines,
        inconsistent_pages,
        public_pages,
        csv_pages,
        missing_in_public,
        extra_in_public,
        lines_dir,
        lines_dir_missing,
        lines_dir_mode,
        len(allowlist) if allowlist is not None else None,
        args.verify_only,
        args.filter_xml,
        args.xml_dir,
        args.writer_policy,
        writer_policy_rows,
        dropped_pages_count,
        dropped_lines_count,
        args.disjoint_mode,
        unit_label,
    )

    if args.report_xml_mismatches:
        xml_report_csv = os.path.join(
            args.out_stats, f"page_disjoint{disjoint_tag}_xml_mismatches{policy_tag}.csv"
        )
        xml_report_md = os.path.join(
            args.out_stats, f"page_disjoint{disjoint_tag}_xml_mismatches{policy_tag}.md"
        )
        report_xml_mismatches(
            args.xml_dir,
            lines_dir,
            csv_pages,
            lines_dir_mode,
            xml_report_csv,
            xml_report_md,
        )

    if not args.verify_only:
        print(f"Wrote splits to: {args.out_splits}")
    if wrote_stats_csv:
        print(f"Wrote stats to: {stats_csv}")
    print(f"Wrote summary to: {stats_md}")


if __name__ == "__main__":
    main()
