#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict

SEEDS = ["42", "570", "1073"]

BASES = [
    "frozen",
    "from_scratch",
    "finetuned_all",
    "finetuned_last_layer",
    "finetuned_last_5",
    "finetuned_last_10",
    "finetuned_last_25",
]
ATTNS = ["no_attention", "attention"]
EXPECTED_KEYS = [f"{b}|{a}" for b in BASES for a in ATTNS]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def config_key(name: str) -> str | None:
    l = norm(name)

    if "no attention" in l:
        attn = "no_attention"
    elif "attention" in l:
        attn = "attention"
    else:
        attn = None

    base = None
    if "from scratch" in l:
        base = "from_scratch"
    elif "frozen" in l:
        base = "frozen"
    elif "all imagenet" in l:
        base = "finetuned_all"
    elif "last 25 layers" in l:
        base = "finetuned_last_25"
    elif "last 10 layers" in l:
        base = "finetuned_last_10"
    elif "last 5 layers" in l:
        base = "finetuned_last_5"
    elif "last layer" in l:
        base = "finetuned_last_layer"

    if base is None or attn is None:
        return None
    return f"{base}|{attn}"


def key_label(key: str) -> str:
    base, attn = key.split("|", 1)
    base_map = {
        "frozen": "Frozen",
        "from_scratch": "From Scratch",
        "finetuned_all": "Finetuned All ImageNet",
        "finetuned_last_layer": "Finetuned ImageNet Last Layer",
        "finetuned_last_5": "Finetuned ImageNet Last 5 Layers",
        "finetuned_last_10": "Finetuned ImageNet Last 10 Layers",
        "finetuned_last_25": "Finetuned ImageNet Last 25 Layers",
    }
    attn_map = {
        "no_attention": "No Attention",
        "attention": "Attention",
    }
    return f"{base_map.get(base, base)} + {attn_map.get(attn, attn)}"


def seed_set_from_files(files):
    seeds = set()
    for f in files:
        for s in re.findall(r"(?<!\d)(42|570|1073)(?!\d)", f):
            seeds.add(s)
    return seeds


def list_dirs(path):
    if not os.path.isdir(path):
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def collect_models_rows(root_dir):
    rows = []
    models_dir = os.path.join(root_dir, "models")
    for arch in sorted(list_dirs(models_dir)):
        arch_path = os.path.join(models_dir, arch)
        cfg_dirs = list_dirs(arch_path)
        key_to_dirs = defaultdict(list)
        for d in cfg_dirs:
            k = config_key(d)
            if k:
                key_to_dirs[k].append(d)
            else:
                key_to_dirs[None].append(d)

        for key in EXPECTED_KEYS:
            if key not in key_to_dirs:
                expected_dir = f"{arch} + {key_label(key)}"
                rows.append(
                    {
                        "section": "models",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "missing_config",
                        "details": "",
                        "paths": os.path.join("models", arch, expected_dir),
                    }
                )

        for d in key_to_dirs.get(None, []):
            rows.append(
                {
                    "section": "models",
                    "architecture": arch,
                    "config_key": "",
                    "config_label": "",
                    "issue": "unparsed_dir",
                    "details": "",
                    "paths": os.path.join("models", arch, d),
                }
            )

        for key, dirs in key_to_dirs.items():
            if key is None:
                continue
            files = []
            for d in dirs:
                p = os.path.join(arch_path, d)
                for fn in os.listdir(p):
                    if fn.lower().endswith(".py") or fn.lower().endswith(".ipynb"):
                        files.append(fn)
            seeds = seed_set_from_files(files)
            missing = sorted(set(SEEDS) - seeds)
            extra = sorted(seeds - set(SEEDS))
            if missing:
                rows.append(
                    {
                        "section": "models",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "missing_seeds",
                        "details": ",".join(missing),
                        "paths": ";".join(
                            [os.path.join("models", arch, d) for d in dirs]
                        ),
                    }
                )
            if extra:
                rows.append(
                    {
                        "section": "models",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "extra_seeds",
                        "details": ",".join(extra),
                        "paths": ";".join(
                            [os.path.join("models", arch, d) for d in dirs]
                        ),
                    }
                )

    return rows


def collect_results_rows(root_dir):
    rows = []
    results_dir = os.path.join(root_dir, "Results")
    for arch in sorted(list_dirs(results_dir)):
        arch_path = os.path.join(results_dir, arch)
        cfg_dirs = list_dirs(arch_path)
        key_to_dirs = defaultdict(list)
        for d in cfg_dirs:
            k = config_key(d)
            if k:
                key_to_dirs[k].append(d)
            else:
                key_to_dirs[None].append(d)

        for key in EXPECTED_KEYS:
            if key not in key_to_dirs:
                expected_dir = f"{arch} + {key_label(key)}"
                rows.append(
                    {
                        "section": "results",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "missing_config",
                        "details": "",
                        "paths": os.path.join("Results", arch, expected_dir),
                    }
                )

        for d in key_to_dirs.get(None, []):
            rows.append(
                {
                    "section": "results",
                    "architecture": arch,
                    "config_key": "",
                    "config_label": "",
                    "issue": "unparsed_dir",
                    "details": "",
                    "paths": os.path.join("Results", arch, d),
                }
            )

        for key, dirs in key_to_dirs.items():
            if key is None:
                continue
            csv_files = []
            json_files = []
            for d in dirs:
                p = os.path.join(arch_path, d)
                for fn in os.listdir(p):
                    if fn.lower().endswith(".csv"):
                        csv_files.append(fn)
                    elif fn.lower().endswith(".json"):
                        json_files.append(fn)
            seeds_csv = seed_set_from_files(csv_files)
            seeds_json = seed_set_from_files(json_files)
            missing_csv = sorted(set(SEEDS) - seeds_csv)
            missing_json = sorted(set(SEEDS) - seeds_json)
            extra_csv = sorted(seeds_csv - set(SEEDS))
            extra_json = sorted(seeds_json - set(SEEDS))
            if missing_csv:
                rows.append(
                    {
                        "section": "results",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "missing_csv_seeds",
                        "details": ",".join(missing_csv),
                        "paths": ";".join(
                            [os.path.join("Results", arch, d) for d in dirs]
                        ),
                    }
                )
            if missing_json:
                rows.append(
                    {
                        "section": "results",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "missing_json_seeds",
                        "details": ",".join(missing_json),
                        "paths": ";".join(
                            [os.path.join("Results", arch, d) for d in dirs]
                        ),
                    }
                )
            if extra_csv:
                rows.append(
                    {
                        "section": "results",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "extra_csv_seeds",
                        "details": ",".join(extra_csv),
                        "paths": ";".join(
                            [os.path.join("Results", arch, d) for d in dirs]
                        ),
                    }
                )
            if extra_json:
                rows.append(
                    {
                        "section": "results",
                        "architecture": arch,
                        "config_key": key,
                        "config_label": key_label(key),
                        "issue": "extra_json_seeds",
                        "details": ",".join(extra_json),
                        "paths": ";".join(
                            [os.path.join("Results", arch, d) for d in dirs]
                        ),
                    }
                )

    return rows


def write_csv(path, rows):
    fieldnames = [
        "section",
        "architecture",
        "config_key",
        "config_label",
        "issue",
        "details",
        "paths",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Detect missing config folders and seed files in models/ and Results/."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default="missing_report.csv",
        help="Output CSV path (default: missing_report.csv)",
    )
    args = parser.parse_args()

    rows = []
    rows.extend(collect_models_rows(args.root))
    rows.extend(collect_results_rows(args.root))

    write_csv(args.out, rows)

    issue_counts = defaultdict(int)
    for r in rows:
        issue_counts[r["issue"]] += 1
    total = sum(issue_counts.values())
    print(f"Wrote {args.out} with {total} issues.")
    if total:
        for issue in sorted(issue_counts):
            print(f"{issue}: {issue_counts[issue]}")


if __name__ == "__main__":
    main()
