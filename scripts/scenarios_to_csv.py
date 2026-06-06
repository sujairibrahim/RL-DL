"""
scripts/scenarios_to_csv.py
----------------------------
Convert all_scenarios.json (or all_scenarios_expanded.json) to CSV files.

Produces four normalised CSVs:
  scenarios.csv       — one row per scenario (main fields)
  requirements.csv    — one row per requirement (functional & hidden/visible)
  stakeholders.csv    — one row per stakeholder
  conflicts.csv       — one row per conflict

Usage:
    python scripts/scenarios_to_csv.py
    python scripts/scenarios_to_csv.py --input data/scenarios/all_scenarios_expanded.json
    python scripts/scenarios_to_csv.py --output-dir exports/
    python scripts/scenarios_to_csv.py --flat          # single wide CSV instead of normalised
"""

import json
import csv
import argparse
import pathlib


def load_scenarios(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def write_scenarios(scenarios: list, out_dir: pathlib.Path):
    path = out_dir / "scenarios.csv"
    fieldnames = [
        "scenario_id", "domain", "difficulty", "rough_idea",
        "num_ground_truth_reqs", "num_hidden_reqs", "num_visible_reqs",
        "num_nfr", "num_stakeholders", "num_domain_entities", "num_conflicts",
        "domain_entities", "nfr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scenarios:
            writer.writerow({
                "scenario_id": s["scenario_id"],
                "domain": s["domain"],
                "difficulty": s["difficulty"],
                "rough_idea": s["rough_idea"],
                "num_ground_truth_reqs": len(s["ground_truth_reqs"]),
                "num_hidden_reqs": len(s["hidden_reqs"]),
                "num_visible_reqs": len(s["visible_reqs"]),
                "num_nfr": len(s["nfr"]),
                "num_stakeholders": len(s["stakeholders"]),
                "num_domain_entities": len(s["domain_entities"]),
                "num_conflicts": len(s["conflicts"]),
                "domain_entities": " | ".join(s["domain_entities"]),
                "nfr": " | ".join(s["nfr"]),
            })
    print(f"  Wrote {len(scenarios)} rows → {path}")


def write_requirements(scenarios: list, out_dir: pathlib.Path):
    path = out_dir / "requirements.csv"
    fieldnames = ["scenario_id", "domain", "req_type", "req_index", "requirement"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        total = 0
        for s in scenarios:
            sid, domain = s["scenario_id"], s["domain"]
            hidden_set = set(s["hidden_reqs"])
            for i, req in enumerate(s["ground_truth_reqs"]):
                req_type = "hidden" if req in hidden_set else "visible"
                writer.writerow({
                    "scenario_id": sid,
                    "domain": domain,
                    "req_type": req_type,
                    "req_index": i,
                    "requirement": req,
                })
                total += 1
    print(f"  Wrote {total} rows → {path}")


def write_stakeholders(scenarios: list, out_dir: pathlib.Path):
    path = out_dir / "stakeholders.csv"
    fieldnames = [
        "scenario_id", "domain", "stakeholder_index",
        "name", "role", "primary_interest", "secondary_interest", "conflict_with",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        total = 0
        for s in scenarios:
            for i, sh in enumerate(s["stakeholders"]):
                writer.writerow({
                    "scenario_id": s["scenario_id"],
                    "domain": s["domain"],
                    "stakeholder_index": i,
                    "name": sh.get("name", ""),
                    "role": sh.get("role", ""),
                    "primary_interest": sh.get("primary_interest", ""),
                    "secondary_interest": sh.get("secondary_interest", ""),
                    "conflict_with": sh.get("conflict_with") or "",
                })
                total += 1
    print(f"  Wrote {total} rows → {path}")


def write_conflicts(scenarios: list, out_dir: pathlib.Path):
    path = out_dir / "conflicts.csv"
    fieldnames = ["scenario_id", "domain", "conflict_index", "req_a", "req_b", "conflict_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        total = 0
        for s in scenarios:
            for i, c in enumerate(s["conflicts"]):
                writer.writerow({
                    "scenario_id": s["scenario_id"],
                    "domain": s["domain"],
                    "conflict_index": i,
                    "req_a": c.get("req_a", ""),
                    "req_b": c.get("req_b", ""),
                    "conflict_type": c.get("type", ""),
                })
                total += 1
    print(f"  Wrote {total} rows → {path}")


def write_flat(scenarios: list, out_dir: pathlib.Path):
    path = out_dir / "scenarios_flat.csv"
    fieldnames = [
        "scenario_id", "domain", "difficulty", "rough_idea",
        "ground_truth_reqs", "hidden_reqs", "visible_reqs",
        "nfr", "domain_entities",
        "stakeholders",
        "conflicts",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scenarios:
            stakeholders_str = " | ".join(
                f"{sh.get('name','')}({sh.get('role','')})"
                for sh in s["stakeholders"]
            )
            conflicts_str = " | ".join(
                f"{c.get('type','')}: [{c.get('req_a','')}] vs [{c.get('req_b','')}]"
                for c in s["conflicts"]
            )
            writer.writerow({
                "scenario_id": s["scenario_id"],
                "domain": s["domain"],
                "difficulty": s["difficulty"],
                "rough_idea": s["rough_idea"],
                "ground_truth_reqs": " | ".join(s["ground_truth_reqs"]),
                "hidden_reqs": " | ".join(s["hidden_reqs"]),
                "visible_reqs": " | ".join(s["visible_reqs"]),
                "nfr": " | ".join(s["nfr"]),
                "domain_entities": " | ".join(s["domain_entities"]),
                "stakeholders": stakeholders_str,
                "conflicts": conflicts_str,
            })
    print(f"  Wrote {len(scenarios)} rows → {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert scenarios JSON to CSV")
    parser.add_argument(
        "--input",
        default="data/scenarios/all_scenarios.json",
        help="Path to the scenarios JSON file (default: data/scenarios/all_scenarios.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/scenarios/csv",
        help="Directory to write CSV files into (default: data/scenarios/csv)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write a single wide CSV instead of normalised tables",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run `python sim/scenario_gen.py` first to generate the scenarios cache.")
        raise SystemExit(1)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scenarios from {input_path} ...")
    scenarios = load_scenarios(input_path)
    print(f"Found {len(scenarios)} scenarios.\n")

    if args.flat:
        write_flat(scenarios, out_dir)
    else:
        write_scenarios(scenarios, out_dir)
        write_requirements(scenarios, out_dir)
        write_stakeholders(scenarios, out_dir)
        write_conflicts(scenarios, out_dir)

    print(f"\nDone. CSVs written to {out_dir}/")


if __name__ == "__main__":
    main()
