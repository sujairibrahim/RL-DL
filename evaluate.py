"""
remarl/evaluate.py
------------------
Evaluate a trained REMARL checkpoint against vanilla MARE baseline.

Usage:
    python evaluate.py --checkpoint data/checkpoints/collector_final
    python evaluate.py --checkpoint data/checkpoints/collector_final --n 100
    python evaluate.py --checkpoint data/checkpoints/collector_final --domain patient_portal
    python evaluate.py --checkpoint data/checkpoints/modeler_final --role modeler
    python evaluate.py --checkpoint data/checkpoints/checker_final  --role checker

FIX [1]: Added --role argument so modeler/checker checkpoints can be evaluated
         without editing source. Previously hardcoded to 'collector' only.
FIX [2]: Renamed --n to --n_eval to match benchmark.py's internal parameter name.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained REMARL policy against the MARE baseline."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SB3 PPO checkpoint (with or without .zip extension)",
    )
    parser.add_argument(
        "--config", default="configs/remarl_config.yaml",
        help="YAML config file used during training",
    )
    parser.add_argument(
        "--n_eval", type=int, default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Restrict evaluation to a single domain (e.g. patient_portal)",
    )
    # FIX [1]: was missing entirely — only collector could be evaluated
    parser.add_argument(
        "--role", default="collector",
        choices=["collector", "modeler", "checker"],
        help="Agent role whose checkpoint is being evaluated",
    )
    args = parser.parse_args()

    from eval.benchmark import benchmark
    benchmark(args.config, args.checkpoint, args.n_eval, args.domain, args.role)


if __name__ == "__main__":
    main()