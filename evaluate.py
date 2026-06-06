"""
remarl/evaluate.py
------------------
Evaluate a trained REMARL checkpoint against vanilla MARE baseline.

Usage:
    python evaluate.py --checkpoint data/checkpoints/collector_final
    python evaluate.py --checkpoint data/checkpoints/collector_final --n 100
    python evaluate.py --checkpoint data/checkpoints/collector_final --domain patient_portal
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SB3 PPO checkpoint (no .zip extension needed)")
    parser.add_argument("--config",  default="configs/remarl_config.yaml")
    parser.add_argument("--n",       type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--domain",  default=None,
                        help="Restrict evaluation to one domain")
    args = parser.parse_args()

    from eval.benchmark import benchmark
    benchmark(args.config, args.checkpoint, args.n, args.domain)


if __name__ == "__main__":
    main()
