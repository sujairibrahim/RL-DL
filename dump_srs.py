# save as dump_srs.py, run: python dump_srs.py --config configs/remarl_ablation.yaml
import argparse
import yaml
from sim.scenario_gen import ScenarioGenerator
from mare.agents.factory import AgentFactory
from mare.rl_adapter import MARERLAgent
from eval.run_negotiator_ablation import build_schedules, make_workspace

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/remarl_ablation.yaml")
ap.add_argument("--out", default="debug_srs_1.md")
args = ap.parse_args()

config = yaml.safe_load(open(args.config))
gen = ScenarioGenerator(config["env"]["scenario_dir"], seed=42)
raw = AgentFactory.create_all_agents_from_config(config)
agents = {r: MARERLAgent(a) for r, a in raw.items()}
sc = gen.sample()

with_neg, _ = build_schedules()
ws = make_workspace(sc)
for role, action in with_neg:
    agents[role].perform_action(action, ws)

with open(args.out, "w") as f:
    f.write("# DEBUG: req_draft (Negotiator's raw output, full)\n\n")
    f.write(ws.get("req_draft", ""))
    f.write("\n\n---\n\n# DEBUG: error_report (full)\n\n")
    f.write(ws.get("error_report", ""))
    f.write("\n\n---\n\n# FINAL SRS (full)\n\n")
    f.write(ws.get("srs_document", ""))

print(f"Wrote {args.out} ({len(ws.get('srs_document',''))} chars in the SRS section)")