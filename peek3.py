# save as peek3.py, run: python peek3.py --config configs/remarl_ablation.yaml
import argparse
import yaml
from sim.scenario_gen import ScenarioGenerator
from mare.agents.factory import AgentFactory
from mare.rl_adapter import MARERLAgent
from eval.run_negotiator_ablation import build_schedules, make_workspace

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/remarl_train_fast.yaml")
args = ap.parse_args()

print(f"Loading config: {args.config}")
config = yaml.safe_load(open(args.config))
print(f"Configured documenter max_tokens: {config['llm']['agent_max_tokens']['documenter']}")

gen = ScenarioGenerator(config["env"]["scenario_dir"], seed=42)
raw = AgentFactory.create_all_agents_from_config(config)

# Confirm the actual agent instance really got that value
doc_inner = getattr(raw["documenter"], "config", None)
print(f"DocumenterAgent.config.max_tokens (actual): {doc_inner.max_tokens if doc_inner else 'N/A'}")

agents = {r: MARERLAgent(a) for r, a in raw.items()}
sc = gen.sample()

with_neg, _ = build_schedules()
ws = make_workspace(sc)
for role, action in with_neg:
    agents[role].perform_action(action, ws)

srs = ws.get("srs_document", "")
print(f"\nFull SRS length: {len(srs)} chars\n")

idx = srs.find("3.2")
if idx == -1:
    idx = srs.lower().find("business rules")
if idx != -1:
    print("=== SECTION 3.2 (Business Rules) ===")
    print(srs[idx:idx+800])
else:
    print("!! Section 3.2 / Business Rules not found in document at all")

print("\n=== All section headers found in the document ===")
import re
for line in srs.split("\n"):
    if re.match(r'^#{1,4}\s|^\d+\.\d*\s', line.strip()):
        print(f"  {line.strip()}")

print("\n=== Resolution word search across FULL document ===")
for w in ["agreed", "resolved", "compromise", "unless"]:
    print(f"  '{w}': {srs.lower().count(w)} occurrence(s)")