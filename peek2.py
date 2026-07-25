# save as peek2.py, run: python peek2.py
import yaml
from sim.scenario_gen import ScenarioGenerator
from mare.agents.factory import AgentFactory
from mare.rl_adapter import MARERLAgent
from eval.run_negotiator_ablation import build_schedules, make_workspace

config = yaml.safe_load(open("configs/remarl_train_fast.yaml"))
gen = ScenarioGenerator(config["env"]["scenario_dir"], seed=42)
raw = AgentFactory.create_all_agents_from_config(config)
agents = {r: MARERLAgent(a) for r, a in raw.items()}
sc = gen.sample()

with_neg, _ = build_schedules()
ws = make_workspace(sc)
for role, action in with_neg:
    agents[role].perform_action(action, ws)

srs = ws.get("srs_document", "")
print(f"Full SRS length: {len(srs)} chars\n")

# Find and print the Business Rules section specifically
idx = srs.find("3.2")
if idx == -1:
    idx = srs.lower().find("business rules")
if idx != -1:
    print("=== SECTION 3.2 (Business Rules) ===")
    print(srs[idx:idx+800])
else:
    print("!! Section 3.2 / Business Rules not found in document at all")

print("\n=== Resolution word search across FULL document ===")
for w in ["agreed", "resolved", "compromise", "unless"]:
    count = srs.lower().count(w)
    print(f"  '{w}': {count} occurrence(s)")