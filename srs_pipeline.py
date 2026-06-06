#!/usr/bin/env python3
"""
remarl/srs_pipeline.py
======================
Human-in-the-Loop SRS Generation Pipeline

Runs the full MARE multi-agent pipeline interactively. After each stage the
human can type natural-language feedback to improve the agent output, or
press Enter to accept it and move on.

At the end it writes a fully structured IEEE 830 SRS document to output/ and
displays a quality dashboard scored by RewardEngine + Oracle.

╔══════════════════════════════════════════════════════════════════╗
║  REMARL has two separate operating modes:                        ║
║                                                                  ║
║  1. HITL PIPELINE (this file)                                    ║
║     python srs_pipeline.py                                       ║
║     Interactive human-guided SRS generation using local Ollama.  ║
║     Uses: mare/agents/, rl/reward.py, sim/oracle.py,             ║
║           rl/memory.py                                           ║
║                                                                  ║
║  2. RL TRAINING + EVALUATION                                     ║
║     python train.py --role collector --episodes 2000             ║
║     python evaluate.py --checkpoint data/checkpoints/collector_final ║
║     Trains PPO policies on synthetic scenarios.                  ║
║     Uses: sim/re_env.py, sim/scenario_gen.py, sim/oracle.py,     ║
║           rl/policy.py, rl/trainer.py, eval/benchmark.py,        ║
║           eval/metrics.py                                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python srs_pipeline.py
    python srs_pipeline.py --config configs/remarl_config.yaml
    python srs_pipeline.py --resume output/srs_session_<timestamp>.json

Requirements:
    - NVIDIA API key in .env:  NVIDIA_API_KEY=nvapi-...
    - Get a key at: https://build.nvidia.com
    - Or set provider: "ollama" in config and run:  ollama serve
"""

import argparse
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Load environment variables from .env (NVIDIA_API_KEY etc.) ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Rich terminal UI ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

MAX_FEEDBACK_RETRIES = 3   # max times a human can give feedback per stage


# ═══════════════════════════════════════════════════════════════════════════════
#  RL layer helpers (lazy-loaded so missing packages don't crash import)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_reward_engine(config: dict):
    """Lazy-load RewardEngine from rl/reward.py."""
    try:
        from rl.reward import RewardEngine
        r = config.get("reward", {})
        return RewardEngine(
            clarity_weight=r.get("clarity_weight", 0.30),
            consistency_weight=r.get("consistency_weight", 0.40),
            coverage_delta_weight=r.get("coverage_delta_weight", 0.30),
        )
    except Exception as e:
        if console:
            console.print(f"[dim yellow]⚠  RewardEngine unavailable: {e}[/dim yellow]")
        return None


def _get_oracle():
    """Lazy-load Oracle from sim/oracle.py."""
    try:
        from sim.oracle import Oracle
        return Oracle()
    except Exception as e:
        if console:
            console.print(f"[dim yellow]⚠  Oracle unavailable: {e}[/dim yellow]")
        return None


def _get_memory(config: dict):
    """Lazy-load EpisodeMemory from rl/memory.py."""
    try:
        from rl.memory import EpisodeMemory
        db_path = config.get("memory", {}).get("db_path", "data/episodes.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return EpisodeMemory(db_path)
    except Exception as e:
        if console:
            console.print(f"[dim yellow]⚠  EpisodeMemory unavailable: {e}[/dim yellow]")
        return None


class _MockWorkspace:
    """Minimal workspace shim so RewardEngine can access req_draft."""
    def __init__(self, req_draft: str = ""):
        self._data = {"req_draft": req_draft}

    def get(self, key, default=""):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
#  UI Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _print_header(title: str, subtitle: str = ""):
    if console:
        console.print()
        console.rule(f"[bold cyan]{title}[/bold cyan]")
        if subtitle:
            console.print(f"  [dim]{subtitle}[/dim]")
        console.print()
    else:
        print(f"\n{'═'*70}")
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(f"{'═'*70}\n")


def _print_agent_output(agent_role: str, model: str, content: str):
    """Display agent output in a nicely formatted panel."""
    if console:
        emoji = {
            "stakeholder": "🧑‍💼",
            "collector":   "🔍",
            "modeler":     "🗂️",
            "checker":     "✅",
            "documenter":  "📄",
        }.get(agent_role, "🤖")
        console.print(
            Panel(
                Markdown(content),
                title=f"{emoji}  [bold yellow]{agent_role.upper()} AGENT[/bold yellow]  "
                      f"[dim]({model})[/dim]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        print(f"\n[{agent_role.upper()}] ({model})\n")
        print(content)
        print()


def _print_stage_score(score: float, label: str = "Stage quality"):
    """Print a coloured score bar after each stage."""
    if score >= 0.7:
        colour = "green"
    elif score >= 0.4:
        colour = "yellow"
    else:
        colour = "red"

    bar_len = int(score * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)

    if console:
        console.print(
            f"  📊 [dim]{label}:[/dim]  "
            f"[{colour}]{bar}[/{colour}]  [{colour}]{score:.2f}[/{colour}]"
        )
    else:
        print(f"  {label}: {score:.2f}  [{bar}]")


def _print_reward_scores(clarity: float, consistency: float, coverage: float):
    if console:
        console.print(
            f"  [dim]RewardEngine →[/dim]  "
            f"Clarity [cyan]{clarity:.2f}[/cyan]  "
            f"Consistency [cyan]{consistency:.2f}[/cyan]  "
            f"Coverage Δ [cyan]{coverage:.2f}[/cyan]"
        )
    else:
        print(f"  Clarity={clarity:.2f}  Consistency={consistency:.2f}  "
              f"Coverage-delta={coverage:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Feedback loop (replaces old A/E/R prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def _ask_human_feedback(stage_name: str) -> str:
    """
    Ask the human for natural-language feedback or to accept the output.

    Returns:
        "" (empty string)  — human pressed Enter → accept
        non-empty string   — feedback text for the agent to incorporate
    """
    if console:
        console.print()
        console.print(
            f"[bold green]💬 {stage_name}[/bold green]  [dim]— Does this look good?[/dim]"
        )
        feedback = Prompt.ask(
            "   Type feedback to improve it, or press [bold]Enter[/bold] to accept",
            default="",
        )
    else:
        print(f"\n{stage_name} — Does this look good?")
        feedback = input("Type feedback to improve (or press Enter to accept): ").strip()

    return feedback


def _human_edit(current_content: str) -> str:
    """Fallback: let the human fully replace content by typing END on a new line."""
    if console:
        console.print(
            "[dim]Enter your replacement below. Type [bold]END[/bold] on a new line when done:[/dim]"
        )
    else:
        print("Enter replacement. Type END on a new line when done:")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines) if lines else current_content


def _ask_human_questions(questions_text: str) -> str:
    """Display each question and collect human answers. Returns formatted Q&A pairs."""
    if console:
        console.print(
            Panel(
                Markdown(questions_text),
                title="[bold yellow]COLLECTOR AGENT — Clarifying Questions[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print(
            "[dim]Please answer each question below. "
            "Your answers will be used to refine the requirements.[/dim]\n"
        )
    else:
        print("\n--- CLARIFYING QUESTIONS ---")
        print(questions_text)
        print("\nPlease answer each question:")

    import re
    question_pattern = re.compile(r"Question\s+\d+\s*[:\-]", re.IGNORECASE)
    lines = questions_text.split("\n")
    questions = []
    current_q = []
    for line in lines:
        if question_pattern.match(line.strip()):
            if current_q:
                questions.append(" ".join(current_q).strip())
            current_q = [line]
        else:
            current_q.append(line)
    if current_q:
        questions.append(" ".join(current_q).strip())

    qa_pairs = []
    for i, q in enumerate(questions, 1):
        if not q.strip():
            continue
        if console:
            console.print(f"[bold]Q{i}:[/bold] {q.strip()}")
            answer = Prompt.ask(f"  [green]Your answer[/green]")
        else:
            print(f"\nQ{i}: {q.strip()}")
            answer = input("  Your answer: ").strip()
        qa_pairs.append(f"Q{i}: {q.strip()}\nA{i}: {answer}")

    return "\n\n".join(qa_pairs)


def _save_session(session: dict, session_path: Path):
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def _load_session(session_path: Path) -> dict:
    with open(session_path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  RL scoring helper
# ═══════════════════════════════════════════════════════════════════════════════

def _score_and_display(output_text: str, req_draft: str,
                       reward_engine, stage_scores: list):
    """Score agent output with RewardEngine and display result."""
    if reward_engine is None:
        return

    try:
        ws = _MockWorkspace(req_draft)
        action_result = {"output": output_text}
        score = reward_engine.score_immediate(action_result, ws)

        # Decompose for display (approximate from total)
        clarity = reward_engine._clarity(output_text)
        consistency = reward_engine._consistency(output_text, ws)
        coverage = reward_engine._coverage_delta(output_text, ws)

        _print_reward_scores(
            max(0.0, (clarity + 1) / 2),      # map [-1,1] → [0,1]
            max(0.0, (consistency + 1) / 2),
            coverage,
        )
        _print_stage_score(max(0.0, (score + 1) / 2), "Stage reward")
        stage_scores.append(score)
    except Exception as e:
        if console:
            console.print(f"[dim]Scoring skipped: {e}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Core pipeline stages
# ═══════════════════════════════════════════════════════════════════════════════

def stage_stakeholder(agents: dict, session: dict,
                      reward_engine, stage_scores: list) -> str:
    """Stage 1 — Stakeholder generates user stories."""
    from mare.agents.base import ActionType
    _print_header("Stage 1 / 6 — User Story Elicitation", "Stakeholder Agent (gemma4)")

    rough_idea = session["rough_idea"]
    domain     = session.get("domain", "general software system")
    agent      = agents["stakeholder"]
    feedback   = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Generating user stories…[/dim]")

        result = agent.execute_action(
            ActionType.SPEAK_USER_STORIES,
            {
                "system_idea":        rough_idea,
                "rough_requirements": "",
                "domain":             domain,
                "human_feedback":     feedback,
            },
        )
        user_stories = result.output_data.get("user_stories", "")
        _print_agent_output("stakeholder", agent.config.model_name, user_stories)
        _score_and_display(user_stories, "", reward_engine, stage_scores)

        feedback = _ask_human_feedback("User Stories")
        if not feedback:
            return user_stories
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — accepting current output.[/yellow]")
            return user_stories

    return user_stories


def stage_collector_questions(agents: dict, session: dict,
                               reward_engine, stage_scores: list) -> str:
    """Stage 2 — Collector proposes clarifying questions; human answers."""
    from mare.agents.base import ActionType
    _print_header("Stage 2 / 6 — Requirements Elicitation Q&A", "Collector Agent (qwen2.5:7b)")

    agent    = agents["collector"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Generating clarifying questions…[/dim]")

        result = agent.execute_action(
            ActionType.PROPOSE_QUESTION,
            {
                "user_stories":  session["user_stories"],
                "requirements":  "",
                "domain":        session.get("domain", "general software system"),
                "focus_area":    "general",
                "human_feedback": feedback,
            },
        )
        questions_text = result.output_data.get("questions", "")
        _score_and_display(questions_text, "", reward_engine, stage_scores)

        feedback = _ask_human_feedback("Clarifying Questions")
        if not feedback:
            break
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — using current questions.[/yellow]")
            break

    qa_pairs = _ask_human_questions(questions_text)
    return qa_pairs


def stage_collector_draft(agents: dict, session: dict,
                          reward_engine, stage_scores: list) -> str:
    """Stage 3 — Collector writes structured requirements draft."""
    from mare.agents.base import ActionType
    _print_header("Stage 3 / 6 — Requirements Drafting", "Collector Agent (qwen2.5:7b)")

    agent    = agents["collector"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Drafting requirements…[/dim]")

        result = agent.execute_action(
            ActionType.WRITE_REQ_DRAFT,
            {
                "user_stories":       session["user_stories"],
                "qa_pairs":           session["qa_pairs"],
                "domain":             session.get("domain", "general software system"),
                "additional_context": "",
                "human_feedback":     feedback,
            },
        )
        draft = result.output_data.get("requirements_draft", "")
        _print_agent_output("collector", agent.config.model_name, draft)
        _score_and_display(draft, "", reward_engine, stage_scores)

        feedback = _ask_human_feedback("Requirements Draft")
        if not feedback:
            return draft
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — accepting current draft.[/yellow]")
            return draft

    return draft


def stage_modeler(agents: dict, session: dict,
                  reward_engine, stage_scores: list) -> tuple:
    """Stage 4 — Modeler extracts entities and relationships."""
    from mare.agents.base import ActionType
    _print_header("Stage 4 / 6 — System Modelling", "Modeler Agent (qwen2.5:7b)")

    agent    = agents["modeler"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Extracting entities…[/dim]")
        ent_result = agent.execute_action(
            ActionType.EXTRACT_ENTITY,
            {
                "requirements":  session["req_draft"],
                "domain":        session.get("domain", "general software system"),
                "focus_type":    "all",
                "human_feedback": feedback,
            },
        )
        entities = ent_result.output_data.get("entities", "")

        if console:
            console.print("[dim]Extracting relationships…[/dim]")
        rel_result = agent.execute_action(
            ActionType.EXTRACT_RELATION,
            {
                "entities":      entities,
                "requirements":  session["req_draft"],
                "domain":        session.get("domain", "general software system"),
                "human_feedback": feedback,
            },
        )
        relationships = rel_result.output_data.get("relationships", "")

        combined = f"## Entities\n\n{entities}\n\n---\n\n## Relationships\n\n{relationships}"
        _print_agent_output("modeler", agent.config.model_name, combined)
        _score_and_display(combined, session.get("req_draft", ""), reward_engine, stage_scores)

        feedback = _ask_human_feedback("System Model (entities + relationships)")
        if not feedback:
            return entities, relationships
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — accepting current model.[/yellow]")
            return entities, relationships

    return entities, relationships


def stage_checker(agents: dict, session: dict,
                  reward_engine, stage_scores: list) -> str:
    """Stage 5 — Checker performs quality analysis."""
    from mare.agents.base import ActionType
    _print_header("Stage 5 / 6 — Quality Assurance Check", "Checker Agent (llama3.1:8b)")

    agent    = agents["checker"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Running quality analysis…[/dim]")

        result = agent.execute_action(
            ActionType.CHECK_REQUIREMENT,
            {
                "requirements":   session["req_draft"],
                "entities":       session.get("entities", ""),
                "relationships":  session.get("relationships", ""),
                "user_stories":   session["user_stories"],
                "domain":         session.get("domain", "general software system"),
                "check_focus":    "comprehensive",
                "human_feedback": feedback,
            },
        )
        check_results = result.output_data.get("check_results", "")
        _print_agent_output("checker", agent.config.model_name, check_results)
        _score_and_display(check_results, session.get("req_draft", ""),
                           reward_engine, stage_scores)

        feedback = _ask_human_feedback("Quality Check Results")
        if not feedback:
            return check_results
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — accepting QA results.[/yellow]")
            return check_results

    return check_results

def stage_negotiator(agents: dict, session: dict,
                     reward_engine, stage_scores: list) -> str:
    """Stage 5b — Negotiator resolves conflicts found by Checker."""
    from mare.agents.base import ActionType
    _print_header("Stage 5b / 6 — Conflict Negotiation",
                  "Negotiator Agent (llama3.1:8b)")

    if "negotiator" not in agents:
        if console:
            console.print("[dim yellow]Negotiator agent not available — skipping.[/dim yellow]")
        return session.get("req_draft", "")

    # Only run if checker found conflicts
    check_results = session.get("check_results", "")
    conflict_keywords = ["conflict", "contradiction", "inconsistent", "incompatible"]
    has_conflicts = any(kw in check_results.lower() for kw in conflict_keywords)

    if not has_conflicts:
        if console:
            console.print("[dim green]No conflicts detected — skipping negotiation.[/dim green]")
        return session.get("req_draft", "")

    agent = agents["negotiator"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print("[dim]Negotiating conflicts…[/dim]")

        # Step 1: Negotiate conflicts
        neg_result = agent.execute_action(
            ActionType.NEGOTIATE_CONFLICT,
            {
                "error_report":   check_results,
                "requirements":   session["req_draft"],
                "domain":         session.get("domain", "general software system"),
                "stakeholders":   "",
                "human_feedback": feedback,
            },
        )
        negotiation_text = neg_result.output_data.get("negotiation_result", "")

        # Step 2: Prioritize
        pri_result = agent.execute_action(
            ActionType.PRIORITIZE_REQUIREMENTS,
            {
                "requirements":  session["req_draft"],
                "domain":        session.get("domain", "general software system"),
                "stakeholders":  "",
                "human_feedback": "",
            },
        )
        priority_text = pri_result.output_data.get("prioritization", "")

        # Step 3: Write resolved requirements
        res_result = agent.execute_action(
            ActionType.WRITE_RESOLUTION,
            {
                "requirements":      session["req_draft"],
                "negotiation_result": negotiation_text,
                "prioritization":    priority_text,
                "domain":            session.get("domain", "general software system"),
                "human_feedback":    feedback,
            },
        )
        resolved = res_result.output_data.get("resolved_requirements", session["req_draft"])

        combined = f"## Conflict Resolutions\n\n{negotiation_text}\n\n---\n\n## Priority Assignments\n\n{priority_text}\n\n---\n\n## Resolved Requirements\n\n{resolved}"
        _print_agent_output("negotiator", agent.config.model_name, combined)
        _score_and_display(resolved, session.get("req_draft", ""), reward_engine, stage_scores)

        feedback = _ask_human_feedback("Negotiated Requirements")
        if not feedback:
            return resolved
        if attempt == MAX_FEEDBACK_RETRIES:
            return resolved

    return resolved

def stage_documenter(agents: dict, session: dict,
                     reward_engine, stage_scores: list) -> str:
    """Stage 6 — Documenter generates the final IEEE 830 SRS document."""
    from mare.agents.base import ActionType
    _print_header(
        "Stage 6 / 6 — IEEE 830 SRS Document Generation",
        "Documenter Agent (llama3.1:8b)",
    )

    agent    = agents["documenter"]
    feedback = ""

    for attempt in range(MAX_FEEDBACK_RETRIES + 1):
        if console:
            console.print(
                "[dim]Generating IEEE 830 SRS document… (this may take a minute)[/dim]"
            )

        result = agent.execute_action(
            ActionType.WRITE_SRS,
            {
                "requirements":   session["req_draft"],
                "entities":       session.get("entities", ""),
                "relationships":  session.get("relationships", ""),
                "user_stories":   session["user_stories"],
                "check_results":  session.get("check_results", ""),
                "project_name":   session.get("project_name", "Software System"),
                "domain":         session.get("domain", "general software system"),
                "version":        "1.0",
                "human_feedback": feedback,
            },
        )
        srs_doc = result.output_data.get("srs_document", "")
        _print_agent_output("documenter", agent.config.model_name, srs_doc)
        _score_and_display(srs_doc, session.get("req_draft", ""),
                           reward_engine, stage_scores)

        # Auto-detect truncation: if Appendix A is missing or fewer than 5
        # requirement IDs found, inject a recovery note before asking the human.
        _oracle = _get_oracle()
        if _oracle is not None:
            _scores = _oracle.score_text(srs_doc)
            if _scores["traceability"] < 0.5 or "appendix" not in srs_doc.lower():
                auto_note = (
                    "The document appears incomplete — Appendix A / Traceability Matrix "
                    "is missing or fewer than 5 requirement IDs (FR-XXX, NFR-XXX) were found. "
                    "Please regenerate the full document, completing all 8 sections and "
                    "Appendix A with a full traceability table (FR-001…FR-008+, NFR-001…NFR-006)."
                )
                if console:
                    console.print(f"[yellow]⚠  Auto-detected incomplete SRS (traceability={_scores['traceability']:.2f}). Queuing recovery prompt.[/yellow]")
                feedback = auto_note
                continue

        feedback = _ask_human_feedback("Final SRS Document")
        if not feedback:
            return srs_doc
        if attempt == MAX_FEEDBACK_RETRIES:
            if console:
                console.print("[yellow]Max retries reached — accepting final SRS.[/yellow]")
            return srs_doc

    return srs_doc


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _print_quality_dashboard(stage_scores: list, oracle_scores: dict,
                              srs_md_path: Path, session_path: Path):
    """Print a final quality dashboard combining RL reward scores + Oracle."""
    _print_header("✅  SRS Generation Complete!", "")

    if console:
        # Oracle scores table
        oracle_table = Table(title="📊 Final SRS Quality (Oracle)", border_style="cyan")
        oracle_table.add_column("Dimension", style="bold")
        oracle_table.add_column("Score", justify="right")
        for key, val in oracle_scores.items():
            colour = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            oracle_table.add_row(
                key.replace("_", " ").title(),
                f"[{colour}]{bar}  {val:.2f}[/{colour}]"
            )
        console.print(oracle_table)

        # Pipeline reward summary
        if stage_scores:
            avg = sum(stage_scores) / len(stage_scores)
            console.print(
                f"\n  🔁 Pipeline avg reward: [bold cyan]{avg:.3f}[/bold cyan]  "
                f"[dim]({len(stage_scores)} stages scored)[/dim]"
            )

        # File paths
        console.print(
            Panel(
                f"[bold green]SRS document saved to:[/bold green]\n"
                f"  [cyan]{srs_md_path}[/cyan]\n\n"
                f"[bold]Session state saved to:[/bold]\n"
                f"  [cyan]{session_path}[/cyan]\n\n"
                f"[dim]To resume this session:\n"
                f"  python srs_pipeline.py --resume {session_path}\n\n"
                f"To train RL policies on collected episodes:\n"
                f"  python train.py --role collector --episodes 2000\n"
                f"To evaluate trained policies:\n"
                f"  python evaluate.py --checkpoint data/checkpoints/collector_final[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        print(f"\nSRS document saved to: {srs_md_path}")
        print(f"Session saved to: {session_path}")
        if oracle_scores:
            print("\nOracle Quality Scores:")
            for k, v in oracle_scores.items():
                print(f"  {k}: {v:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config_path: str, resume_path: str = None):
    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Check LLM provider health ─────────────────────────────────────────────
    from mare.utils.llm_client import LLMClient
    llm_cfg  = config.get("llm", {})
    provider = llm_cfg.get("provider", "nvidia")
    base_url = llm_cfg.get("base_url", "https://integrate.api.nvidia.com/v1")
    if not LLMClient.check_provider_health(provider, base_url):
        if provider == "nvidia":
            sys.exit(
                "\n❌  NVIDIA_API_KEY is not set or invalid.\n"
                "    Add it to your .env file:  NVIDIA_API_KEY=nvapi-...\n"
                "    Get a key at: https://build.nvidia.com\n"
                "    Then re-run: python srs_pipeline.py\n"
            )
        elif provider == "ollama":
            sys.exit(
                "\n❌  Ollama is not running.\n"
                "    Start it with:  ollama serve\n"
                "    Then re-run:    python srs_pipeline.py\n"
            )
        else:
            sys.exit(f"\n❌  LLM provider '{provider}' is not available. Check credentials.\n")

    # ── Load RL layer (non-fatal if unavailable) ──────────────────────────────
    reward_engine = _get_reward_engine(config)
    oracle        = _get_oracle()
    memory        = _get_memory(config)
    stage_scores: list = []

    # ── Build agents ──────────────────────────────────────────────────────────
    from mare.agents.factory import AgentFactory
    if console:
        console.print("[dim]Loading agents…[/dim]")
    agents = AgentFactory.create_all_agents_from_config(config)

    if console:
        console.print()
        for role, agent in agents.items():
            console.print(
                f"  ✓ [cyan]{role:<12}[/cyan] → "
                f"[yellow]{agent.config.model_name}[/yellow]"
            )
        if reward_engine:
            console.print("  ✓ [cyan]RewardEngine[/cyan]  → [yellow]rl/reward.py[/yellow]")
        if oracle:
            console.print("  ✓ [cyan]Oracle       [/cyan]  → [yellow]sim/oracle.py[/yellow]")
        if memory:
            console.print("  ✓ [cyan]EpisodeMemory[/cyan]  → [yellow]rl/memory.py[/yellow]")
        console.print()

    # ── Load or start session ─────────────────────────────────────────────────
    if resume_path:
        session = _load_session(Path(resume_path))
        if console:
            console.print(f"[green]Resumed session from[/green] {resume_path}")
    else:
        session = {}

    # ── Welcome banner ────────────────────────────────────────────────────────
    _print_header(
        "REMARL — Human-in-the-Loop SRS Generator",
        "Multi-Agent Requirements Engineering · IEEE 830 Output",
    )

    # ── Collect project metadata ──────────────────────────────────────────────
    if "rough_idea" not in session:
        if console:
            session["project_name"] = Prompt.ask(
                "[bold]Project name[/bold]", default="My Software System"
            )
            session["domain"] = Prompt.ask(
                "[bold]Domain / industry[/bold]", default="general software system"
            )
            console.print(
                "\n[bold]Describe your requirements idea[/bold] "
                "[dim](as much or as little detail as you like)[/dim]:"
            )
            lines = []
            console.print("[dim]Type your requirements, then press Enter twice to finish:[/dim]")
            blank_count = 0
            while blank_count < 2:
                line = input()
                if line == "":
                    blank_count += 1
                else:
                    blank_count = 0
                lines.append(line)
            session["rough_idea"] = "\n".join(lines).strip()
        else:
            session["project_name"] = input("Project name: ").strip() or "My Software System"
            session["domain"] = input("Domain / industry: ").strip() or "general software system"
            print("\nDescribe your requirements idea (press Enter twice to finish):")
            lines = []
            blank_count = 0
            while blank_count < 2:
                line = input()
                if line == "":
                    blank_count += 1
                else:
                    blank_count = 0
                lines.append(line)
            session["rough_idea"] = "\n".join(lines).strip()

    # ── Session save path ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in session["project_name"]
    )
    session_path = Path("output") / f"srs_session_{safe_name}_{timestamp}.json"
    session_path.parent.mkdir(exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 1 — User Stories
    # ═════════════════════════════════════════════════════════════════════════
    if "user_stories" not in session:
        session["user_stories"] = stage_stakeholder(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 2 — Q&A Elicitation
    # ═════════════════════════════════════════════════════════════════════════
    if "qa_pairs" not in session:
        session["qa_pairs"] = stage_collector_questions(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 3 — Requirements Draft
    # ═════════════════════════════════════════════════════════════════════════
    if "req_draft" not in session:
        session["req_draft"] = stage_collector_draft(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 4 — System Model
    # ═════════════════════════════════════════════════════════════════════════
    if "entities" not in session or "relationships" not in session:
        session["entities"], session["relationships"] = stage_modeler(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 5 — Quality Check
    # ═════════════════════════════════════════════════════════════════════════
    if "check_results" not in session:
        session["check_results"] = stage_checker(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 5b — Conflict Negotiation
    # ═════════════════════════════════════════════════════════════════════════
    if "negotiated_draft" not in session:
        session["negotiated_draft"] = stage_negotiator(
            agents, session, reward_engine, stage_scores
        )
        # Update req_draft with the resolved version
        if session["negotiated_draft"]:
            session["req_draft"] = session["negotiated_draft"]
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Stage 6 — Final SRS
    # ═════════════════════════════════════════════════════════════════════════
    if "srs_document" not in session:
        session["srs_document"] = stage_documenter(
            agents, session, reward_engine, stage_scores
        )
        _save_session(session, session_path)

    # ═════════════════════════════════════════════════════════════════════════
    #  Oracle final scoring
    # ═════════════════════════════════════════════════════════════════════════
    oracle_scores = {}
    if oracle is not None:
        try:
            oracle_scores = oracle.score_text(
                session["srs_document"],
                user_stories=session.get("user_stories", ""),
            )
            session["oracle_scores"] = oracle_scores
        except Exception as e:
            if console:
                console.print(f"[dim yellow]Oracle scoring failed: {e}[/dim yellow]")

    # ═════════════════════════════════════════════════════════════════════════
    #  EpisodeMemory — log the session for future RL training
    # ═════════════════════════════════════════════════════════════════════════
    if memory is not None:
        try:
            avg_reward = (sum(stage_scores) / len(stage_scores)) if stage_scores else 0.0
            oracle_overall = oracle_scores.get("overall", 0.0)
            memory.store(
                episode_id=int(timestamp),
                domain=session.get("domain", "hitl"),
                reward=float((avg_reward + 1) / 2 * 0.7 + oracle_overall * 0.3),
                experiences=[
                    {"stage": i, "score": s}
                    for i, s in enumerate(stage_scores)
                ],
                covered_pct=oracle_overall,
            )
        except Exception as e:
            if console:
                console.print(f"[dim yellow]EpisodeMemory store failed: {e}[/dim yellow]")

    # ═════════════════════════════════════════════════════════════════════════
    #  Write output files
    # ═════════════════════════════════════════════════════════════════════════
    srs_md_path = Path("output") / f"srs_{safe_name}_{timestamp}.md"
    with open(srs_md_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- Generated by REMARL Human-in-the-Loop Pipeline -->\n")
        f.write(f"<!-- Project: {session['project_name']} | Date: {timestamp} -->\n\n")
        if oracle_scores:
            f.write("<!-- Oracle Quality Scores: " +
                    ", ".join(f"{k}={v:.3f}" for k, v in oracle_scores.items()) +
                    " -->\n\n")
        f.write(session["srs_document"])

    _save_session(session, session_path)
    _print_quality_dashboard(stage_scores, oracle_scores, srs_md_path, session_path)
    return str(srs_md_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="REMARL Human-in-the-Loop SRS Generator — "
                    "produces an IEEE 830 SRS using local Ollama LLMs"
    )
    parser.add_argument(
        "--config",
        default="configs/remarl_config.yaml",
        help="Path to REMARL config YAML (default: configs/remarl_config.yaml)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_JSON",
        help="Path to a saved session JSON to resume from",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.resume)


if __name__ == "__main__":
    main()
