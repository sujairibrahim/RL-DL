"""
mare/agents/negotiator.py
--------------------------
NegotiatorAgent — REMARL's unique contribution over MARE.

MARE has no conflict resolution. When two requirements contradict each
other, MARE's Checker flags them but cannot resolve them.

The NegotiatorAgent fills this gap. It:
  1. Reads the error_report (conflicts flagged by Checker)
  2. Reads all stakeholder requirements from req_draft
  3. Uses multi-objective reasoning to propose a resolution
  4. Outputs a revised req_draft with resolved conflicts and an
     explicit decision trace explaining why

In RL training, the NegotiatorAgent's policy learns which resolution
strategy produces the highest downstream oracle score — i.e. which
resolutions lead to SRS documents that cover all ground-truth
requirements without contradiction.
"""

from typing import Any, Dict
from mare.agents.base import AbstractAgent, AgentRole, ActionType, AgentConfig


class NegotiatorAgent(AbstractAgent):
    """
    Negotiator Agent — resolves conflicts between stakeholder requirements.

    Actions:
        NEGOTIATE_CONFLICT  → proposes resolution for detected conflicts
        PRIORITIZE_REQS     → ranks requirements by stakeholder importance
        WRITE_RESOLUTION    → documents the final resolution decision
    """

    # Add these to ActionType enum in base.py (see instructions below)
    NEGOTIATE_CONFLICT = "negotiate_conflict"
    PRIORITIZE_REQS    = "prioritize_requirements"
    WRITE_RESOLUTION   = "write_resolution"

    def __init__(self, config: AgentConfig):
        config.role = AgentRole.NEGOTIATOR
        if not config.system_prompt:
            config.system_prompt = self.get_system_prompt()
        super().__init__(config)

    def can_perform_action(self, action_type: ActionType) -> bool:
        return action_type in {
            ActionType.NEGOTIATE_CONFLICT,
            ActionType.PRIORITIZE_REQUIREMENTS,
            ActionType.WRITE_RESOLUTION,
        }

    def get_system_prompt(self) -> str:
        return """You are an expert requirements negotiator in a software engineering team.

Your job is to resolve conflicts between stakeholder requirements. When two requirements
contradict each other, you find a solution that best satisfies all parties.

Your resolution principles:
1. Identify the core need behind each conflicting requirement
2. Find solutions that satisfy both underlying needs when possible
3. When trade-offs are unavoidable, explain them clearly and propose a priority order
4. Always document your reasoning — why was this resolution chosen?
5. Produce revised requirement statements that are consistent and unambiguous

Output format for each conflict:
  CONFLICT: [describe the contradiction]
  STAKEHOLDER A NEEDS: [core need behind req A]
  STAKEHOLDER B NEEDS: [core need behind req B]
  RESOLUTION: [the resolved requirement text]
  RATIONALE: [why this resolution best serves both parties]

All output must be in English."""

    def _execute_specific_action(
        self, action_type: ActionType, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_type == ActionType.NEGOTIATE_CONFLICT:
            return self._negotiate_conflict(input_data)
        elif action_type == ActionType.PRIORITIZE_REQUIREMENTS:
            return self._prioritize_requirements(input_data)
        elif action_type == ActionType.WRITE_RESOLUTION:
            return self._write_resolution(input_data)
        else:
            raise ValueError(f"NegotiatorAgent cannot perform {action_type}")

    def _negotiate_conflict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        error_report  = input_data.get("error_report", "")
        req_draft     = input_data.get("requirements", "")
        domain        = input_data.get("domain", "software system")
        stakeholders  = input_data.get("stakeholders", "")
        human_feedback = input_data.get("human_feedback", "")

        feedback_block = (
            f"\n\nHuman Reviewer Feedback:\n{human_feedback}"
            if human_feedback else ""
        )

        prompt = f"""You are resolving requirement conflicts in a {domain} project.

CONFLICT REPORT from Quality Checker:
{error_report}

CURRENT REQUIREMENTS DRAFT:
{req_draft}

STAKEHOLDER CONTEXT:
{stakeholders}

For each conflict identified in the report:
1. Analyse the root cause of the contradiction
2. Identify what each stakeholder truly needs (the underlying goal, not the literal statement)
3. Propose a resolution that addresses both underlying needs
4. If a trade-off is unavoidable, state which requirement takes priority and why
5. Rewrite the affected requirements to be internally consistent

Format your response as structured CONFLICT/RESOLUTION blocks as described in your instructions.

All output must be in English.{feedback_block}"""

        response = self._generate_response(prompt)
        return {
            "negotiation_result": response,
            "conflicts_addressed": self._count_conflicts(error_report),
            "domain": domain,
        }

    def _prioritize_requirements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        req_draft     = input_data.get("requirements", "")
        domain        = input_data.get("domain", "software system")
        stakeholders  = input_data.get("stakeholders", "")
        human_feedback = input_data.get("human_feedback", "")

        feedback_block = (
            f"\n\nHuman Reviewer Feedback:\n{human_feedback}"
            if human_feedback else ""
        )

        prompt = f"""You are prioritizing requirements for a {domain} project.

REQUIREMENTS DRAFT:
{req_draft}

STAKEHOLDERS:
{stakeholders}

Analyse all functional and non-functional requirements and produce:

1. MUST HAVE (P1): Requirements critical for the system to function at all
2. SHOULD HAVE (P2): Important requirements that significantly add value
3. COULD HAVE (P3): Nice-to-have features that can be deferred
4. WON'T HAVE (P4): Requirements out of scope for this release

For each requirement, explain the priority assignment in one sentence.
Consider: business impact, technical dependency, stakeholder urgency, implementation risk.

Format: [REQ-ID] [PRIORITY] [one-line rationale]

All output must be in English.{feedback_block}"""

        response = self._generate_response(prompt)
        return {
            "prioritization": response,
            "domain": domain,
        }

    def _write_resolution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        req_draft          = input_data.get("requirements", "")
        negotiation_result = input_data.get("negotiation_result", "")
        prioritization     = input_data.get("prioritization", "")
        domain             = input_data.get("domain", "software system")
        human_feedback     = input_data.get("human_feedback", "")

        feedback_block = (
            f"\n\nHuman Reviewer Feedback:\n{human_feedback}"
            if human_feedback else ""
        )

        prompt = f"""You are writing the final resolved requirements document for a {domain} project.

ORIGINAL REQUIREMENTS DRAFT:
{req_draft}

CONFLICT RESOLUTIONS:
{negotiation_result}

PRIORITY ASSIGNMENTS:
{prioritization}

Write the final revised requirements document that:
1. Incorporates all conflict resolutions
2. Marks each requirement with its priority (P1/P2/P3)
3. Includes a decision log section documenting every conflict that was resolved
4. Ensures all requirements are internally consistent — no contradictions remain
5. Maintains the original FR-XXX / NFR-XXX numbering scheme

The output should be a complete, final requirements document ready for the Documenter agent.

All output must be in English.{feedback_block}"""

        response = self._generate_response(prompt)
        return {
            "resolved_requirements": response,
            "domain": domain,
        }

    def _count_conflicts(self, error_report: str) -> int:
        keywords = ["conflict", "contradiction", "inconsistent", "incompatible"]
        return sum(error_report.lower().count(k) for k in keywords)