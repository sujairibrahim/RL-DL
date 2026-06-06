"""
mare/rl_adapter.py
------------------
Adapter that translates RESimEnv's interface into MARE's interface.

RESimEnv calls:  agent.perform_action(action_name: str, workspace)
MARE agents use: agent.execute_action(ActionType, input_data: dict)

This class is the only bridge you need. One instance wraps one real MARE agent.
"""

from mare.agents.base import ActionType

# Maps REMARL action name strings → MARE ActionType enums
ACTION_TYPE_MAP = {
    "speak_user_stories":         ActionType.SPEAK_USER_STORIES,
    "speak_detailed_user_stories": ActionType.SPEAK_USER_STORIES,
    "clarify_ambiguity":          ActionType.SPEAK_USER_STORIES,
    "add_nfr_context":            ActionType.SPEAK_USER_STORIES,

    "propose_question":           ActionType.PROPOSE_QUESTION,
    "flag_missing_coverage":      ActionType.PROPOSE_QUESTION,

    "write_req_draft":            ActionType.WRITE_REQ_DRAFT,
    "refine_req_draft":           ActionType.WRITE_REQ_DRAFT,

    "extract_entity":             ActionType.EXTRACT_ENTITY,
    "extract_relation":           ActionType.EXTRACT_RELATION,

    "build_use_case":             ActionType.EXTRACT_RELATION,
    "flag_modeling_inconsistency": ActionType.EXTRACT_ENTITY,

    "check_completeness":         ActionType.CHECK_REQUIREMENT,
    "check_consistency":          ActionType.CHECK_REQUIREMENT,
    "request_revision":           ActionType.CHECK_REQUIREMENT,

    "approve_and_document":       ActionType.WRITE_SRS,
    "write_final_srs":            ActionType.WRITE_SRS,
    "write_srs_section":          ActionType.WRITE_SRS,
    "refine_srs_section":         ActionType.WRITE_SRS,
    "add_traceability_matrix":    ActionType.WRITE_SRS,

    "accept_requirement":    ActionType.NEGOTIATE_CONFLICT,
    "reject_requirement":    ActionType.NEGOTIATE_CONFLICT,
    "modify_priority":       ActionType.PRIORITIZE_REQUIREMENTS,
    "defer_to_next_sprint":  ActionType.WRITE_RESOLUTION,
}

# Maps action name → which workspace field to write the output into
FIELD_MAP = {
    "speak_user_stories":         "user_stories",
    "speak_detailed_user_stories": "user_stories",
    "clarify_ambiguity":          "user_stories",
    "add_nfr_context":            "user_stories",
    "propose_question":           "questions",
    "flag_missing_coverage":      "questions",
    "write_req_draft":            "req_draft",
    "refine_req_draft":           "req_draft",
    "extract_entity":             "req_model",
    "extract_relation":           "req_model",
    "build_use_case":             "req_model",
    "flag_modeling_inconsistency": "error_report",
    "check_completeness":         "error_report",
    "check_consistency":          "error_report",
    "request_revision":           "error_report",
    "approve_and_document":       "srs_document",
    "write_final_srs":            "srs_document",
    "write_srs_section":          "srs_document",
    "refine_srs_section":         "srs_document",
    "add_traceability_matrix":    "srs_document",
    "accept_requirement":    "req_draft",
    "reject_requirement":    "req_draft",
    "modify_priority":       "req_draft",
    "defer_to_next_sprint":  "req_draft",
}

# Maps action name → which output_data key holds the text
OUTPUT_KEY_MAP = {
    ActionType.SPEAK_USER_STORIES: "user_stories",
    ActionType.PROPOSE_QUESTION:   "questions",
    ActionType.WRITE_REQ_DRAFT:    "requirements_draft",
    ActionType.EXTRACT_ENTITY:     "entities",
    ActionType.EXTRACT_RELATION:   "relationships",
    ActionType.CHECK_REQUIREMENT:  "check_results",
    ActionType.WRITE_SRS:          "srs_document",
    ActionType.NEGOTIATE_CONFLICT:      "negotiation_result",
    ActionType.PRIORITIZE_REQUIREMENTS: "prioritization",
    ActionType.WRITE_RESOLUTION:        "resolved_requirements",
}


class MARERLAgent:
    """
    Wraps a real MARE AbstractAgent so RESimEnv can call it.

    Usage:
        raw_agents = AgentFactory.create_all_agents_from_config(config)
        rl_agents = {role: MARERLAgent(raw_agents[role]) for role in raw_agents}
        # Pass rl_agents into RESimEnv instead of StubAgents
    """

    def __init__(self, mare_agent):
        self.agent = mare_agent
        self._role = mare_agent.role.value
        self._step_count = 0

    def perform_action(self, action_name: str, workspace) -> dict:
        """
        Called by RESimEnv.step().

        Translates action_name → ActionType,
        builds input_data from workspace,
        calls the real MARE agent,
        writes output back to workspace,
        returns {"output": text, "action_used": action_name}.
        """
        self._step_count += 1
        # 1. Get the ActionType for this action name
        action_type = ACTION_TYPE_MAP.get(action_name)
        if action_type is None:
            # Unknown action — return empty (safe fallback)
            return {"output": "", "action_used": action_name, "error": f"unknown action: {action_name}"}

        # 2. Build input_data from workspace — all agents get these fields
        input_data = {
            "user_stories":     workspace.get("user_stories", ""),
            "requirements":     workspace.get("req_draft", ""),
            "system_idea":      workspace.get("rough_idea", ""),
            "rough_requirements": workspace.get("initial_context", ""),
            "domain":           workspace.get("domain", "general software system"),
            "entities":         workspace.get("req_model", ""),
            "relationships":    workspace.get("req_model", ""),
            "qa_pairs":         workspace.get("questions", ""),
            "additional_context": workspace.get("initial_context", ""),
            "check_results":    workspace.get("error_report", ""),
            "project_name":     workspace.get("domain", "Software System"),
            "focus_area":       "general",
            "focus_type":       "all",
            "check_focus":      "comprehensive",
            "version":          "1.0",
            "human_feedback":   "",   # no human in RL mode
        }

        # 3. Call the real MARE agent — this calls Ollama
        try:
            result = self.agent.execute_action(action_type, input_data)
        except Exception as e:
            # Agent call failed (e.g. Ollama timeout) — return safe empty result
            return {"output": "", "action_used": action_name, "error": str(e)}

        # 4. Extract the text output from result.output_data
        output_key = OUTPUT_KEY_MAP.get(action_type, "")
        output = ""
        if result.output_data:
            output = result.output_data.get(output_key, "")
            if not output:
                # Fallback: take first non-empty value in output_data
                for v in result.output_data.values():
                    if isinstance(v, str) and v.strip():
                        output = v
                        break

        # 5. Write output to the correct workspace field
        field = FIELD_MAP.get(action_name, "req_draft")
        existing = workspace.get(field, "")
        if existing:
            workspace.set(field, existing + "\n\n" + output)
        else:
            workspace.set(field, output)

        return {
            "output": output,
            "action_used": action_name,
            "action_type": action_type.value,
            "role": self._role,
        }

    def reset(self):
        """Reset agent conversation history between episodes."""
        self._step_count = 0
        if hasattr(self.agent, "reset"):
            self.agent.reset()