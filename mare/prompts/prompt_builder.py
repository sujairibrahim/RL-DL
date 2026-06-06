"""
remarl/mare/prompts/prompt_builder.py
--------------------------------------
Builds LLM prompts for each MARE agent action.

This is a stub that returns structured prompts based on the action name
and workspace contents.  When you integrate the real MARE repo, replace
the prompt templates here with MARE's originals (or point the import
in base_agent.py to MARE's own PromptBuilder class).
"""

PROMPT_TEMPLATES = {
    "speak_user_stories": """
You are a stakeholder for a software project.
The rough project idea is: {rough_idea}
Generate 5-8 user stories in the format:
"As a [role], I want [feature] so that [benefit]."
Known context: {initial_context}
""",

    "propose_question": """
You are a Requirements Collector agent.
Current user stories:
{user_stories}
Current requirement draft:
{req_draft}
Identify the single most important ambiguity or gap in the requirements.
Ask one clear, specific clarifying question to resolve it.
""",

    "write_req_draft": """
You are a Requirements Collector agent.
User stories: {user_stories}
Initial context: {initial_context}
Write a structured requirements draft. For each requirement use the format:
"The system shall [verb] [object] [condition]."
Include both functional and non-functional requirements.
""",

    "extract_entity": """
You are a Requirements Modeler.
Requirements draft: {req_draft}
Extract all domain entities (nouns) mentioned in the requirements.
Return a list: one entity per line, capitalised.
""",

    "extract_relation": """
You are a Requirements Modeler.
Requirements draft: {req_draft}
Entities identified: {req_model}
Identify the relationships between entities.
Format: EntityA --[relationship]--> EntityB
""",

    "build_use_case": """
You are a Requirements Modeler.
Requirements draft: {req_draft}
Entities: {req_model}
Create use case descriptions for the 3 most important system interactions.
For each use case: Name, Actor, Preconditions, Main Flow, Postconditions.
""",

    "check_completeness": """
You are a Requirements Checker.
Requirements draft:
{req_draft}
Check for completeness against these criteria:
1. All user roles are defined
2. All CRUD operations are covered
3. Non-functional requirements present (performance, security, availability)
4. Error handling specified
List any gaps as: INCOMPLETE: [description]
If complete, output: COMPLETE
""",

    "check_consistency": """
You are a Requirements Checker.
Requirements draft:
{req_draft}
Check for contradictions or conflicts between requirements.
List any conflicts as: CONFLICT: [req1] contradicts [req2] because [reason]
If no conflicts found, output: CONSISTENT
""",

    "write_final_srs": """
You are a Requirements Documenter.
Requirements draft: {req_draft}
Requirements model: {req_model}
Produce a complete Software Requirements Specification (SRS) document.
Structure: 1. Introduction, 2. Overall Description, 3. Functional Requirements,
           4. Non-Functional Requirements, 5. System Constraints.
Use IEEE 830 format. Each requirement: unique ID, shall-statement, rationale.
""",

    "approve_and_document": """
You are a Requirements Documenter.
The requirements have passed all checks.
Requirements draft: {req_draft}
Produce the final approved SRS document with traceability matrix.
""",
}


class PromptBuilder:
    """
    Builds LLM prompts for MARE agent actions.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build("write_req_draft", workspace, "collector")
    """

    def build(self, action_name: str, workspace, agent_role: str) -> str:
        template = PROMPT_TEMPLATES.get(action_name)
        if not template:
            # Fallback for unmapped actions
            return (
                f"You are a {agent_role} agent.\n"
                f"Action: {action_name}\n"
                f"Workspace context:\n{self._workspace_summary(workspace)}\n"
                f"Perform the action and return your output."
            )

        # Fill template with workspace values
        context = {
            key: workspace.get(key, "[not yet available]")
            for key in [
                "rough_idea", "initial_context", "user_stories",
                "req_draft", "req_model", "error_report",
            ]
        }
        try:
            return template.format(**context).strip()
        except KeyError as e:
            return template.strip()   # return unfilled template on key error

    def _workspace_summary(self, workspace) -> str:
        keys = ["rough_idea", "user_stories", "req_draft", "req_model"]
        parts = []
        for k in keys:
            v = workspace.get(k, "")
            if v:
                parts.append(f"[{k}]\n{v[:300]}")
        return "\n\n".join(parts) or "[empty workspace]"
