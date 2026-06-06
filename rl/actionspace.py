# Discrete action sets per agent role
ACTION_SPACES = {
    "collector": [
        "ask_clarifying_question",   # 0: probe ambiguity
        "write_user_story",          # 1: commit to story
        "propose_requirement_draft", # 2: move to draft
        "request_more_stakeholders", # 3: need more input
    ],
    "modeler": [
        "extract_entities",          # 0: entity extraction
        "extract_relations",         # 1: relation extraction
        "build_use_case",            # 2: use case model
        "flag_modeling_gap",         # 3: signal incomplete
    ],
    "checker": [
        "approve_draft",             # 0: pass to documenter
        "flag_inconsistency",        # 1: send back to modeler
        "flag_incompleteness",       # 2: send back to collector
        "request_human_review",      # 3: escalate
    ],
    "negotiator": [                  # NEW agent in REMARL
        "accept_requirement",        # 0
        "reject_requirement",        # 1
        "modify_priority",           # 2
        "defer_to_next_sprint",      # 3
    ],
}