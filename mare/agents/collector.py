"""
MARE CLI - Collector Agent Implementation
Agent responsible for collecting requirements through questioning and drafting
"""

from typing import Any, Dict, List
from mare.agents.base import AbstractAgent, AgentRole, ActionType, AgentConfig


class CollectorAgent(AbstractAgent):
    """
    Collector Agent Implementation.
    
    This agent is responsible for:
    - Proposing questions to clarify requirements (ProposeQuestion)
    - Writing requirements drafts based on collected information (WriteReqDraft)
    """
    
    def __init__(self, config: AgentConfig):
        """Initializes the Collector Agent."""
        # Ensure the role is defined correctly
        config.role = AgentRole.COLLECTOR
        
        # Set default system prompt if not provided
        if not config.system_prompt:
            config.system_prompt = self.get_system_prompt()
        
        super().__init__(config)
    
    def can_perform_action(self, action_type: ActionType) -> bool:
        """Checks if this agent can perform the specified action."""
        allowed_actions = {
            ActionType.PROPOSE_QUESTION,
            ActionType.WRITE_REQ_DRAFT
        }
        return action_type in allowed_actions
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Collector Agent."""
        return """You are an experienced requirements collector in a software engineering process. Your role is to:

1. Analyze user stories and initial requirements to identify gaps and ambiguities
2. Propose targeted, strategic questions that help clarify and refine requirements
3. Write comprehensive requirements drafts based on collected information
4. Ensure requirements are complete, clear, and actionable

Questioning guidelines:
- Ask specific, focused questions that address key gaps in understanding
- Focus on clarifying both functional and non-functional requirements
- Identify edge cases and exceptional scenarios
- Explore user workflows and business processes
- Clarify data requirements and system interfaces

Requirements drafting guidelines:
- Write clear, unambiguous requirement statements
- Use a structured format with unique identifiers (FR-001, NFR-001, etc.)
- Include acceptance criteria where appropriate
- Organize requirements logically by feature or component
- Ensure traceability to original user stories

Your goal is to transform high-level user needs into detailed, implementable requirements. Always respond in English."""
    
    def _execute_specific_action(
        self, 
        action_type: ActionType, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes a specific action implementation."""
        
        if action_type == ActionType.PROPOSE_QUESTION:
            return self._propose_question(input_data)
        elif action_type == ActionType.WRITE_REQ_DRAFT:
            return self._write_req_draft(input_data)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
    
    def _propose_question(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proposes questions to clarify requirements.
        
        Args:
            input_data: Must contain 'user_stories' or 'requirements'
            
        Returns:
            Dictionary containing proposed questions
        """
        user_stories = input_data.get('user_stories', '')
        requirements = input_data.get('requirements', '')
        domain = input_data.get('domain', 'general software system')
        focus_area = input_data.get('focus_area', 'general')
        human_feedback = input_data.get('human_feedback', '')

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """Analyze the following user stories and requirements to identify areas that need clarification. Propose specific, strategic questions that will help refine and complete the requirements.

User Stories: {user_stories}
Current Requirements: {requirements}
Domain: {domain}
Focus Area: {focus_area}

Please provide:
1. 3-5 specific questions that address the most important gaps or ambiguities
2. For each question, explain why it is important and what information it will help clarify
3. Prioritize questions that will have the greatest impact on system design and implementation
4. Consider functional requirements, non-functional requirements, constraints, and edge cases

Format your response as:
Question 1: [question]
Rationale: [why this question is important]

Question 2: [question]
Rationale: [why this question is important]

All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'user_stories': user_stories,
            'requirements': requirements,
            'domain': domain,
            'focus_area': focus_area,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'questions': response,
            'focus_area': focus_area,
            'analysis_basis': 'user_stories_and_requirements',
            'question_count': self._extract_question_count(response)
        }
    
    def _write_req_draft(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes requirements drafts based on the collected information.
        
        Args:
            input_data: Must contain 'user_stories', 'qa_pairs', etc.
            
        Returns:
            Dictionary containing the requirements draft
        """
        user_stories = input_data.get('user_stories', '')
        qa_pairs = input_data.get('qa_pairs', '')
        domain = input_data.get('domain', 'general software system')
        additional_context = input_data.get('additional_context', '')
        human_feedback = input_data.get('human_feedback', '')

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """Based on the user stories and question-answer pairs, write a comprehensive requirements draft. Transform high-level needs into detailed, implementable requirements.

User Stories: {user_stories}

Question-Answer Pairs: {qa_pairs}

Domain: {domain}

Additional Context: {additional_context}

Please provide:
1. Functional Requirements (numbered FR-001, FR-002, etc.)
   - Clear, testable requirement statements
   - Acceptance criteria for each requirement
   - Priority level (High/Medium/Low)

2. Non-Functional Requirements (numbered NFR-001, NFR-002, etc.)
   - Performance, security, usability, scalability, etc.
   - Measurable criteria where possible

3. System Constraints and Assumptions
   - Technical constraints
   - Business constraints
   - Key assumptions made

4. Data Requirements
   - Key data entities and attributes
   - Data validation rules
   - Data relationships

Format each requirement clearly with:
- Unique ID
- Title
- Description
- Acceptance Criteria
- Priority
- Source (which user story or Q&A it derives from)

All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'user_stories': user_stories,
            'qa_pairs': qa_pairs,
            'domain': domain,
            'additional_context': additional_context,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'requirements_draft': response,
            'domain': domain,
            'source_stories': user_stories,
            'qa_basis': qa_pairs,
            'draft_type': 'comprehensive_requirements'
        }
    
    def _extract_question_count(self, questions_text: str) -> int:
        """Extract the number of questions from the response."""
        # Simple heuristic to count questions
        return questions_text.count('Question ')
    
    def analyze_and_question(
        self, 
        user_stories: str, 
        domain: str = "general software system",
        focus_area: str = "general"
    ) -> Dict[str, Any]:
        """
        Convenience method to analyze user stories and propose questions.
        
        Args:
            user_stories: User stories to analyze
            domain: Domain context
            focus_area: Specific area to focus on
            
        Returns:
            Action result with proposed questions
        """
        action = self.execute_action(
            ActionType.PROPOSE_QUESTION,
            {
                'user_stories': user_stories,
                'domain': domain,
                'focus_area': focus_area
            }
        )
        return action.output_data
    
    def draft_requirements(
        self, 
        user_stories: str,
        qa_pairs: str = "",
        domain: str = "general software system",
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Convenience method to draft requirements.
        
        Args:
            user_stories: Source user stories
            qa_pairs: Question-answer pairs for clarification
            domain: Domain context
            additional_context: Any additional context
            
        Returns:
            Action result with requirements draft
        """
        action = self.execute_action(
            ActionType.WRITE_REQ_DRAFT,
            {
                'user_stories': user_stories,
                'qa_pairs': qa_pairs,
                'domain': domain,
                'additional_context': additional_context
            }
        )
        return action.output_data


