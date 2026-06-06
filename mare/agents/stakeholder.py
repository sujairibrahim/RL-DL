"""
MARE CLI - Stakeholder Agent Implementation
Agent responsible for expressing stakeholder needs and answering questions
"""

from typing import Any, Dict, List
from mare.agents.base import AbstractAgent, AgentRole, ActionType, AgentConfig


class StakeholderAgent(AbstractAgent):
    """
    Stakeholder Agent Implementation.
    
    This agent represents stakeholders and is responsible for:
    - Expressing user stories and requirements (SpeakUserStories)
    - Answering questions from other agents (AnswerQuestion)
    """
    
    def __init__(self, config: AgentConfig):
        """Initializes the Stakeholder Agent."""
        # Ensure the role is defined correctly
        config.role = AgentRole.STAKEHOLDER
        
        # Set default system prompt if not provided
        if not config.system_prompt:
            config.system_prompt = self.get_system_prompt()
        
        super().__init__(config)
    
    def can_perform_action(self, action_type: ActionType) -> bool:
        """Checks if this agent can perform the specified action."""
        allowed_actions = {
            ActionType.SPEAK_USER_STORIES,
            ActionType.ANSWER_QUESTION
        }
        return action_type in allowed_actions
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Stakeholder Agent."""
        return """You are an experienced stakeholder representative in a software requirements engineering process. Your role is to:

1. Clearly and comprehensively express user needs and requirements
2. Provide detailed user stories that capture what users truly want from the system
3. Answer questions from other team members to clarify requirements
4. Think from the perspective of end-users and business stakeholders

Guidelines:
- Be specific and detailed in your descriptions
- Consider different types of users and their varying needs
- Include functional and non-functional requirements when relevant
- Provide context and justification for your statements
- Be consistent with previously stated requirements
- Always respond in English"""
    
    def _execute_specific_action(
        self, 
        action_type: ActionType, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes a specific agent action."""
        if action_type == ActionType.SPEAK_USER_STORIES:
            return self._speak_user_stories(input_data)
        elif action_type == ActionType.ANSWER_QUESTION:
            return self._answer_question(input_data)
        else:
            raise ValueError(f"Unsupported action: {action_type}")
    
    def _speak_user_stories(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expresses user stories based on the system idea and initial requirements.
        
        Args:
            input_data: Must contain 'system_idea' and optionally 'rough_requirements' and 'domain'
            
        Returns:
            Dictionary containing user stories and metadata
        """
        system_idea = input_data.get('system_idea', '')
        rough_requirements = input_data.get('rough_requirements', '')
        domain = input_data.get('domain', 'general software system')
        human_feedback = input_data.get('human_feedback', '')

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """Based on the following system idea and requirements, please express detailed user stories that capture what stakeholders and end-users need from this system.

System Idea: {system_idea}
Initial Requirements: {rough_requirements}
Domain: {domain}

Please provide:
1. A comprehensive set of user stories in the format: "As a [user type], I want [goal] so that [benefit]"
2. Include different user types (primary users, administrators, external systems, etc.)
3. Cover both functional and non-functional aspects when relevant
4. Provide context and justification for each story

Focus on being specific, actionable, and user-centric. Consider the full user journey and edge-case scenarios. All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'system_idea': system_idea,
            'rough_requirements': rough_requirements,
            'domain': domain,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'user_stories': response,
            'system_idea': system_idea,
            'domain': domain,
            'stakeholder_perspective': 'primary_stakeholder'
        }
    
    def _answer_question(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answers questions from other agents to clarify requirements.
        
        Args:
            input_data: Must contain 'question' and optionally 'context'
            
        Returns:
            Dictionary containing the answer
        """
        question = input_data.get('question', '')
        context = input_data.get('context', '')
        previous_stories = input_data.get('previous_stories', '')
        human_feedback = input_data.get('human_feedback', '')

        if not question:
            raise ValueError("Question is required for the AnswerQuestion action")

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """You are being asked to clarify system requirements. Please provide a detailed and helpful answer based on your understanding of stakeholder needs.

Question: {question}

Context: {context}

Previous User Stories / Requirements: {previous_stories}

Please provide:
1. A clear and direct answer to the question
2. Additional context or details that may be useful
3. Any assumptions you are making
4. Related requirements or considerations that may be relevant

Be specific and ensure your answer is consistent with any previously stated requirements. All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'question': question,
            'context': context,
            'previous_stories': previous_stories,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'answer': response,
            'question': question,
            'context': context,
            'stakeholder_perspective': 'clarification_provided'
        }
    
    def express_initial_requirements(
        self, 
        system_idea: str, 
        domain: str = "general software system"
    ) -> Dict[str, Any]:
        """
        Convenience method to express initial requirements.
        
        Args:
            system_idea: High-level description of the system
            domain: Domain or industry context
            
        Returns:
            Action result with user stories
        """
        action = self.execute_action(
            ActionType.SPEAK_USER_STORIES,
            {
                'system_idea': system_idea,
                'domain': domain
            }
        )
        return action.output_data
    
    def respond_to_question(
        self, 
        question: str, 
        context: str = "",
        previous_stories: str = ""
    ) -> Dict[str, Any]:
        """
        Convenience method to respond to questions.
        
        Args:
            question: The question to answer
            context: Additional context for the question
            previous_stories: Previously stated requirements/stories
            
        Returns:
            Action result with answer
        """
        action = self.execute_action(
            ActionType.ANSWER_QUESTION,
            {
                'question': question,
                'context': context,
                'previous_stories': previous_stories
            }
        )
        return action.output_data

