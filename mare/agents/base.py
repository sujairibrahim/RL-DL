"""
MARE CLI - Base Agent Implementation
Abstract base class for all MARE agents

Updated for REMARL local-LLM support:
  - AgentConfig now has `provider` and `base_url` fields
  - _create_llm() supports "ollama" provider via langchain-ollama
  - OpenAI / Anthropic imports are lazy (only loaded if explicitly chosen)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from mare.utils.logging import MARELoggerMixin
from mare.utils.exceptions import AgentExecutionError, ConfigurationError


class AgentRole(Enum):
    """Enumeration of agent roles in the MARE framework."""
    STAKEHOLDER = "stakeholder"
    COLLECTOR = "collector"
    MODELER = "modeler"
    CHECKER = "checker"
    DOCUMENTER = "documenter"
    NEGOTIATOR  = "negotiator"   # ← ADD THIS


class ActionType(Enum):
    """Enumeration of action types in the MARE framework."""
    SPEAK_USER_STORIES = "speak_user_stories"
    PROPOSE_QUESTION = "propose_question"
    ANSWER_QUESTION = "answer_question"
    WRITE_REQ_DRAFT = "write_req_draft"
    EXTRACT_ENTITY = "extract_entity"
    EXTRACT_RELATION = "extract_relation"
    CHECK_REQUIREMENT = "check_requirement"
    WRITE_SRS = "write_srs"
    WRITE_CHECK_REPORT = "write_check_report"
    NEGOTIATE_CONFLICT       = "negotiate_conflict"
    PRIORITIZE_REQUIREMENTS  = "prioritize_requirements"
    WRITE_RESOLUTION         = "write_resolution"


@dataclass
class AgentAction:
    """Represents an action performed by an agent."""
    id: str
    agent_role: AgentRole
    action_type: ActionType
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    role: AgentRole
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    enabled: bool = True
    custom_parameters: Optional[Dict[str, Any]] = None
    # ── LLM provider fields ───────────────────────────────────────────────
    provider: str = "nvidia"                                  # "nvidia" | "ollama" | "openai" | "anthropic"
    base_url: str = "https://integrate.api.nvidia.com/v1"    # API endpoint URL


class AbstractAgent(ABC, MARELoggerMixin):
    """
    Abstract base class for all MARE agents.
    
    Provides common functionality and defines the interface that all
    specialized agents must implement.
    """
    
    def __init__(
        self, 
        config: AgentConfig,
        llm: Optional[BaseChatModel] = None
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            llm: Language model instance (optional, will be created if not provided)
        """
        self.config = config
        self.role = config.role
        self._llm = llm
        self._conversation_history: List[BaseMessage] = []
        self._action_history: List[AgentAction] = []
        
        # Initialize LLM if not provided
        if self._llm is None:
            self._llm = self._create_llm()
        
        # Set system prompt if provided
        if config.system_prompt:
            self._conversation_history.append(
                SystemMessage(content=config.system_prompt)
            )
        
        self.log_info(
            f"Initialized {self.role.value} agent | "
            f"provider={config.provider} | model={config.model_name}"
        )
    
    def _create_llm(self) -> BaseChatModel:
        """Create and configure the language model based on provider."""
        provider = self.config.provider.lower()

        try:
            if provider == "nvidia":
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                import os
                api_key = os.environ.get("NVIDIA_API_KEY")
                if not api_key:
                    raise ConfigurationError(
                        "NVIDIA_API_KEY is not set.\n"
                        "  → Add NVIDIA_API_KEY=nvapi-... to your .env file\n"
                        "  → Get a key at: https://build.nvidia.com",
                        config_file="agent_config",
                    )
                return ChatNVIDIA(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                    nvidia_api_key=api_key,
                )

            elif provider == "ollama":
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens,
                    num_ctx=8192,
                    base_url=self.config.base_url,
                )

            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            elif provider == "anthropic":
                from langchain_community.chat_models import ChatAnthropic
                return ChatAnthropic(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            else:
                raise ConfigurationError(
                    f"Unknown LLM provider: '{provider}'. "
                    "Use 'nvidia', 'ollama', 'openai', or 'anthropic'.",
                    config_file="agent_config",
                )

        except ImportError as e:
            raise ConfigurationError(
                f"Missing package for provider '{provider}': {e}\n"
                "  For NVIDIA:    pip install langchain-nvidia-ai-endpoints\n"
                "  For Ollama:    pip install langchain-ollama\n"
                "  For OpenAI:    pip install langchain-openai\n"
                "  For Anthropic: pip install langchain-community",
                config_file="agent_config",
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create LLM for {self.role.value} agent: {e}",
                config_file="agent_config",
            )
    
    @property
    def llm(self) -> BaseChatModel:
        """Get the language model instance."""
        return self._llm
    
    @property
    def conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history."""
        return self._conversation_history.copy()
    
    @property
    def action_history(self) -> List[AgentAction]:
        """Get the action history."""
        return self._action_history.copy()
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation history."""
        self._conversation_history.append(message)
        self.log_debug(f"Added message to {self.role.value} agent: {type(message).__name__}")
    
    def clear_conversation(self) -> None:
        """Clear the conversation history (except system prompt)."""
        system_messages = [
            msg for msg in self._conversation_history 
            if isinstance(msg, SystemMessage)
        ]
        self._conversation_history = system_messages
        self.log_debug(f"Cleared conversation history for {self.role.value} agent")
    
    def execute_action(
        self, 
        action_type: ActionType, 
        input_data: Dict[str, Any]
    ) -> AgentAction:
        """
        Execute an action and return the result.
        
        Args:
            action_type: Type of action to execute
            input_data: Input data for the action
            
        Returns:
            AgentAction with results
        """
        action = AgentAction(
            id=str(uuid.uuid4()),
            agent_role=self.role,
            action_type=action_type,
            input_data=input_data
        )
        
        try:
            self.log_info(f"Executing {action_type.value} action for {self.role.value} agent")
            
            # Validate that this agent can perform the requested action
            if not self.can_perform_action(action_type):
                raise AgentExecutionError(
                    f"Agent {self.role.value} cannot perform action {action_type.value}",
                    agent_name=self.role.value,
                    action=action_type.value
                )
            
            # Execute the specific action
            output_data = self._execute_specific_action(action_type, input_data)
            
            action.output_data = output_data
            action.status = "completed"
            
            self.log_info(f"Successfully completed {action_type.value} action")
            
        except Exception as e:
            action.status = "failed"
            action.error_message = str(e)
            self.log_error(f"Failed to execute {action_type.value} action: {e}")
            raise AgentExecutionError(
                f"Action execution failed: {e}",
                agent_name=self.role.value,
                action=action_type.value
            )
        
        finally:
            self._action_history.append(action)
        
        return action
    
    @abstractmethod
    def can_perform_action(self, action_type: ActionType) -> bool:
        """Check if this agent can perform the specified action."""
        pass
    
    @abstractmethod
    def _execute_specific_action(
        self, 
        action_type: ActionType, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific action implementation."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    def _generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using the language model.
        
        Args:
            prompt: The prompt to send to the model
            context: Optional context information
            
        Returns:
            Generated response
        """
        try:
            # ── Build capped message list ─────────────────────────────
            # Keep system prompt + last 2 exchanges (5 messages max)
            system_msgs = [m for m in self._conversation_history
                           if isinstance(m, SystemMessage)]
            recent_msgs = [m for m in self._conversation_history
                           if not isinstance(m, SystemMessage)][-4:]

            if self.config.provider == "nvidia":
                # NVIDIA models require strict user/assistant alternation.
                # Some endpoints treat SystemMessage as a user role, producing
                # [user, user] → 400. Fix: bake system prompt into the first
                # user message and send only Human/AI pairs thereafter.
                if system_msgs and not recent_msgs:
                    # First turn: prepend system prompt to user message
                    combined = f"{system_msgs[0].content}\n\n{prompt}"
                    messages = [HumanMessage(content=combined)]
                else:
                    # Subsequent turns: strict Human/AI alternation, no system msg
                    messages = recent_msgs + [HumanMessage(content=prompt)]
            else:
                capped_history = system_msgs + recent_msgs
                messages = capped_history + [HumanMessage(content=prompt)]
            # ──────────────────────────────────────────────────────────

            response = self._llm.invoke(messages)

            self.add_message(HumanMessage(content=prompt))
            self.add_message(AIMessage(content=response.content))

            return response.content
            
        except Exception as e:
            self.log_error(f"Failed to generate response: {e}")
            raise AgentExecutionError(
                f"Response generation failed: {e}",
                agent_name=self.role.value
            )
    
    def _format_prompt(
        self, 
        template: str, 
        variables: Dict[str, Any]
    ) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise AgentExecutionError(
                f"Missing variable in prompt template: {e}",
                agent_name=self.role.value
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "role": self.role.value,
            "provider": self.config.provider,
            "model": self.config.model_name,
            "enabled": self.config.enabled,
            "conversation_length": len(self._conversation_history),
            "actions_performed": len(self._action_history),
            "last_action": (
                self._action_history[-1].action_type.value 
                if self._action_history else None
            )
        }
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.clear_conversation()
        self._action_history.clear()
        
        # Re-add system prompt if configured
        if self.config.system_prompt:
            self._conversation_history.append(
                SystemMessage(content=self.config.system_prompt)
            )
        
        self.log_info(f"Reset {self.role.value} agent to initial state")
