"""
MARE CLI - Agent Factory
Factory for creating and configuring MARE agents

Updated for REMARL local-LLM support:
  - create_agent_from_dict() reads provider, base_url, and per-role model
  - create_all_agents_from_config() reads the full YAML config structure
  - Default configs now target Ollama + local models
"""

from typing import Dict, Any, Optional, Type
from mare.agents.base import AbstractAgent, AgentRole, AgentConfig
from mare.agents.stakeholder import StakeholderAgent
from mare.agents.collector import CollectorAgent
from mare.agents.modeler import ModelerAgent
from mare.agents.checker import CheckerAgent
from mare.agents.documenter import DocumenterAgent
from mare.utils.exceptions import ConfigurationError
from mare.utils.logging import get_logger
from mare.agents.negotiator import NegotiatorAgent

logger = get_logger(__name__)

# Default NVIDIA Build API models per role
_DEFAULT_MODELS: Dict[str, str] = {
    "stakeholder": "meta/llama-3.2-3b-instruct",
    "collector":   "mistralai/mixtral-8x7b-instruct-v0.1",
    "modeler":     "meta/llama-3.2-3b-instruct",
    "checker":     "meta/llama-3.1-8b-instruct",
    "documenter":  "upstage/solar-10.7b-instruct",
    "negotiator":  "nvidia/llama-3.1-nemotron-nano-8b-v1",
}
    



class AgentFactory:
    """
    Factory class for creating MARE agents.
    
    Provides centralized agent creation and configuration management.
    """
    
    # Mapping of agent roles to their implementation classes
    AGENT_CLASSES: Dict[AgentRole, Type[AbstractAgent]] = {
        AgentRole.STAKEHOLDER: StakeholderAgent,
        AgentRole.COLLECTOR:   CollectorAgent,
        AgentRole.MODELER:     ModelerAgent,
        AgentRole.CHECKER:     CheckerAgent,
        AgentRole.DOCUMENTER:  DocumenterAgent,
        AgentRole.NEGOTIATOR:  NegotiatorAgent,   # ← ADD THIS
    }
    
    @classmethod
    def create_agent(
        cls, 
        role: AgentRole, 
        config: AgentConfig
    ) -> AbstractAgent:
        """
        Create an agent instance for the specified role.
        
        Args:
            role: The role of the agent to create
            config: Configuration for the agent
            
        Returns:
            Configured agent instance
        """
        if role not in cls.AGENT_CLASSES:
            raise ConfigurationError(
                f"Unsupported agent role: {role.value}",
                config_file="agent_factory"
            )
        
        agent_class = cls.AGENT_CLASSES[role]
        
        try:
            logger.info(
                f"Creating {role.value} agent | "
                f"provider={config.provider} model={config.model_name}"
            )
            agent = agent_class(config)
            logger.info(f"Successfully created {role.value} agent")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create {role.value} agent: {e}")
            raise ConfigurationError(
                f"Failed to create {role.value} agent: {e}",
                config_file="agent_factory"
            )
    
    @classmethod
    def create_agent_from_dict(
        cls, 
        role: AgentRole, 
        config_dict: Dict[str, Any]
    ) -> AbstractAgent:
        """
        Create an agent from a configuration dictionary.
        
        Args:
            role: The role of the agent to create
            config_dict: Configuration dictionary (may include 'provider',
                         'base_url', 'model', 'temperature', 'max_tokens')
        """
        role_name = role.value
        config = AgentConfig(
            role=role,
            model_name=config_dict.get("model", _DEFAULT_MODELS.get(role_name, "meta/llama-3.1-8b-instruct")),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 2048),
            system_prompt=config_dict.get("system_prompt"),
            enabled=config_dict.get("enabled", True),
            custom_parameters=config_dict.get("custom_parameters"),
            provider=config_dict.get("provider", "nvidia"),
            base_url=config_dict.get("base_url", "https://integrate.api.nvidia.com/v1"),
        )
        
        return cls.create_agent(role, config)
    
    @classmethod
    def create_all_agents(
        cls, 
        agent_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[AgentRole, AbstractAgent]:
        """
        Create all agents from a per-role configuration dict.
        
        Args:
            agent_configs: {role_name: config_dict} mapping
        """
        agents = {}
        
        for role in AgentRole:
            role_name = role.value
            if role_name in agent_configs:
                config_dict = agent_configs[role_name]
                if config_dict.get("enabled", True):
                    try:
                        agent = cls.create_agent_from_dict(role, config_dict)
                        agents[role] = agent
                        logger.info(f"Created and registered {role_name} agent")
                    except Exception as e:
                        logger.error(f"Failed to create {role_name} agent: {e}")
                else:
                    logger.info(f"Skipping disabled {role_name} agent")
            else:
                logger.warning(f"No configuration found for {role_name} agent")
        
        return agents

    @classmethod
    def create_all_agents_from_config(
        cls,
        full_config: Dict[str, Any],
    ) -> Dict[str, AbstractAgent]:
        """
        Create all 5 MARE agents using the master REMARL YAML config.

        Reads llm.provider, llm.base_url, llm.temperature, llm.max_tokens,
        and llm.agent_models to assign the correct model to each role.

        Returns:
            dict mapping role_name (str) → agent instance
            e.g. {"stakeholder": StakeholderAgent, ...}
        """
        llm_cfg = full_config.get("llm", {})
        provider   = llm_cfg.get("provider", "nvidia")
        base_url   = llm_cfg.get("base_url", "https://integrate.api.nvidia.com/v1")
        default_max  = llm_cfg.get("max_tokens", 512)
        temperature = llm_cfg.get("temperature", 0.2)
        agent_models = llm_cfg.get("agent_models", {})
        # NEW: per-agent token limits
        agent_tokens = llm_cfg.get("agent_max_tokens", {})

        agents: Dict[str, AbstractAgent] = {}

        for role in AgentRole:
            role_name = role.value
            model = agent_models.get(role_name, _DEFAULT_MODELS.get(role_name, "meta/llama-3.1-8b-instruct"))
            max_tok   = agent_tokens.get(role_name, default_max)

            config_dict = {
                "model":       model,
                "provider":    provider,
                "base_url":    base_url,
                "temperature": temperature,
                "max_tokens":  max_tok,
                "enabled":     True,
            }

            try:
                agent = cls.create_agent_from_dict(role, config_dict)
                agents[role_name] = agent
                logger.info(
                    f"[{role_name}] created → {provider}/{model}"
                )
            except Exception as e:
                logger.error(f"Failed to create {role_name} agent: {e}")

        return agents

    @classmethod
    def get_default_config(cls, role: AgentRole) -> AgentConfig:
        """
        Get default configuration for an agent role using local Ollama models.
        """
        role_name = role.value
        return AgentConfig(
            role=role,
            model_name=_DEFAULT_MODELS.get(role_name, "meta/llama-3.1-8b-instruct"),
            temperature=0.2,
            max_tokens=2048,
            provider="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
        )
    
    @classmethod
    def validate_agent_config(cls, config: AgentConfig) -> bool:
        """Validate agent configuration."""
        if not config.model_name:
            raise ConfigurationError(
                "Model name is required for agent configuration",
                config_file="agent_config"
            )
        
        if config.temperature < 0 or config.temperature > 2:
            raise ConfigurationError(
                "Temperature must be between 0 and 2",
                config_file="agent_config"
            )
        
        if config.max_tokens <= 0:
            raise ConfigurationError(
                "Max tokens must be positive",
                config_file="agent_config"
            )
        
        if config.role not in cls.AGENT_CLASSES:
            raise ConfigurationError(
                f"Unsupported agent role: {config.role.value}",
                config_file="agent_config"
            )
        
        return True
