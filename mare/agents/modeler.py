"""
MARE CLI - Modeler Agent Implementation
Agent responsible for extracting entities and relationships from requirements
"""

from typing import Any, Dict, List
from mare.agents.base import AbstractAgent, AgentRole, ActionType, AgentConfig


class ModelerAgent(AbstractAgent):
    """
    Modeler Agent Implementation.
    
    This agent is responsible for:
    - Extracting entities from requirements (ExtractEntity)
    - Extracting relationships between entities (ExtractRelation)
    """
    
    def __init__(self, config: AgentConfig):
        """Initializes the Modeler Agent."""
        # Ensure the role is defined correctly
        config.role = AgentRole.MODELER
        
        # Set default system prompt if not provided
        if not config.system_prompt:
            config.system_prompt = self.get_system_prompt()
        
        super().__init__(config)
    
    def can_perform_action(self, action_type: ActionType) -> bool:
        """Checks if this agent can perform the specified action."""
        allowed_actions = {
            ActionType.EXTRACT_ENTITY,
            ActionType.EXTRACT_RELATION
        }
        return action_type in allowed_actions
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Modeler Agent."""
        return """You are an experienced requirements modeler specializing in extracting structured information from requirements documents. Your role is to:

1. Identify and extract key entities (actors, systems, data objects, processes) from requirements
2. Determine relationships between entities (associations, dependencies, compositions)
3. Create structured models that represent the conceptual architecture of the system
4. Ensure completeness and consistency in the extracted model

Entity extraction guidelines:
- Identify different entity types: Actors (users, external systems), Data Objects (entities, databases), Processes (functions, services, workflows), System Components
- For each entity, determine: Name, Type, Description, Attributes, Constraints
- Consider both explicit entities mentioned in requirements and implicit ones that are necessary

Relationship extraction guidelines:
- Identify relationship types: Association (uses, interacts with), Composition (contains, consists of), Dependency (depends on, requires), Inheritance (is a type of), Flow (data flow, control flow)
- For each relationship, specify: Source entity, Target entity, Relationship type, Cardinality, Description
- Consider both direct and derived relationships

Your output should be structured, accurate, and suitable for creating system models and diagrams. Always respond in English."""
    
    def _execute_specific_action(
        self, 
        action_type: ActionType, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific action implementation."""
        
        if action_type == ActionType.EXTRACT_ENTITY:
            return self._extract_entity(input_data)
        elif action_type == ActionType.EXTRACT_RELATION:
            return self._extract_relation(input_data)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
    
    def _extract_entity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities from requirements.
        
        Args:
            input_data: Should contain 'requirements_draft' or 'requirements'
            
        Returns:
            Dictionary containing extracted entities
        """
        requirements = input_data.get('requirements_draft', input_data.get('requirements', ''))
        domain = input_data.get('domain', 'general software system')
        focus_type = input_data.get('focus_type', 'all')
        human_feedback = input_data.get('human_feedback', '')

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """Analyze the following requirements and extract all relevant entities. Focus on identifying the key components that will be part of the system model.

Requirements: {requirements}
Domain: {domain}
Focus Type: {focus_type}

Please extract entities in the following categories:

1. ACTORS (Users, External Systems, Stakeholders)
   - Name: [entity name]
   - Type: Actor
   - Description: [brief description]
   - Attributes: [key characteristics]
   - Role: [what they do in the system]

2. DATA OBJECTS (Entities, Files, Databases, Information)
   - Name: [entity name]
   - Type: Data Object
   - Description: [what it represents]
   - Attributes: [key data fields/properties]
   - Constraints: [validation rules, business rules]

3. PROCESSES (Functions, Services, Workflows, Operations)
   - Name: [entity name]
   - Type: Process
   - Description: [what it does]
   - Inputs: [what it receives as input]
   - Outputs: [what it produces]
   - Triggers: [what starts it]

4. SYSTEM COMPONENTS (Modules, Services, Interfaces)
   - Name: [entity name]
   - Type: System Component
   - Description: [its purpose and functionality]
   - Responsibilities: [what it is responsible for]
   - Interfaces: [how it connects to other components]

For each entity, make sure to provide:
- A unique and descriptive name
- Clear categorization
- Comprehensive description
- Relevant attributes or properties
- Any constraints or business rules

+Focus on entities that are essential for understanding and implementing the system. All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'requirements': requirements,
            'domain': domain,
            'focus_type': focus_type,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'entities': response,
            'domain': domain,
            'focus_type': focus_type,
            'extraction_basis': 'requirements_analysis',
            'entity_categories': ['actors', 'data_objects', 'processes', 'system_components']
        }
    
    def _extract_relation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relationships between entities.
        
        Args:
            input_data: Must contain 'entities' and 'requirements'
            
        Returns:
            Dictionary containing extracted relationships
        """
        entities = input_data.get('entities', '')
        requirements = input_data.get('requirements', '')
        domain = input_data.get('domain', 'general software system')
        relationship_types = input_data.get('relationship_types', 'all')
        human_feedback = input_data.get('human_feedback', '')

        feedback_block = (
            f"\n\nHuman Reviewer Feedback (please revise your output to address this):\n{human_feedback}"
            if human_feedback else ""
        )

        prompt_template = """Based on the extracted entities and requirements, identify and extract relationships between the entities. Focus on creating a comprehensive relationship model.

Extracted Entities: {entities}

Requirements: {requirements}

Domain: {domain}

Relationship Types to Focus On: {relationship_types}

Please identify relationships in the following categories:

1. ASSOCIATION RELATIONSHIPS (uses, interacts with, accesses)
   - Source Entity: [entity name]
   - Target Entity: [entity name]
   - Relationship Type: Association
   - Relationship Name: [specific relationship, e.g.: "uses", "manages", "views"]
   - Cardinality: [1:1, 1:many, many:many]
   - Description: [how they are related]

2. COMPOSITION RELATIONSHIPS (contains, consists of, includes)
   - Source Entity: [container entity]
   - Target Entity: [contained entity]
   - Relationship Type: Composition
   - Relationship Name: [e.g.: "contains", "includes"]
   - Cardinality: [how many of the target in the source]
   - Description: [nature of the containment]

3. DEPENDENCY RELATIONSHIPS (depends on, requires, needs)
   - Source Entity: [dependent entity]
   - Target Entity: [dependency entity]
   - Relationship Type: Dependency
   - Relationship Name: [e.g.: "depends on", "requires"]
   - Strength: [strong, weak]
   - Description: [nature of the dependency]

4. INHERITANCE RELATIONSHIPS (is a type of, specializes)
   - Source Entity: [specialized entity]
   - Target Entity: [general entity]
   - Relationship Type: Inheritance
   - Relationship Name: "is a type of"
   - Description: [how source specializes target]

5. FLOW RELATIONSHIPS (data flow, control flow, communication)
   - Source Entity: [source]
   - Target Entity: [destination]
   - Relationship Type: Flow
   - Flow Type: [data, control, message]
   - Flow Content: [what flows between them]
   - Direction: [unidirectional, bidirectional]

For each relationship, provide:
- Clear identification of the source and target entities
- Specific type and name of the relationship
- Cardinality or multiplicity where applicable
- Description of the nature and purpose of the relationship
- Any constraints or conditions on the relationship

Ensure the relationships are consistent with the requirements and make logical sense in the domain context. All output must be in English.{feedback_block}"""

        prompt = self._format_prompt(prompt_template, {
            'entities': entities,
            'requirements': requirements,
            'domain': domain,
            'relationship_types': relationship_types,
            'feedback_block': feedback_block
        })
        
        response = self._generate_response(prompt)
        
        return {
            'relationships': response,
            'domain': domain,
            'relationship_types': relationship_types,
            'source_entities': entities,
            'relationship_categories': ['association', 'composition', 'dependency', 'inheritance', 'flow']
        }
    
    def extract_system_entities(
        self, 
        requirements: str, 
        domain: str = "general software system",
        focus_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Convenience method to extract entities from requirements.
        
        Args:
            requirements: Requirements text to analyze
            domain: Domain context
            focus_type: Type of entities to focus on
            
        Returns:
            Action result with extracted entities
        """
        action = self.execute_action(
            ActionType.EXTRACT_ENTITY,
            {
                'requirements': requirements,
                'domain': domain,
                'focus_type': focus_type
            }
        )
        return action.output_data
    
    def extract_entity_relationships(
        self, 
        entities: str,
        requirements: str,
        domain: str = "general software system",
        relationship_types: str = "all"
    ) -> Dict[str, Any]:
        """
        Convenience method to extract relationships between entities.
        
        Args:
            entities: Previously extracted entities
            requirements: Source requirements
            domain: Domain context
            relationship_types: Types of relationships to focus on
            
        Returns:
            Action result with extracted relationships
        """
        action = self.execute_action(
            ActionType.EXTRACT_RELATION,
            {
                'entities': entities,
                'requirements': requirements,
                'domain': domain,
                'relationship_types': relationship_types
            }
        )
        return action.output_data
    
    def create_complete_model(
        self, 
        requirements: str, 
        domain: str = "general software system"
    ) -> Dict[str, Any]:
        """
        Create a complete model by extracting both entities and relationships.
        
        Args:
            requirements: Requirements to model
            domain: Domain context
            
        Returns:
            Dictionary with both entities and relationships
        """
        # First extract entities
        entities_result = self.extract_system_entities(requirements, domain)
        
        # Then extract relationships based on the entities
        relationships_result = self.extract_entity_relationships(
            entities_result['entities'], 
            requirements, 
            domain
        )
        
        return {
            'entities': entities_result,
            'relationships': relationships_result,
            'domain': domain,
            'model_type': 'complete_system_model'
        }
