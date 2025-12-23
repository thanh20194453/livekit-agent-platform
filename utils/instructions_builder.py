"""
Instructions Builder Module
Provides helper functions for building agent instructions.
"""
import json

from utils.bot_config import BotConfiguration


def build_agent_instructions(bot_config: BotConfiguration) -> str:
    """
    Build comprehensive instructions for the dynamic voice agent.
    
    Args:
        bot_config: Bot configuration with instructions and metadata
        
    Returns:
        str: Complete instructions string for the agent
    """
    base_instructions = bot_config.instructions
    kb_tables = bot_config.knowledge_base_tables
    call_info = bot_config.metadata
    
    instructions = ""
    # Add custom instructions
    if base_instructions.strip():
        instructions += f"\n# SPECIFIC ROLE INSTRUCTIONS\n{base_instructions}\n"
        
    # Add knowledge base info
    if kb_tables:
        instructions += f"""
# KNOWLEDGE BASE ACCESS
You have access to search the following knowledge bases: {', '.join(kb_tables)}
- Use when users ask questions requiring specific information
- Always inform users when searching: "Để tôi tìm kiếm thông tin này trong cơ sở dữ liệu..."
- Present search results clearly and helpfully
- If search fails, apologize and offer alternative help
"""
    # Add call metadata info
    if call_info:
        instructions += f"""# CALL METADATA
You have access to the following call metadata: {json.dumps(call_info)}
- Use this information to provide contextually relevant responses
"""

    return instructions
