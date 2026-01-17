"""
Tool Discovery Module
Author: Bryce Fountain | Skoll.dev

Automatically discovers and registers tools from Python files in this directory.
Each tool must define: TOOL_NAME, TOOL_ICON, TOOL_DESCRIPTION, and render()
"""
import importlib
import os
from pathlib import Path

# Cache for discovered tools
_tools_cache = None

def get_available_tools() -> dict:
    """
    Scan the tools directory and return metadata for all valid tools.
    
    Returns:
        dict: Tool ID mapped to {name, icon, description}
    """
    global _tools_cache
    
    # Return cached results if available (cleared on app restart)
    if _tools_cache is not None:
        return _tools_cache
    
    tools = {}
    tools_dir = Path(__file__).parent
    
    # Scan for Python files (excluding __init__.py)
    for file_path in tools_dir.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        
        tool_id = file_path.stem
        try:
            # Import the module to extract metadata
            module = importlib.import_module(f"tools.{tool_id}")
            
            # Validate required attributes
            if not hasattr(module, "render"):
                continue
            
            tools[tool_id] = {
                "name": getattr(module, "TOOL_NAME", tool_id.replace("_", " ").title()),
                "icon": getattr(module, "TOOL_ICON", "📊"),
                "description": getattr(module, "TOOL_DESCRIPTION", "No description.")
            }
        except Exception as e:
            # Skip tools that fail to load
            print(f"Warning: Failed to load tool '{tool_id}': {e}")
            continue
    
    _tools_cache = tools
    return tools

def load_tool(tool_id: str):
    """
    Load and return a tool module by its ID.
    
    Args:
        tool_id: The tool identifier (filename without .py)
    
    Returns:
        module: The imported tool module
    
    Raises:
        ImportError: If tool cannot be loaded
    """
    return importlib.import_module(f"tools.{tool_id}")
