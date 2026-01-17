"""
Real Estate Investment Tools - Main Application
Author: Bryce Fountain | Skoll.dev

Entry point for the Streamlit app. Handles navigation and tool discovery.
Tools are loaded dynamically from the tools/ directory.
"""
import streamlit as st
from tools import get_available_tools, load_tool

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Tools",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# Discovers tools from tools/ directory and builds dynamic menu
# -----------------------------------------------------------------------------
def render_sidebar():
    """Build sidebar with dynamically discovered tools."""
    st.sidebar.title("🏠 RE Tools")
    st.sidebar.markdown("---")
    
    # Home button
    if st.sidebar.button("🏡 Home", use_container_width=True):
        st.session_state.current_tool = None
    
    st.sidebar.markdown("### Tools")
    
    # Load available tools and create navigation buttons
    tools = get_available_tools()
    for tool_id, tool_info in tools.items():
        icon = tool_info.get("icon", "📊")
        name = tool_info.get("name", tool_id)
        if st.sidebar.button(f"{icon} {name}", use_container_width=True, key=f"nav_{tool_id}"):
            st.session_state.current_tool = tool_id
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Built by [Skoll.dev](https://skoll.dev)")
    
    return tools

# -----------------------------------------------------------------------------
# Landing Page
# Displays welcome message and tool cards for quick access
# -----------------------------------------------------------------------------
def render_landing_page(tools: dict):
    """Render the home/landing page with tool overview."""
    st.title("🏠 Real Estate Investment Tools")
    st.markdown("""
    Welcome to the Real Estate Investment Tools suite. Select a tool from the 
    sidebar or click below to get started.
    """)
    
    st.markdown("---")
    st.subheader("Available Tools")
    
    if not tools:
        st.info("No tools available. Add Python files to the `tools/` directory.")
        return
    
    # Display tool cards in columns
    cols = st.columns(min(len(tools), 3))
    for idx, (tool_id, tool_info) in enumerate(tools.items()):
        with cols[idx % 3]:
            icon = tool_info.get("icon", "📊")
            name = tool_info.get("name", tool_id)
            desc = tool_info.get("description", "No description available.")
            
            st.markdown(f"### {icon} {name}")
            st.write(desc)
            if st.button("Open", key=f"open_{tool_id}"):
                st.session_state.current_tool = tool_id
                st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### Quick Tips
    - **CAP Rate** = Net Operating Income / Property Value
    - **Cash-on-Cash Return** = Annual Cash Flow / Total Cash Invested
    - **1% Rule** = Monthly rent should be ≥ 1% of purchase price
    """)

# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    """Main application entry point."""
    # Initialize session state for navigation
    if "current_tool" not in st.session_state:
        st.session_state.current_tool = None
    
    # Render sidebar and get available tools
    tools = render_sidebar()
    
    # Route to appropriate view
    if st.session_state.current_tool is None:
        render_landing_page(tools)
    else:
        tool_id = st.session_state.current_tool
        if tool_id in tools:
            try:
                tool_module = load_tool(tool_id)
                tool_module.render()
            except Exception as e:
                st.error(f"Error loading tool: {e}")
                st.session_state.current_tool = None
        else:
            st.error(f"Tool '{tool_id}' not found.")
            st.session_state.current_tool = None

if __name__ == "__main__":
    main()
