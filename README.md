# Real Estate Investment Tools
**Author:** Bryce Fountain | [Skoll.dev](https://skoll.dev)

A modular Streamlit application providing real estate investment analysis tools for property buyers and investors.

## Features

- **Modular Tool Architecture** - Add new tools by dropping Python files in the `tools/` directory
- **CAP Report Calculator** - Comprehensive property analysis including income, expenses, cash flow, and ROI metrics
- **Export Capabilities** - Download reports as CSV for record keeping

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | [Streamlit](https://streamlit.io/) |
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Streamlit native charts |
| Export | CSV via Pandas |

## Project Structure

```
realEstateTools/
├── app.py              # Main entry point & landing page
├── requirements.txt    # Python dependencies
├── README.md
└── tools/              # Modular tools directory
    ├── __init__.py     # Tool discovery & registration
    └── cap_report.py   # CAP Rate & Cash Flow Calculator
```

## Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Adding New Tools

To add a new tool, create a Python file in the `tools/` directory with this structure:

```python
"""
Tool Name: Your Tool Name
Description: Brief description of what the tool does
"""
import streamlit as st

TOOL_NAME = "Your Tool Name"
TOOL_ICON = "📊"  # Emoji icon for sidebar
TOOL_DESCRIPTION = "Brief description for the landing page"

def render():
    """Main render function called by the app framework."""
    st.title("Your Tool Name")
    # Your tool implementation here
```

The tool will automatically appear in the sidebar menu.

## Tools Included

### CAP Report Calculator
Comprehensive property investment analysis tool featuring:
- Property information tracking
- Cost breakdown (purchase price, down payment, mortgage)
- Monthly rental income projections (supports multi-unit)
- Operating expenses calculator (taxes, insurance, maintenance, utilities, etc.)
- Key metrics: CAP Rate, Cash-on-Cash Return, NOI, Cash Flow
- Amortization schedule preview
- CSV export functionality

## License

MIT License - See LICENSE file for details.
# REStreamlitTools
