"""
CAP Report Calculator
Author: Bryce Fountain | Skoll.dev

Comprehensive property investment analysis tool.
Calculates CAP rate, cash flow, NOI, and generates detailed expense reports.
"""
import json
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# Tool metadata for auto-discovery
TOOL_NAME = "CAP Report"
TOOL_ICON = "📊"
TOOL_DESCRIPTION = "Calculate CAP rate, cash flow, NOI, and analyze property investments."

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def calculate_monthly_mortgage(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate monthly mortgage payment using standard amortization formula.
    
    Args:
        principal: Loan amount
        annual_rate: Annual interest rate as decimal (e.g., 0.035 for 3.5%)
        years: Loan term in years
    
    Returns:
        Monthly payment amount
    """
    if principal <= 0 or years <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / (years * 12)
    
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
              ((1 + monthly_rate)**num_payments - 1)
    return payment

def generate_amortization_schedule(principal: float, annual_rate: float, years: int) -> pd.DataFrame:
    """
    Generate full amortization schedule showing principal/interest breakdown.
    
    Returns DataFrame with columns: Month, Payment, Interest, Principal, Balance, Equity
    """
    if principal <= 0 or years <= 0:
        return pd.DataFrame()
    
    monthly_payment = calculate_monthly_mortgage(principal, annual_rate, years)
    monthly_rate = annual_rate / 12 if annual_rate > 0 else 0
    
    schedule = []
    balance = principal
    total_equity = 0
    
    for month in range(1, years * 12 + 1):
        interest = balance * monthly_rate
        principal_paid = monthly_payment - interest
        balance -= principal_paid
        total_equity += principal_paid
        
        # Record yearly snapshots and key milestones
        if month % 12 == 0 or month == 1:
            schedule.append({
                "Year": month // 12 if month % 12 == 0 else 0,
                "Month": month,
                "Payment": round(monthly_payment, 2),
                "Interest": round(interest, 2),
                "Principal": round(principal_paid, 2),
                "Balance": round(max(balance, 0), 2),
                "Equity": round(total_equity, 2)
            })
    
    return pd.DataFrame(schedule)

def format_currency(value: float) -> str:
    """Format number as currency string."""
    return f"${value:,.2f}"

INPUT_KEYS = [
    "client_name",
    "address",
    "property_type",
    "mls",
    "sqft",
    "units",
    "purchase_price",
    "land_value",
    "down_pct",
    "interest_rate",
    "loan_term",
    "vacancy_rate",
    "other_income",
    "taxes",
    "insurance",
    "pmi",
    "hoa",
    "electric",
    "gas",
    "water",
    "trash",
    "prop_mgmt_pct",
    "repairs",
    "lawn",
    "supplies",
    "screening",
    "security",
    "leased",
    "appreciation",
]


def apply_loaded_data(data: dict) -> bool:
    """Apply uploaded report data to Streamlit session state."""
    if not isinstance(data, dict):
        return False

    inputs = data.get("inputs", {})
    if isinstance(inputs, dict):
        for key in INPUT_KEYS:
            if key in inputs:
                st.session_state[key] = inputs[key]

    unit_rents = data.get("unit_rents")
    if isinstance(unit_rents, list):
        for i, rent in enumerate(unit_rents):
            st.session_state[f"unit_{i}_rent"] = rent

    return True


def build_report_payload(prop: dict, cost: dict, income: dict, expenses: dict, analysis: dict) -> dict:
    """Build JSON-serializable payload for saving the report."""
    inputs = {key: st.session_state.get(key) for key in INPUT_KEYS}
    return {
        "tool": TOOL_NAME,
        "inputs": inputs,
        "unit_rents": income.get("unit_rents", []),
        "property": prop,
        "cost": cost,
        "income": income,
        "expenses": expenses,
        "analysis": analysis,
    }


def render_save_load(prop: dict, cost: dict, income: dict, expenses: dict, analysis: dict):
    """Render save/load section for JSON reports."""
    st.subheader("💾 Save / Load Report")

    col1, col2 = st.columns(2)
    with col1:
        payload = build_report_payload(prop, cost, income, expenses, analysis)
        json_data = json.dumps(payload, indent=2)
        st.download_button(
            label="Save Report (JSON)",
            data=json_data,
            file_name="cap_report.json",
            mime="application/json",
        )

    with col2:
        uploaded = st.file_uploader("Upload Report (JSON)", type=["json"], key="cap_report_upload")
        if uploaded is not None:
            if st.button("Load Report", type="primary"):
                try:
                    data = json.loads(uploaded.getvalue().decode("utf-8"))
                    st.session_state["pending_load"] = data
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to load file: {exc}")


def build_llm_prompt(prop: dict, cost: dict, income: dict, expenses: dict, analysis: dict, question: str) -> str:
    """Create a prompt for the LLM based on report data."""
    data = {
        "property": prop,
        "cost": cost,
        "income": income,
        "expenses": expenses,
        "analysis": analysis,
    }
    return (
        "You are a conservative real estate investment analyst. Only use the provided data "
        "and avoid assumptions. If a recommendation depends on missing data, say so and list "
        "the exact inputs needed. Be critical and grounded: tie each point to a specific "
        "metric from the data (e.g., NOI, vacancy, expenses, cap rate, cash flow).\n\n"
        "Output format:\n"
        "1) Key observations (bullet list, data-grounded)\n"
        "2) Improvement actions (bullet list) — for each action include: rationale tied to data, "
        "estimated impact example using the provided numbers (e.g., +$X/mo NOI), and any assumptions.\n"
        "3) Missing data needed (if any, bullet list).\n\n"
        f"Report Data (JSON):\n{json.dumps(data, indent=2)}\n\n"
        f"User Question: {question}"
    )


def render_llm_section(prop: dict, cost: dict, income: dict, expenses: dict, analysis: dict):
    """Render LLM insights section."""
    st.subheader("🤖 LLM Insights (GPT-4o)")
    st.caption("Provide an OpenAI API key to generate improvement ideas.")

    api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
    question = st.text_area(
        "Question",
        value="How can I improve this property's NOI and cash flow based on the data?",
        height=120,
        key="llm_question",
    )

    if st.button("Ask GPT-4o", type="primary"):
        if not api_key:
            st.error("Please provide an OpenAI API key.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                prompt = build_llm_prompt(prop, cost, income, expenses, analysis, question)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a real estate investment analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                st.session_state["llm_response"] = response.choices[0].message.content
            except Exception as exc:
                st.error(f"LLM request failed: {exc}")

    if st.session_state.get("llm_response"):
        st.markdown("#### Recommendations")
        st.write(st.session_state["llm_response"])


def render_projections_tab(cost: dict, income: dict, expenses: dict, analysis: dict):
    """Render projections with what-if sliders and charts."""
    st.subheader("📊 Projections & Visuals")
    st.caption("Adjust sliders to explore scenarios without changing your saved inputs.")

    if cost.get("purchase_price", 0) <= 0:
        st.info("Enter a purchase price in the Report tab to view projections.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        rent_adjust = st.slider(
            "Total Rent Adjustment (Monthly)",
            min_value=-2000.0,
            max_value=2000.0,
            value=0.0,
            step=50.0,
            key="proj_rent_adjust",
        )
        other_income_adjust = st.slider(
            "Other Income Adjustment (Monthly)",
            min_value=-500.0,
            max_value=500.0,
            value=0.0,
            step=25.0,
            key="proj_other_income_adjust",
        )

    with col2:
        vacancy_adjust = st.slider(
            "Vacancy Rate Adjustment (pp)",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            key="proj_vacancy_adjust",
        )
        expense_adjust_pct = st.slider(
            "Operating Expense Adjustment (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
            key="proj_expense_adjust_pct",
        )

    with col3:
        st.metric("Baseline Monthly NOI", format_currency(analysis.get("monthly_noi", 0)))
        st.metric("Baseline Monthly Cash Flow", format_currency(analysis.get("monthly_cash_flow", 0)))

    projected_gross = max(income.get("gross_potential", 0) + rent_adjust, 0)
    projected_vacancy_rate = max(min(income.get("vacancy_rate", 0) + vacancy_adjust, 100), 0)
    projected_vacancy_loss = projected_gross * (projected_vacancy_rate / 100)
    projected_other_income = max(income.get("other_income", 0) + other_income_adjust, 0)
    projected_effective_income = projected_gross - projected_vacancy_loss + projected_other_income

    projected_operating_expenses = expenses.get("operating_expenses", 0) * (1 + expense_adjust_pct / 100)
    projected_noi = projected_effective_income - projected_operating_expenses

    debt_service = cost.get("monthly_mortgage", 0) + expenses.get("financing_expenses", 0)
    projected_cash_flow = projected_noi - debt_service
    projected_dscr = projected_noi / debt_service if debt_service > 0 else 0

    st.markdown("---")
    st.markdown("#### Monthly Cash Flow Comparison")
    comparison = pd.DataFrame(
        {
            "Baseline": [
                income.get("effective_income", 0),
                -expenses.get("operating_expenses", 0),
                -debt_service,
                analysis.get("monthly_cash_flow", 0),
            ],
            "Projection": [
                projected_effective_income,
                -projected_operating_expenses,
                -debt_service,
                projected_cash_flow,
            ],
        },
        index=["Effective Income", "Operating Expenses", "Debt Service", "Cash Flow"],
    )
    st.bar_chart(comparison)

    st.markdown("---")
    st.markdown("#### 12-Month Cash Flow Trend")
    trend = pd.DataFrame(
        {
            "Month": list(range(1, 13)),
            "Baseline": [analysis.get("monthly_cash_flow", 0)] * 12,
            "Projection": [projected_cash_flow] * 12,
        }
    ).set_index("Month")
    st.line_chart(trend)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Projected Monthly NOI", format_currency(projected_noi))
    with col_b:
        st.metric("Projected Monthly Cash Flow", format_currency(projected_cash_flow))
    with col_c:
        st.metric("Projected DSCR", f"{projected_dscr:.2f}")

# -----------------------------------------------------------------------------
# UI Sections
# -----------------------------------------------------------------------------
def render_property_info() -> dict:
    """Render property information input section."""
    st.subheader("🏠 Property Information")
    
    col1, col2 = st.columns(2)
    with col1:
        client_name = st.text_input("Client Name", value="", key="client_name")
        address = st.text_input("Property Address", value="", key="address")
        property_type = st.selectbox(
            "Property Type",
            ["Single Family", "Duplex", "Triplex", "4-Plex", "Multi-Family (5+)", "Commercial"],
            key="property_type"
        )
    with col2:
        mls_number = st.text_input("MLS #", value="", key="mls")
        square_footage = st.number_input("Square Footage", min_value=0, value=0, key="sqft")
        num_units = st.number_input("Number of Units", min_value=1, max_value=100, value=1, key="units")
    
    return {
        "client_name": client_name,
        "address": address,
        "property_type": property_type,
        "mls_number": mls_number,
        "square_footage": square_footage,
        "num_units": num_units
    }

def render_cost_info() -> dict:
    """Render cost and financing input section."""
    st.subheader("💰 Cost & Financing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        purchase_price = st.number_input(
            "Purchase Price", min_value=0.0, value=0.0, 
            step=10000.0, format="%.2f", key="purchase_price"
        )
        land_value = st.number_input(
            "Land Value (Tax Record)", min_value=0.0, value=0.0,
            step=1000.0, format="%.2f", key="land_value"
        )
        building_cost = purchase_price - land_value if purchase_price > land_value else 0.0
        st.metric("Building Cost", format_currency(building_cost))
    
    with col2:
        down_payment_pct = st.slider(
            "Down Payment %", min_value=0.0, max_value=100.0, 
            value=20.0, step=0.5, key="down_pct"
        )
        down_payment = purchase_price * (down_payment_pct / 100)
        st.metric("Down Payment", format_currency(down_payment))
        
        mortgage_amount = purchase_price - down_payment
        st.metric("Mortgage Amount", format_currency(mortgage_amount))
    
    with col3:
        interest_rate = st.number_input(
            "Interest Rate %", min_value=0.0, max_value=20.0,
            value=3.5, step=0.125, format="%.3f", key="interest_rate"
        )
        loan_term = st.selectbox(
            "Loan Term (Years)",
            [15, 20, 25, 30],
            index=3,
            key="loan_term"
        )
        
        monthly_mortgage = calculate_monthly_mortgage(
            mortgage_amount, interest_rate / 100, loan_term
        )
        st.metric("Monthly P&I Payment", format_currency(monthly_mortgage))
    
    return {
        "purchase_price": purchase_price,
        "land_value": land_value,
        "building_cost": building_cost,
        "down_payment_pct": down_payment_pct,
        "down_payment": down_payment,
        "mortgage_amount": mortgage_amount,
        "interest_rate": interest_rate,
        "loan_term": loan_term,
        "monthly_mortgage": monthly_mortgage
    }

def render_income_section(num_units: int) -> dict:
    """Render rental income input section."""
    st.subheader("💵 Rental Income")
    
    st.write("Enter monthly rent for each unit:")
    
    # Dynamic unit inputs based on property type
    unit_rents = []
    cols = st.columns(min(num_units, 4))
    
    for i in range(num_units):
        with cols[i % 4]:
            rent = st.number_input(
                f"Unit {i+1} Rent",
                min_value=0.0, value=0.0, step=50.0,
                format="%.2f", key=f"unit_{i}_rent"
            )
            unit_rents.append(rent)
    
    col1, col2 = st.columns(2)
    with col1:
        vacancy_rate = st.slider(
            "Vacancy Rate %", min_value=0.0, max_value=50.0,
            value=5.0, step=0.5, key="vacancy_rate"
        )
        other_income = st.number_input(
            "Other Monthly Income (Laundry, Parking, etc.)",
            min_value=0.0, value=0.0, step=25.0, format="%.2f", key="other_income"
        )
    
    gross_potential = sum(unit_rents)
    vacancy_loss = gross_potential * (vacancy_rate / 100)
    effective_income = gross_potential - vacancy_loss + other_income
    
    with col2:
        st.metric("Gross Potential Rent", format_currency(gross_potential))
        st.metric("Less: Vacancy", format_currency(-vacancy_loss))
        st.metric("Effective Gross Income", format_currency(effective_income))
    
    return {
        "unit_rents": unit_rents,
        "gross_potential": gross_potential,
        "vacancy_rate": vacancy_rate,
        "vacancy_loss": vacancy_loss,
        "other_income": other_income,
        "effective_income": effective_income
    }

def render_expenses_section(effective_income: float) -> dict:
    """Render operating expenses input section."""
    st.subheader("📋 Operating Expenses (Monthly)")
    st.caption("Operating expenses exclude financing costs. PMI/MIP is treated as financing and excluded from NOI.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Fixed Costs**")
        real_estate_taxes = st.number_input(
            "Real Estate Taxes", min_value=0.0, value=0.0,
            step=50.0, format="%.2f", key="taxes"
        )
        home_insurance = st.number_input(
            "Home Insurance", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="insurance"
        )
        mortgage_insurance = st.number_input(
            "Mortgage Insurance (PMI/MIP)", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="pmi"
        )
        hoa_fees = st.number_input(
            "HOA Fees", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="hoa"
        )
    
    with col2:
        st.markdown("**Utilities**")
        electric = st.number_input(
            "Electric", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="electric"
        )
        gas = st.number_input(
            "Gas", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="gas"
        )
        water_sewer = st.number_input(
            "Water & Sewer", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="water"
        )
        trash = st.number_input(
            "Trash/Dumpster", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="trash"
        )
    
    with col3:
        st.markdown("**Maintenance & Services**")
        property_mgmt_pct = st.slider(
            "Property Management % of EGI", min_value=0.0, max_value=25.0,
            value=8.0, step=0.5, key="prop_mgmt_pct"
        )
        property_mgmt = effective_income * (property_mgmt_pct / 100)
        st.metric("Property Management (Monthly)", format_currency(property_mgmt))
        repairs_maintenance = st.number_input(
            "Repairs & Maintenance", min_value=0.0, value=0.0,
            step=50.0, format="%.2f", key="repairs"
        )
        lawn_snow = st.number_input(
            "Lawn Care & Snow Removal", min_value=0.0, value=0.0,
            step=25.0, format="%.2f", key="lawn"
        )
        supplies = st.number_input(
            "Supplies", min_value=0.0, value=0.0,
            step=10.0, format="%.2f", key="supplies"
        )
    
    # Additional optional expenses in expander
    with st.expander("Additional Expenses"):
        col4, col5 = st.columns(2)
        with col4:
            tenant_screening = st.number_input(
                "Tenant Screening", min_value=0.0, value=0.0,
                step=10.0, format="%.2f", key="screening"
            )
            security_system = st.number_input(
                "Security System", min_value=0.0, value=0.0,
                step=10.0, format="%.2f", key="security"
            )
        with col5:
            leased_equipment = st.number_input(
                "Leased Equipment (Laundry, etc.)", min_value=0.0, value=0.0,
                step=25.0, format="%.2f", key="leased"
            )
            tenant_appreciation = st.number_input(
                "Tenant Turnover / Incentives", min_value=0.0, value=0.0,
                step=10.0, format="%.2f", key="appreciation"
            )
    
    operating_expenses = (
        real_estate_taxes + home_insurance + hoa_fees +
        electric + gas + water_sewer + trash +
        property_mgmt + repairs_maintenance + lawn_snow + supplies +
        tenant_screening + security_system + leased_equipment + tenant_appreciation
    )
    financing_expenses = mortgage_insurance
    
    return {
        "real_estate_taxes": real_estate_taxes,
        "home_insurance": home_insurance,
        "mortgage_insurance": mortgage_insurance,
        "hoa_fees": hoa_fees,
        "electric": electric,
        "gas": gas,
        "water_sewer": water_sewer,
        "trash": trash,
        "property_mgmt": property_mgmt,
        "property_mgmt_pct": property_mgmt_pct,
        "repairs_maintenance": repairs_maintenance,
        "lawn_snow": lawn_snow,
        "supplies": supplies,
        "tenant_screening": tenant_screening,
        "security_system": security_system,
        "leased_equipment": leased_equipment,
        "tenant_appreciation": tenant_appreciation,
        "operating_expenses": operating_expenses,
        "financing_expenses": financing_expenses
    }

def render_analysis(cost: dict, income: dict, expenses: dict) -> dict:
    """Calculate and display key investment metrics."""
    st.subheader("📈 Investment Analysis")
    
    # Core calculations
    monthly_noi = income["effective_income"] - expenses["operating_expenses"]
    annual_noi = monthly_noi * 12
    
    monthly_debt_service = cost["monthly_mortgage"] + expenses["financing_expenses"]
    monthly_cash_flow = monthly_noi - monthly_debt_service
    annual_cash_flow = monthly_cash_flow * 12
    
    # CAP Rate = NOI / Purchase Price
    cap_rate = (annual_noi / cost["purchase_price"] * 100) if cost["purchase_price"] > 0 else 0
    
    # Cash-on-Cash Return = Annual Cash Flow / Total Cash Invested
    total_cash_invested = cost["down_payment"]  # Simplified; could add closing costs
    cash_on_cash = (annual_cash_flow / total_cash_invested * 100) if total_cash_invested > 0 else 0
    
    # 1% Rule Check
    one_pct_rule = income["gross_potential"] / cost["purchase_price"] * 100 if cost["purchase_price"] > 0 else 0
    
    # Debt Service Coverage Ratio
    dscr = monthly_noi / monthly_debt_service if monthly_debt_service > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CAP Rate", f"{cap_rate:.2f}%")
        st.metric("Cash-on-Cash Return", f"{cash_on_cash:.2f}%")
    
    with col2:
        st.metric("Monthly NOI", format_currency(monthly_noi))
        st.metric("Annual NOI", format_currency(annual_noi))
    
    with col3:
        st.metric("Monthly Cash Flow", format_currency(monthly_cash_flow))
        st.metric("Annual Cash Flow", format_currency(annual_cash_flow))
    
    with col4:
        st.metric("1% Rule", f"{one_pct_rule:.2f}%", 
                  delta="Pass" if one_pct_rule >= 1 else "Fail",
                  delta_color="normal" if one_pct_rule >= 1 else "inverse")
        st.metric("DSCR", f"{dscr:.2f}",
                  delta="Good" if dscr >= 1.25 else "Risk",
                  delta_color="normal" if dscr >= 1.25 else "inverse")
    
    # Summary table
    st.markdown("---")
    st.markdown("#### Monthly Summary")
    
    summary_data = {
        "Category": ["Gross Potential Income", "Less: Vacancy", "Plus: Other Income",
                     "= Effective Gross Income", "Less: Operating Expenses",
                     "= Net Operating Income (NOI)", "Less: Debt Service (P&I)",
                     "Less: Financing Costs (PMI/MIP)", "= Cash Flow Before Taxes"],
        "Monthly": [
            format_currency(income["gross_potential"]),
            format_currency(-income["vacancy_loss"]),
            format_currency(income["other_income"]),
            format_currency(income["effective_income"]),
            format_currency(-expenses["operating_expenses"]),
            format_currency(monthly_noi),
            format_currency(-cost["monthly_mortgage"]),
            format_currency(-expenses["financing_expenses"]),
            format_currency(monthly_cash_flow)
        ],
        "Annual": [
            format_currency(income["gross_potential"] * 12),
            format_currency(-income["vacancy_loss"] * 12),
            format_currency(income["other_income"] * 12),
            format_currency(income["effective_income"] * 12),
            format_currency(-expenses["operating_expenses"] * 12),
            format_currency(annual_noi),
            format_currency(-cost["monthly_mortgage"] * 12),
            format_currency(-expenses["financing_expenses"] * 12),
            format_currency(annual_cash_flow)
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    return {
        "monthly_noi": monthly_noi,
        "annual_noi": annual_noi,
        "monthly_cash_flow": monthly_cash_flow,
        "annual_cash_flow": annual_cash_flow,
        "cap_rate": cap_rate,
        "cash_on_cash": cash_on_cash,
        "one_pct_rule": one_pct_rule,
        "dscr": dscr
    }

def render_amortization(cost: dict):
    """Display amortization schedule preview."""
    with st.expander("📅 Amortization Schedule"):
        if cost["mortgage_amount"] <= 0:
            st.info("Enter mortgage details to view amortization schedule.")
            return
        
        schedule = generate_amortization_schedule(
            cost["mortgage_amount"],
            cost["interest_rate"] / 100,
            cost["loan_term"]
        )
        
        if not schedule.empty:
            st.dataframe(schedule, use_container_width=True, hide_index=True)

def render_export(prop: dict, cost: dict, income: dict, expenses: dict, analysis: dict):
    """Provide CSV export of the full report."""
    st.subheader("📥 Export Report")
    
    # Build comprehensive report data
    report_lines = [
        ["CAP REPORT - Generated by Real Estate Tools"],
        [""],
        ["PROPERTY INFORMATION"],
        ["Client Name", prop.get("client_name", "")],
        ["Address", prop.get("address", "")],
        ["Property Type", prop.get("property_type", "")],
        ["MLS #", prop.get("mls_number", "")],
        ["Square Footage", prop.get("square_footage", "")],
        ["Units", prop.get("num_units", "")],
        [""],
        ["COST & FINANCING"],
        ["Purchase Price", cost["purchase_price"]],
        ["Down Payment", cost["down_payment"]],
        ["Mortgage Amount", cost["mortgage_amount"]],
        ["Interest Rate", f"{cost['interest_rate']}%"],
        ["Loan Term", f"{cost['loan_term']} years"],
        ["Monthly P&I", cost["monthly_mortgage"]],
        [""],
        ["INCOME (Monthly)"],
        ["Gross Potential Rent", income["gross_potential"]],
        ["Vacancy Loss", income["vacancy_loss"]],
        ["Other Income", income["other_income"]],
        ["Effective Gross Income", income["effective_income"]],
        [""],
        ["EXPENSES (Monthly)"],
        ["Property Management %", f"{expenses['property_mgmt_pct']:.2f}%"],
        ["Operating Expenses", expenses["operating_expenses"]],
        ["Financing Costs (PMI/MIP)", expenses["financing_expenses"]],
        [""],
        ["ANALYSIS"],
        ["CAP Rate", f"{analysis['cap_rate']:.2f}%"],
        ["Cash-on-Cash Return", f"{analysis['cash_on_cash']:.2f}%"],
        ["Monthly NOI", analysis["monthly_noi"]],
        ["Annual NOI", analysis["annual_noi"]],
        ["Monthly Cash Flow", analysis["monthly_cash_flow"]],
        ["Annual Cash Flow", analysis["annual_cash_flow"]],
        ["DSCR", f"{analysis['dscr']:.2f}"],
    ]
    
    # Convert to CSV
    df = pd.DataFrame(report_lines)
    csv = df.to_csv(index=False, header=False)
    
    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name="cap_report.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Main Render Function (Required by framework)
# -----------------------------------------------------------------------------
def render():
    """Main entry point for the CAP Report tool."""
    st.title("📊 CAP Report Calculator")
    st.markdown("Analyze property investments with comprehensive income, expense, and ROI metrics.")
    st.markdown("---")

    if st.session_state.get("pending_load"):
        if apply_loaded_data(st.session_state["pending_load"]):
            st.success("Report loaded. Updating inputs...")
        else:
            st.error("Invalid report format.")
        st.session_state.pop("pending_load", None)
        st.rerun()
    
    tab_report, tab_projections = st.tabs(["📄 Report", "📊 Projections"])

    with tab_report:
        # Render all sections and collect data
        prop_info = render_property_info()
        st.markdown("---")
        
        cost_info = render_cost_info()
        st.markdown("---")
        
        income_info = render_income_section(prop_info["num_units"])
        st.markdown("---")
        
        expenses_info = render_expenses_section(income_info["effective_income"])
        st.markdown("---")
        
        analysis_info = {}
        # Only show analysis if we have purchase price
        if cost_info["purchase_price"] > 0:
            analysis_info = render_analysis(cost_info, income_info, expenses_info)
            st.markdown("---")
            
            render_amortization(cost_info)
            st.markdown("---")
            
            render_export(prop_info, cost_info, income_info, expenses_info, analysis_info)
            st.markdown("---")
            
            render_llm_section(prop_info, cost_info, income_info, expenses_info, analysis_info)
        else:
            st.info("Enter a purchase price to see investment analysis.")

        st.markdown("---")
        render_save_load(prop_info, cost_info, income_info, expenses_info, analysis_info)

    with tab_projections:
        render_projections_tab(cost_info, income_info, expenses_info, analysis_info)
