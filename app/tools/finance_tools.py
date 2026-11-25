from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.services.container import container


class UserMonthInput(BaseModel):
    user_id: int = Field(..., gt=0)
    month: str | None = Field(default=None, description="Format: YYYY-MM")


class UserOnlyInput(BaseModel):
    user_id: int = Field(..., gt=0)


class TransactionSearchInput(BaseModel):
    user_id: int = Field(..., gt=0)
    query: str = Field(..., min_length=1)
    month: str | None = Field(default=None, description="Format: YYYY-MM")


class AffordabilityInput(BaseModel):
    user_id: int = Field(..., gt=0)
    month: str = Field(..., description="Format: YYYY-MM")
    amount: float = Field(..., gt=0)


class ReportInput(BaseModel):
    user_id: int = Field(..., gt=0)
    month: str = Field(..., description="Format: YYYY-MM")


class WhatIfInput(BaseModel):
    user_id: int = Field(..., gt=0)
    month: str = Field(..., description="Baseline month, format: YYYY-MM")
    income_change: float = Field(
        default=0.0, description="Monthly change in income, e.g. +10000 for a 10k raise, -5000 for a pay cut"
    )
    expense_change: float = Field(
        default=0.0, description="Monthly change in expenses, e.g. +8000 for new rent, -3000 for a cut"
    )
    one_time_cost: float = Field(
        default=0.0, ge=0, description="One-off cost being evaluated (surgery, trip, purchase). 0 if none"
    )
    months_until_cost: int = Field(
        default=0, ge=0, description="Months from now until the cost is due, e.g. 1 for 'next month'"
    )


class FundingPlanInput(BaseModel):
    user_id: int = Field(..., gt=0)
    month: str = Field(..., description="Baseline month, format: YYYY-MM")
    cost: float = Field(..., gt=0, description="Total amount that needs to be funded")
    months_until_cost: int = Field(default=0, ge=0, description="Months from now until payment is due")


class CSVLoaderInput(BaseModel):
    table: str = Field(..., description="users|income|expenses|transactions|budgets|goals|investments|savings")


def _to_json(data: Any) -> str:
    return json.dumps(data, default=str)


def build_finance_tools() -> dict[str, StructuredTool]:
    finance = container.finance_service
    repo = container.repository
    report = container.report_service

    def income_analyzer(user_id: int, month: str | None = None) -> str:
        return _to_json(finance.income_vs_expense(user_id, month))

    def expense_analyzer(user_id: int, month: str | None = None) -> str:
        return _to_json(finance.monthly_spending(user_id, month))

    def budget_checker(user_id: int, month: str | None = None) -> str:
        resolved_month = month or datetime.today().strftime("%Y-%m")
        return _to_json(finance.budget_status(user_id, resolved_month))

    def savings_calculator(user_id: int, month: str | None = None) -> str:
        return _to_json(repo.calculate_totals(user_id, month))

    def investment_summary(user_id: int) -> str:
        return _to_json(finance.investment_summary(user_id))

    def goal_progress(user_id: int) -> str:
        return _to_json(finance.goal_progress(user_id))

    def affordability_checker(user_id: int, month: str, amount: float) -> str:
        return _to_json(finance.can_afford(user_id=user_id, month=month, amount=amount))

    def what_if_scenario(
        user_id: int,
        month: str,
        income_change: float = 0.0,
        expense_change: float = 0.0,
        one_time_cost: float = 0.0,
        months_until_cost: int = 0,
    ) -> str:
        return _to_json(
            finance.what_if_scenario(
                user_id=user_id,
                month=month,
                income_change=income_change,
                expense_change=expense_change,
                one_time_cost=one_time_cost,
                months_until_cost=months_until_cost,
            )
        )

    def funding_plan(user_id: int, month: str, cost: float, months_until_cost: int = 0) -> str:
        return _to_json(finance.funding_plan(user_id=user_id, month=month, cost=cost, months_until_cost=months_until_cost))

    def transaction_search(user_id: int, query: str, month: str | None = None) -> str:
        return _to_json(repo.get_transactions(user_id=user_id, query=query, month=month))

    def financial_report_generator(user_id: int, month: str) -> str:
        payload = {
            "month": month,
            "totals": repo.calculate_totals(user_id, month),
            "spending": finance.monthly_spending(user_id, month),
            "budget": finance.budget_status(user_id, month),
            "investments": finance.investment_summary(user_id),
            "goals": finance.goal_progress(user_id),
            "recommendations": finance.personalized_recommendations(user_id, month),
            "risks": finance.spending_risks(user_id, month),
        }
        path = report.save_monthly_report(user_id=user_id, month=month, payload=payload)
        return _to_json({"report_path": str(path), "report": payload})

    def csv_loader(table: str) -> str:
        allowed = {
            "users": repo.get_users,
            "income": lambda: repo.get_income(),
            "expenses": lambda: repo.get_expenses(),
            "transactions": lambda: repo.get_transactions(),
            "budgets": lambda: repo.get_budgets(),
            "goals": lambda: repo.get_goals(),
            "investments": lambda: repo.get_investments(),
            "savings": lambda: repo.get_savings(),
        }
        if table not in allowed:
            return _to_json({"error": f"Unsupported table: {table}"})
        return _to_json(allowed[table]())

    return {
        "income_analyzer": StructuredTool.from_function(
            func=income_analyzer,
            name="income_analyzer",
            description="Analyze income and compare against expenses and savings.",
            args_schema=UserMonthInput,
        ),
        "expense_analyzer": StructuredTool.from_function(
            func=expense_analyzer,
            name="expense_analyzer",
            description="Calculate total expenses and expense category distribution.",
            args_schema=UserMonthInput,
        ),
        "budget_checker": StructuredTool.from_function(
            func=budget_checker,
            name="budget_checker",
            description="Check if user is exceeding their monthly budget.",
            args_schema=UserMonthInput,
        ),
        "savings_calculator": StructuredTool.from_function(
            func=savings_calculator,
            name="savings_calculator",
            description="Calculate total income, total expense, and net savings.",
            args_schema=UserMonthInput,
        ),
        "investment_summary": StructuredTool.from_function(
            func=investment_summary,
            name="investment_summary",
            description="Summarize user's investment allocation and total amount.",
            args_schema=UserOnlyInput,
        ),
        "goal_progress": StructuredTool.from_function(
            func=goal_progress,
            name="goal_progress",
            description="Show progress for each financial goal.",
            args_schema=UserOnlyInput,
        ),
        "affordability_checker": StructuredTool.from_function(
            func=affordability_checker,
            name="affordability_checker",
            description=(
                "Quick check whether the user can afford a one-off amount in a month with their CURRENT finances "
                "(this month's savings + 20% of emergency fund). For hypothetical changes use what_if_scenario."
            ),
            args_schema=AffordabilityInput,
        ),
        "what_if_scenario": StructuredTool.from_function(
            func=what_if_scenario,
            name="what_if_scenario",
            description=(
                "Simulate hypothetical changes: income change (raise / pay cut), expense change, and an optional "
                "one-off cost due in N months. Returns new monthly surplus, cash available by then, shortfall and "
                "months needed to save. Use for any 'if my salary / expenses changed...' question. "
                "Keep income_change and one_time_cost as separate arguments."
            ),
            args_schema=WhatIfInput,
        ),
        "funding_plan": StructuredTool.from_function(
            func=funding_plan,
            name="funding_plan",
            description=(
                "Build a concrete plan for HOW to pay for a cost: mix of monthly surplus, temporary cuts to "
                "discretionary categories, emergency fund, liquid investments, plus computed EMI / loan options. "
                "Use whenever the user asks how to afford something or cannot fully afford it."
            ),
            args_schema=FundingPlanInput,
        ),
        "transaction_search": StructuredTool.from_function(
            func=transaction_search,
            name="transaction_search",
            description="Search user transactions by keyword and optional month.",
            args_schema=TransactionSearchInput,
        ),
        "financial_report_generator": StructuredTool.from_function(
            func=financial_report_generator,
            name="financial_report_generator",
            description="Generate and persist a monthly financial report JSON.",
            args_schema=ReportInput,
        ),
        "csv_loader": StructuredTool.from_function(
            func=csv_loader,
            name="csv_loader",
            description="Load all rows from a requested CSV table.",
            args_schema=CSVLoaderInput,
        ),
    }
