from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from app.repository.csv_repository import CSVRepository


class FinanceService:
    def __init__(self, repository: CSVRepository) -> None:
        self.repository = repository

    def monthly_spending(self, user_id: int, month: str | None = None) -> dict[str, Any]:
        expenses = pd.DataFrame(self.repository.get_expenses(user_id=user_id, month=month))
        total = float(expenses["amount"].sum()) if not expenses.empty else 0.0
        by_category = (
            expenses.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict()
            if not expenses.empty
            else {}
        )
        return {"total_spending": total, "by_category": by_category}

    def income_vs_expense(self, user_id: int, month: str | None = None) -> dict[str, float]:
        return self.repository.calculate_totals(user_id=user_id, month=month)

    def budget_status(self, user_id: int, month: str) -> dict[str, Any]:
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        budgets = self.repository.get_budgets(user_id=user_id, month=month)
        budget_limit = float(budgets[0]["budget_limit"]) if budgets else 0.0
        spent = totals["total_expense"]
        usage_pct = (spent / budget_limit * 100.0) if budget_limit > 0 else 0.0
        return {
            "month": month,
            "budget_limit": budget_limit,
            "spent": spent,
            "is_exceeded": spent > budget_limit if budget_limit > 0 else False,
            "usage_pct": usage_pct,
        }

    def investment_summary(self, user_id: int) -> dict[str, Any]:
        investments = pd.DataFrame(self.repository.get_investments(user_id=user_id))
        if investments.empty:
            return {"total": 0.0, "by_asset": {}}

        by_asset = investments.groupby("asset_type")["amount"].sum().to_dict()
        total = float(investments["amount"].sum())
        return {"total": total, "by_asset": by_asset}

    def goal_progress(self, user_id: int) -> list[dict[str, Any]]:
        goals = self.repository.get_goals(user_id=user_id)
        output: list[dict[str, Any]] = []
        for goal in goals:
            target = float(goal["target_amount"])
            current = float(goal["current_amount"])
            output.append(
                {
                    **goal,
                    "progress_pct": (current / target * 100.0) if target > 0 else 0.0,
                }
            )
        return output

    def monthly_series(self, user_id: int) -> dict[str, Any]:
        income_df = pd.DataFrame(self.repository.get_income(user_id=user_id))
        expense_df = pd.DataFrame(self.repository.get_expenses(user_id=user_id))

        if income_df.empty:
            return {"months": [], "income": [], "expense": [], "savings": []}

        income_df["month"] = pd.to_datetime(income_df["date"]).dt.strftime("%Y-%m")
        expense_df["month"] = pd.to_datetime(expense_df["date"]).dt.strftime("%Y-%m")

        income_map = income_df.groupby("month")["amount"].sum().to_dict()
        expense_map = expense_df.groupby("month")["amount"].sum().to_dict()
        months = sorted(set(income_map.keys()) | set(expense_map.keys()))

        income_vals = [float(income_map.get(m, 0.0)) for m in months]
        expense_vals = [float(expense_map.get(m, 0.0)) for m in months]
        savings_vals = [inc - exp for inc, exp in zip(income_vals, expense_vals)]

        return {
            "months": months,
            "income": income_vals,
            "expense": expense_vals,
            "savings": savings_vals,
        }

    def can_afford(self, user_id: int, month: str, amount: float) -> dict[str, Any]:
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        emergency_goal = [g for g in self.goal_progress(user_id) if g["goal_name"] == "Emergency Fund"]
        emergency_fund = float(emergency_goal[0]["current_amount"]) if emergency_goal else 0.0

        available = max(0.0, totals["savings"]) + emergency_fund * 0.2
        return {
            "requested_amount": amount,
            "available_estimate": available,
            "can_afford": available >= amount,
        }

    DISCRETIONARY_CATEGORIES = ("Entertainment", "Shopping", "Travel", "Food")
    LIQUID_ASSETS = ("Fixed Deposit", "FD", "Mutual Fund", "Savings", "Liquid Fund", "Gold")

    def _emergency_fund_balance(self, user_id: int) -> float:
        ef = [g for g in self.goal_progress(user_id) if g["goal_name"] == "Emergency Fund"]
        # Goals can repeat across periods; the largest balance is the fund actually built up.
        return max((float(g["current_amount"]) for g in ef), default=0.0)

    def what_if_scenario(
        self,
        user_id: int,
        month: str,
        income_change: float = 0.0,
        expense_change: float = 0.0,
        one_time_cost: float = 0.0,
        months_until_cost: int = 0,
    ) -> dict[str, Any]:
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        current_surplus = totals["total_income"] - totals["total_expense"]
        new_income = totals["total_income"] + income_change
        new_expense = totals["total_expense"] + expense_change
        new_surplus = new_income - new_expense
        emergency_fund = self._emergency_fund_balance(user_id)

        # Cash available when the cost is due: surplus saved until then (incl. this month) + 20% of emergency fund.
        saving_months = max(1, months_until_cost + 1) if months_until_cost >= 0 else 1
        cash_from_surplus = max(0.0, new_surplus) * saving_months
        available_without_ef = cash_from_surplus
        available_with_ef_buffer = cash_from_surplus + emergency_fund * 0.2
        shortfall = max(0.0, one_time_cost - available_with_ef_buffer)
        months_to_save = (one_time_cost / new_surplus) if new_surplus > 0 and one_time_cost > 0 else None

        return {
            "month": month,
            "current": {
                "income": totals["total_income"],
                "expense": totals["total_expense"],
                "monthly_surplus": current_surplus,
            },
            "scenario": {
                "income_change": income_change,
                "expense_change": expense_change,
                "income": new_income,
                "expense": new_expense,
                "monthly_surplus": new_surplus,
                "surplus_increase": new_surplus - current_surplus,
            },
            "one_time_cost": one_time_cost,
            "months_until_cost": months_until_cost,
            "surplus_saved_by_then": cash_from_surplus,
            "emergency_fund_balance": emergency_fund,
            "available_without_emergency_fund": available_without_ef,
            "available_with_20pct_emergency_fund": available_with_ef_buffer,
            "can_afford_from_surplus_alone": available_without_ef >= one_time_cost if one_time_cost else None,
            "can_afford_with_emergency_buffer": available_with_ef_buffer >= one_time_cost if one_time_cost else None,
            "can_afford_using_full_emergency_fund": (cash_from_surplus + emergency_fund) >= one_time_cost if one_time_cost else None,
            "shortfall": shortfall,
            "months_of_new_surplus_to_cover_cost": months_to_save,
        }

    def funding_plan(self, user_id: int, month: str, cost: float, months_until_cost: int = 0) -> dict[str, Any]:
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        surplus = max(0.0, totals["total_income"] - totals["total_expense"])
        spending = self.monthly_spending(user_id=user_id, month=month)["by_category"]
        investments = self.investment_summary(user_id)["by_asset"]
        emergency_fund = self._emergency_fund_balance(user_id)

        saving_months = max(1, months_until_cost + 1)
        discretionary = {c: float(v) for c, v in spending.items() if c in self.DISCRETIONARY_CATEGORIES}
        # Assume half of discretionary spend can be cut temporarily.
        cut_per_month = sum(discretionary.values()) * 0.5
        liquid = {a: float(v) for a, v in investments.items() if any(k.lower() in a.lower() for k in self.LIQUID_ASSETS)}

        remaining = cost
        sources: list[dict[str, Any]] = []

        def take(name: str, available: float, note: str) -> None:
            nonlocal remaining
            if remaining <= 0 or available <= 0:
                return
            used = min(available, remaining)
            remaining -= used
            sources.append({"source": name, "amount": round(used, 2), "available": round(available, 2), "note": note})

        take("Monthly surplus saved until the cost is due", surplus * saving_months, f"{saving_months} month(s) x Rs {surplus:,.0f}")
        take("Temporary cuts to discretionary spending", cut_per_month * saving_months, "50% cut on " + ", ".join(discretionary) if discretionary else "")
        take("Emergency fund", emergency_fund, "this is what the emergency fund is for; rebuild it afterwards")
        for asset, amount in sorted(liquid.items(), key=lambda x: -x[1]):
            take(f"Redeem part of {asset}", amount, "check exit load / tax before redeeming")

        def emi(principal: float, annual_rate: float, months: int) -> float:
            if principal <= 0:
                return 0.0
            r = annual_rate / 12
            if r == 0:
                return principal / months
            return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

        emi_options = [
            {"option": "0% hospital / card EMI", "months": m, "monthly_emi": round(emi(cost, 0.0, m), 2),
             "fits_in_surplus": emi(cost, 0.0, m) <= surplus}
            for m in (3, 6)
        ] + [
            {"option": "Medical / personal loan @12%", "months": m, "monthly_emi": round(emi(cost, 0.12, m), 2),
             "fits_in_surplus": emi(cost, 0.12, m) <= surplus}
            for m in (12, 24)
        ]

        return {
            "cost": cost,
            "monthly_surplus": surplus,
            "discretionary_spend_by_category": discretionary,
            "emergency_fund_balance": emergency_fund,
            "liquid_investments": liquid,
            "suggested_funding_mix": sources,
            "uncovered_after_own_funds": round(max(0.0, remaining), 2),
            "emi_options_for_full_cost": emi_options,
            "notes": [
                "Check health insurance / employer cover first; it may cover most of a medical cost.",
                "Funding mix is applied in order: surplus, discretionary cuts, emergency fund, liquid investments.",
            ],
        }

    def spending_risks(self, user_id: int, month: str | None = None) -> list[str]:
        spending = self.monthly_spending(user_id=user_id, month=month)
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        warnings: list[str] = []

        if totals["total_income"] > 0 and totals["savings_rate"] < 15:
            warnings.append("Savings rate is below 15 percent, which increases financial risk.")

        category_map = spending["by_category"]
        total_spending = spending["total_spending"]
        for category, value in category_map.items():
            if total_spending > 0 and (value / total_spending) > 0.35:
                warnings.append(f"High concentration in {category} expenses.")

        return warnings

    def create_charts(self, user_id: int, month: str | None = None) -> dict[str, plt.Figure]:
        charts: dict[str, plt.Figure] = {}
        spending = self.monthly_spending(user_id=user_id, month=month)
        series = self.monthly_series(user_id=user_id)
        budget_info = self.budget_status(user_id, month) if month else None
        investments = self.investment_summary(user_id=user_id)

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        cats = list(spending["by_category"].keys())
        vals = list(spending["by_category"].values())
        ax1.bar(cats, vals, color="#1f77b4")
        ax1.set_title("Expense Categories")
        ax1.tick_params(axis="x", rotation=35)
        charts["expense_categories"] = fig1

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(series["months"], series["income"], label="Income", marker="o")
        ax2.plot(series["months"], series["expense"], label="Expense", marker="o")
        ax2.set_title("Income vs Expense")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=35)
        charts["income_vs_expense"] = fig2

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(series["months"], series["savings"], label="Savings", marker="o", color="#2ca02c")
        ax3.set_title("Savings Trend")
        ax3.tick_params(axis="x", rotation=35)
        charts["savings_trend"] = fig3

        month_table = self.monthly_expense_table(user_id=user_id)
        if month_table:
            fig_monthly, ax_monthly = plt.subplots(figsize=(7, 4))
            months = [row["month"] for row in month_table]
            amounts = [row["amount"] for row in month_table]
            ax_monthly.bar(months, amounts, color="#ff7f0e")
            ax_monthly.set_title("Monthly Expenses")
            ax_monthly.tick_params(axis="x", rotation=35)
            charts["monthly_expenses"] = fig_monthly

        if budget_info:
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.bar(["Spent", "Remaining"], [budget_info["spent"], max(0, budget_info["budget_limit"] - budget_info["spent"])], color=["#d62728", "#17becf"])
            ax4.set_title("Budget Usage")
            charts["budget_usage"] = fig4

        if investments["by_asset"]:
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.pie(investments["by_asset"].values(), labels=investments["by_asset"].keys(), autopct="%1.1f%%")
            ax5.set_title("Investment Allocation")
            charts["investment_allocation"] = fig5

        return charts

    def monthly_expense_table(self, user_id: int) -> list[dict[str, Any]]:
        expenses = pd.DataFrame(self.repository.get_expenses(user_id=user_id))
        if expenses.empty:
            return []
        expenses["month"] = pd.to_datetime(expenses["date"]).dt.strftime("%Y-%m")
        grouped = expenses.groupby("month")["amount"].sum().reset_index()
        return grouped.to_dict(orient="records")

    def emergency_fund(self, user_id: int) -> dict[str, float]:
        goals = self.goal_progress(user_id)
        ef = [g for g in goals if g["goal_name"] == "Emergency Fund"]
        if not ef:
            return {"emergency_fund": 0.0}
        return {"emergency_fund": float(ef[0]["current_amount"])}

    def personalized_recommendations(self, user_id: int, month: str | None = None) -> list[str]:
        recs: list[str] = []
        totals = self.repository.calculate_totals(user_id=user_id, month=month)
        spending = self.monthly_spending(user_id=user_id, month=month)
        top_category = ""
        top_value = 0.0
        for category, value in spending["by_category"].items():
            if value > top_value:
                top_category = category
                top_value = float(value)

        if totals["savings_rate"] < 20:
            recs.append("Increase monthly savings to at least 20 percent by reducing discretionary spending.")
        if top_category:
            recs.append(f"Set a category cap for {top_category}; it is currently your largest spend bucket.")
        if totals["total_expense"] > totals["total_income"] * 0.85:
            recs.append("Your expenses are too close to income; build a larger monthly buffer.")
        if not recs:
            recs.append("Your finances look balanced. Continue investing surplus cash consistently.")

        return recs
