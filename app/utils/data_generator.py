from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.config import config
from app.utils.logger import logger


CATEGORIES = [
    "Food",
    "Rent",
    "Travel",
    "Medical",
    "Entertainment",
    "Education",
    "Shopping",
    "Insurance",
    "Bills",
]

USERS = [
    ("Girish", 30, "Bengaluru"),
    ("Asha Verma", 27, "Mumbai"),
    ("Rohan Iyer", 35, "Pune"),
    ("Priya Nair", 29, "Kochi"),
    ("Arjun Mehta", 41, "Ahmedabad"),
    ("Sneha Reddy", 33, "Hyderabad"),
    ("Vikram Singh", 38, "Delhi"),
    ("Kavya Rao", 25, "Mysuru"),
    ("Rahul Sharma", 45, "Jaipur"),
    ("Meera Joshi", 31, "Chennai"),
]


def _random_date_in_last_n_days(n_days: int) -> date:
    today = date.today()
    return today - timedelta(days=random.randint(0, n_days))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_if_missing(path: Path, df: pd.DataFrame, overwrite: bool = False) -> None:
    _ensure_parent(path)
    if overwrite or not path.exists():
        df.to_csv(path, index=False)
        logger.info("Generated synthetic file: %s", path)


def generate_synthetic_data_if_missing(num_users: int = 10, num_months: int = 24, overwrite: bool = False) -> None:
    random.seed(42)
    storage = Path(config.storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    Path(config.reports_dir).mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame(
        [
            {"user_id": i + 1, "name": name, "age": age, "city": city}
            for i, (name, age, city) in enumerate(USERS[:num_users])
        ]
    )

    income_rows: list[dict] = []
    expenses_rows: list[dict] = []
    transactions_rows: list[dict] = []
    budgets_rows: list[dict] = []
    goals_rows: list[dict] = []
    investments_rows: list[dict] = []
    savings_rows: list[dict] = []

    month_starts = pd.date_range(end=pd.Timestamp.today(), periods=num_months, freq="MS")

    txn_id = 1
    for user_id in users["user_id"]:
        for month in month_starts:
            monthly_income = random.randint(70_000, 140_000)
            income_rows.append(
                {
                    "income_id": len(income_rows) + 1,
                    "user_id": user_id,
                    "date": month.strftime("%Y-%m-%d"),
                    "source": "Salary",
                    "amount": monthly_income,
                }
            )

            total_expense = 0
            for category in CATEGORIES:
                if category == "Rent":
                    amount = random.randint(15_000, 30_000)
                else:
                    amount = random.randint(2_500, 14_000)
                total_expense += amount

                expenses_rows.append(
                    {
                        "expense_id": len(expenses_rows) + 1,
                        "user_id": user_id,
                        "date": month.strftime("%Y-%m-%d"),
                        "category": category,
                        "amount": amount,
                    }
                )

                for _ in range(2):
                    txn_amount = max(200, int(amount / 2 + random.randint(-800, 1000)))
                    txn_date = _random_date_in_last_n_days(num_months * 30)
                    transactions_rows.append(
                        {
                            "transaction_id": txn_id,
                            "user_id": user_id,
                            "date": txn_date.strftime("%Y-%m-%d"),
                            "type": "expense",
                            "category": category,
                            "description": f"{category} payment",
                            "amount": txn_amount,
                        }
                    )
                    txn_id += 1

            transactions_rows.append(
                {
                    "transaction_id": txn_id,
                    "user_id": user_id,
                    "date": month.strftime("%Y-%m-%d"),
                    "type": "income",
                    "category": "Salary",
                    "description": "Monthly salary",
                    "amount": monthly_income,
                }
            )
            txn_id += 1

            savings_estimate = max(1_000, monthly_income - total_expense)
            savings_rows.append(
                {
                    "saving_id": len(savings_rows) + 1,
                    "user_id": user_id,
                    "month": month.strftime("%Y-%m"),
                    "amount": savings_estimate,
                }
            )
            budgets_rows.append(
                {
                    "budget_id": len(budgets_rows) + 1,
                    "user_id": user_id,
                    "month": month.strftime("%Y-%m"),
                    "budget_limit": int(total_expense * 1.05),
                }
            )

            goals_rows.append(
                {
                    "goal_id": len(goals_rows) + 1,
                    "user_id": user_id,
                    "goal_name": "Emergency Fund",
                    "target_amount": 300_000,
                    "current_amount": min(300_000, savings_estimate * random.randint(2, 6)),
                    "target_date": (month + pd.DateOffset(months=12)).strftime("%Y-%m-%d"),
                }
            )

        investments_rows.extend(
            [
                {
                    "investment_id": len(investments_rows) + 1,
                    "user_id": user_id,
                    "asset_type": "Mutual Fund",
                    "amount": random.randint(120_000, 400_000),
                },
                {
                    "investment_id": len(investments_rows) + 2,
                    "user_id": user_id,
                    "asset_type": "Stocks",
                    "amount": random.randint(80_000, 350_000),
                },
                {
                    "investment_id": len(investments_rows) + 3,
                    "user_id": user_id,
                    "asset_type": "Fixed Deposit",
                    "amount": random.randint(70_000, 200_000),
                },
            ]
        )

    _write_if_missing(storage / "users.csv", users, overwrite)
    _write_if_missing(storage / "income.csv", pd.DataFrame(income_rows), overwrite)
    _write_if_missing(storage / "expenses.csv", pd.DataFrame(expenses_rows), overwrite)
    _write_if_missing(storage / "transactions.csv", pd.DataFrame(transactions_rows), overwrite)
    _write_if_missing(storage / "budgets.csv", pd.DataFrame(budgets_rows), overwrite)
    _write_if_missing(storage / "savings.csv", pd.DataFrame(savings_rows), overwrite)
    _write_if_missing(storage / "goals.csv", pd.DataFrame(goals_rows), overwrite)
    _write_if_missing(storage / "investments.csv", pd.DataFrame(investments_rows), overwrite)

    history_path = Path(config.history_file)
    _ensure_parent(history_path)
    if not history_path.exists():
        history_path.write_text("[]", encoding="utf-8")
        logger.info("Created conversation history file: %s", history_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate synthetic finance CSV data.")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()
    generate_synthetic_data_if_missing(num_users=args.users, num_months=args.months, overwrite=True)
