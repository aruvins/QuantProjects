import numpy as np
import pandas as pd
from numpy_financial import irr

class LBOModel:
    def __init__(self, purchase_price, debt_ratio, projections, exit_multiple, interest_rate):
        self.purchase_price = purchase_price
        self.debt_ratio = debt_ratio
        self.equity_ratio = 1 - debt_ratio
        self.exit_multiple = exit_multiple
        self.interest_rate = interest_rate
        self.projections = projections

        self.debt = purchase_price * debt_ratio
        self.equity = purchase_price * self.equity_ratio

    def build_operating_model(self):
        revenue = self.projections['revenue']
        EBITDA_margin = self.projections['EBITDA_margin']
        CapEx = self.projections['CapEx']
        NWC_change = self.projections['NWC_change']

        EBITDA = [r * EBITDA_margin for r in revenue]
        Depreciation = [c * 0.8 for c in CapEx]  # assume D&A is 80% of CapEx
        EBIT = [e - d for e, d in zip(EBITDA, Depreciation)]

        interest_expense = []
        debt_balance = self.debt
        FCF = []
        debt_remaining = []

        for i in range(len(revenue)):
            interest = debt_balance * self.interest_rate
            interest_expense.append(interest)

            tax = max(0, EBIT[i] - interest) * 0.25  # 25% tax rate
            FCF_i = EBITDA[i] - interest - tax - CapEx[i] - NWC_change[i]
            FCF.append(FCF_i)

            principal_payment = min(debt_balance, max(0, FCF_i))
            debt_balance -= principal_payment
            debt_remaining.append(debt_balance)

        self.EBITDA = EBITDA
        self.Debt_Remaining = debt_remaining
        self.FCF = FCF
        self.interest_expense = interest_expense

    def calculate_exit(self):
        final_EBITDA = self.EBITDA[-1]
        exit_value = final_EBITDA * self.exit_multiple
        remaining_debt = self.Debt_Remaining[-1]
        equity_value = exit_value - remaining_debt
        return equity_value

    def calculate_IRR(self):
        cash_flows = [-self.equity]
        for fcf in self.FCF:
            equity_cf = max(0, fcf)
            cash_flows.append(equity_cf)
        cash_flows[-1] += self.calculate_exit()
        return irr(cash_flows), cash_flows

# Sample usage
data = {
    'years': [2025, 2026, 2027, 2028, 2029],
    'revenue': [100, 110, 121, 133, 146],
    'EBITDA_margin': 0.25,
    'CapEx': [5, 5, 5, 5, 5],
    'NWC_change': [1, 1, 1, 1, 1]
}

model = LBOModel(purchase_price=100, debt_ratio=0.6, projections=data, exit_multiple=8, interest_rate=0.08)
model.build_operating_model()
irr_result, cash_flows = model.calculate_IRR()

print("Cash Flows to Equity:", cash_flows)
print("IRR:", round(irr_result * 100, 2), "%")
