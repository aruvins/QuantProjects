import numpy as np
import pandas as pd

class DCFModel:
    def __init__(self, projections, discount_rate, terminal_growth):
        self.projections = projections
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth

    def calculate_fcf(self):
        revenue = self.projections['revenue']
        EBITDA_margin = self.projections['EBITDA_margin']
        CapEx = self.projections['CapEx']
        DnA = self.projections['D&A']
        NWC_change = self.projections['NWC_change']
        tax_rate = self.projections['tax_rate']

        EBITDA = [r * EBITDA_margin for r in revenue]
        EBIT = [e - d for e, d in zip(EBITDA, DnA)]
        Taxes = [max(0, e) * tax_rate for e in EBIT]
        FCF = [e - t + d - c - n for e, t, d, c, n in zip(EBITDA, Taxes, DnA, CapEx, NWC_change)]

        self.EBITDA = EBITDA
        self.FCF = FCF
        return FCF

    def calculate_terminal_value(self):
        final_fcf = self.FCF[-1]
        TV = final_fcf * (1 + self.terminal_growth) / (self.discount_rate - self.terminal_growth)
        return TV

    def calculate_present_value(self):
        FCF = self.calculate_fcf()
        PV_FCF = [fcf / (1 + self.discount_rate) ** (i + 1) for i, fcf in enumerate(FCF)]

        TV = self.calculate_terminal_value()
        PV_TV = TV / (1 + self.discount_rate) ** len(FCF)

        enterprise_value = sum(PV_FCF) + PV_TV
        return enterprise_value, PV_FCF, PV_TV

# Sample usage
data = {
    'years': [2025, 2026, 2027, 2028, 2029],
    'revenue': [100, 110, 121, 133, 146],
    'EBITDA_margin': 0.3,
    'CapEx': [5, 6, 6, 7, 7],
    'D&A': [3, 3, 3, 4, 4],
    'NWC_change': [1, 1, 1, 1, 1],
    'tax_rate': 0.25
}

model = DCFModel(projections=data, discount_rate=0.10, terminal_growth=0.03)
enterprise_value, PV_FCF, PV_TV = model.calculate_present_value()

print("Present Value of FCFs:", PV_FCF)
print("Present Value of Terminal Value:", round(PV_TV, 2))
print("Enterprise Value:", round(enterprise_value, 2))