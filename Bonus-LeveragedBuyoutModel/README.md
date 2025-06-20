# Leveraged Buyout (LBO) Model in Python

This project is a simplified Leveraged Buyout (LBO) model built in Python. It's designed for educational purposes and to demonstrate key LBO mechanics commonly used in private equity.

## What It Does

* Projects future cash flows of a company using revenue, EBITDA margin, CapEx, and working capital assumptions.
* Models debt repayment from Free Cash Flow (FCF).
* Calculates the value of the business at exit using an EBITDA exit multiple.
* Computes the internal rate of return (IRR) for equity investors.

## Key Assumptions

* Purchase price: \$100 million (example)
* Debt ratio: 60% (leveraged capital structure)
* EBITDA margin: constant across projection period
* Exit multiple: 8x EBITDA
* Interest rate on debt: 8%
* Taxes: 25% applied to earnings before tax (EBT)

## Outputs

* Annual Free Cash Flows to Equity (after interest, taxes, CapEx, and working capital changes)
* Ending equity value at exit (exit enterprise value minus debt remaining)
* Equity IRR based on cash inflows/outflows

## Sample Usage

The model can be run by updating the `data` dictionary and initializing the `LBOModel` class:

```python
model = LBOModel(purchase_price=100, debt_ratio=0.6, projections=data, exit_multiple=8, interest_rate=0.08)
model.build_operating_model()
irr_result, cash_flows = model.calculate_IRR()
```

## Requirements

* Python 3.x
* pandas
* numpy
* numpy-financial (`pip install numpy-financial`)

## Notes

* This model assumes all FCF is used to pay down debt each year.
* No consideration for management fees, transaction fees, dividend recapitalizations, or multiple tranches of debt.

## Future Improvements

* Add sensitivity analysis for exit multiples and leverage
* Model more complex capital structures
* Build visualization/dashboard using matplotlib or Plotly
* Add Excel import/export capability
