"""Coca-Cola (KO) and Visa (V) betas vs. the S&P 500 (^GSPC).

Daily unadjusted closes, 2016-06-30 to 2026-06-30, rounded to 2 decimals.
Returns r_t = (p_t - p_{t-1}) / p_{t-1} * 100.
Regression: Return_Stock_i = a + b * Return_SPX_i + e_i
"""
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

TICKERS = {"KO": "KO", "V": "V", "SPX": "^GSPC"}
START, END = "2016-06-30", "2026-07-01"  # yfinance end is exclusive

raw = yf.download(list(TICKERS.values()), start=START, end=END,
                  auto_adjust=False, progress=False)["Close"]
prices = raw.rename(columns={v: k for k, v in TICKERS.items()})[list(TICKERS)]
prices = prices.dropna()
# Verify the closes are prices to the cent (Yahoo returns float noise like 45.369998931884766)
print("Max deviation from 2-decimal rounding:", (prices - prices.round(2)).abs().max().max())
prices = prices.round(2)
prices.to_csv("prices.csv")

returns = (prices.pct_change() * 100).dropna()
returns.to_csv("returns.csv")

print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
print(f"Returns: {len(returns)} observations\n")
for stock in ["KO", "V"]:
    model = sm.OLS(returns[stock], sm.add_constant(returns["SPX"])).fit()
    print(f"===== {stock} on SPX =====")
    print(model.summary().tables[1])
    print(f"beta (b) = {model.params['SPX']:.4f}, alpha (a) = {model.params['const']:.4f}, "
          f"R^2 = {model.rsquared:.4f}\n")
