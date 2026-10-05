"""Coca-Cola (KO) and Visa (V) betas against the S&P 500 (SPX).

Downloads daily unadjusted closing prices from Yahoo Finance for
2016-06-30 through 2026-06-30, computes daily percentage returns
r_t = (P_t - P_{t-1}) / P_{t-1} * 100, and estimates
Return_Stock = a + b * Return_SPX + e by OLS.
"""

import pandas as pd
import statsmodels.api as sm
import yfinance as yf

TICKERS = {"KO": "KO", "V": "V", "SPX": "^GSPC"}
START, END = "2016-06-30", "2026-07-01"  # yfinance's end date is exclusive

raw = yf.download(list(TICKERS.values()), start=START, end=END,
                  auto_adjust=False, progress=False)["Close"]
prices = raw.rename(columns={v: k for k, v in TICKERS.items()})[list(TICKERS)]
prices = prices.dropna().round(2)  # prices are quoted to two decimals
prices.to_csv("prices.csv")

# Verify every price has at most two decimal places
assert ((prices * 100).round(6) % 1 == 0).all().all(), "prices not at 2 decimals"

returns = prices.pct_change().mul(100).dropna()
returns.to_csv("returns.csv")

print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, "
      f"{len(prices)} days; {len(returns)} daily returns\n")

for stock in ["KO", "V"]:
    model = sm.OLS(returns[stock], sm.add_constant(returns["SPX"])).fit()
    a, b = model.params["const"], model.params["SPX"]
    print(f"=== {stock} ===")
    print(f"Return_{stock} = {a:.4f} + {b:.4f} x Return_SPX   "
          f"(R^2 = {model.rsquared:.4f}, n = {int(model.nobs)})")
    print(model.summary().tables[1], "\n")
