#Project 1
# Problem 1 
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

funds = ["FSPTX","FSELX","FSCSX","FSPHX","FBIOX","FSMEX","FPHAX",
         "FIDSX","FSRBX","FSPCX","FSENX","FSDAX","FSRFX","FSCPX",
         "FSAVX","FSRPX","FDFAX","FSTCX","FWRLX"]
tickers = funds + ["SPY", "XLK"]

prices = yf.download(tickers, start="2015-01-01", end="2026-10-01",
                     auto_adjust=True)["Close"]
rets = np.log(prices / prices.shift(1)).dropna()
R = rets[funds]

expense = pd.Series({"FSPTX":0.61,"FSELX":0.60,"FSCSX":0.61,"FSPHX":0.62,
    "FBIOX":0.62,"FSMEX":0.62,"FPHAX":0.67,"FIDSX":0.68,"FSRBX":0.69,
    "FSPCX":0.68,"FSENX":0.65,"FSDAX":0.63,"FSRFX":0.68,"FSCPX":0.69,
    "FSAVX":0.78,"FSRPX":0.63,"FDFAX":0.68,"FSTCX":0.74,"FWRLX":0.71}) / 100

# 1.1
mu = R.mean() * 252
sigma = R.std() * np.sqrt(252)
SR = mu / sigma
abs_perf = pd.DataFrame({"Mean": mu, "Volatility": sigma, "SR": SR})

def summary(df):
    return pd.DataFrame({
        "Min": df.min(),
        "Q1": df.quantile(0.25),
        "Mean": df.mean(),
        "Median": df.median(),
        "Q3": df.quantile(0.75),
        "Max": df.max(),
    }).T

table_1_1 = summary(abs_perf)
print(table_1_1.round(4))

# 1.2
M = rets["SPY"]
mu_spy = M.mean() * 252
sig_spy = M.std() * np.sqrt(252)

plt.figure(figsize=(8, 6))
plt.scatter(sigma, mu, label="Funds")
for t in funds:
    plt.annotate(t, (sigma[t], mu[t]), fontsize=8)
plt.scatter(sig_spy, mu_spy, color="red", marker="*", s=250, label="SPY")
plt.xlabel("Annual volatility")
plt.ylabel("Annual mean return")
plt.title("Mean vs Volatility (2015-2026)")
plt.legend()
plt.grid(True)
plt.savefig(r"C:\Users\ricca\OneDrive\Desktop\FRM\fig_1_2.png", dpi=300)


beat = abs_perf[abs_perf["Mean"] > mu_spy]
print("SPY mean:", round(mu_spy, 4), "SPY vol:", round(sig_spy, 4))
print(beat[["Mean", "Volatility"]].round(4))

# 1.3
def relative(R, M):
    beta = R.apply(lambda x: x.cov(M)) / M.var()
    alpha = (R.mean() - beta * M.mean()) * 252
    tr = (R.mean() * 252) / beta
    te = R.sub(M, axis=0).std() * np.sqrt(252)
    ir = (R.mean() * 252 - M.mean() * 252) / te
    return pd.DataFrame({"Alpha": alpha, "Beta": beta, "TR": tr,
                         "TE": te, "IR": ir})

# 1.3a
rel_spy = relative(R, M)
table_1_3a = summary(rel_spy)
print(table_1_3a.round(4))

# 1.3b
top5 = rel_spy.sort_values("Alpha", ascending=False).head(5).index
table_1_3b = pd.concat([abs_perf.loc[top5], rel_spy.loc[top5]], axis=1)
print(table_1_3b.round(4))

# 1.4a
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(expense[funds], mu[funds])
ax[0].set_xlabel("Expense ratio")
ax[0].set_ylabel("Annual mean return")
ax[0].set_title("Mean return vs Expense ratio")
ax[0].grid(True)
ax[1].scatter(expense[funds], rel_spy["Alpha"][funds])
ax[1].set_xlabel("Expense ratio")
ax[1].set_ylabel("Jensen's alpha (SPY)")
ax[1].set_title("Alpha vs Expense ratio")
ax[1].grid(True)
for t in funds:
    ax[0].annotate(t, (expense[t], mu[t]), fontsize=7)
    ax[1].annotate(t, (expense[t], rel_spy["Alpha"][t]), fontsize=7)
plt.tight_layout()
plt.savefig(r"C:\Users\ricca\OneDrive\Desktop\FRM\fig_1_4.png", dpi=300)

# 1.4b
print("Corr(ER, Mean):", round(expense[funds].corr(mu[funds]), 4))
print("Corr(ER, Alpha):", round(expense[funds].corr(rel_spy["Alpha"][funds]), 4))

# 1.5a
X = rets["XLK"]
rel_xlk = relative(R, X)
table_1_5a = summary(rel_xlk)
print(table_1_5a.round(4))

# 1.5b
compare = pd.DataFrame({
    "Alpha_SPY": rel_spy["Alpha"],
    "Rank_SPY": rel_spy["Alpha"].rank(ascending=False).astype(int),
    "Alpha_XLK": rel_xlk["Alpha"],
    "Rank_XLK": rel_xlk["Alpha"].rank(ascending=False).astype(int),
}).sort_values("Rank_SPY")
print(compare.round(4))