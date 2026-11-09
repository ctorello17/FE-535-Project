# FE 535 Project 1: Performance Evaluation, Mean-Variance Frontier, Risk Modeling and Dynamic Games

Team project for **FE 535 Introduction to Financial Risk Management**, Stevens Institute of Technology, Fall 2026.

All code is in one notebook, `project1.ipynb`, organized as one code cell per question part. Written answers sit in markdown cells directly below the cell they refer to.

## Contents

| Problem | Topic | Points |
|---|---|---|
| 1 | Performance summary of 19 Fidelity Select funds (absolute and relative, SPY and XLK benchmarks) | 65 |
| 2 | Mean-variance efficient frontier (in-sample, out-of-sample, diagonal-covariance shrinkage) | 75 |
| 3 | Risk modeling of SPY (GBM calibration, Monte Carlo, VaR, model risk) | 75 |
| 4 | Dynamic optimal stopping game solved by least-squares Monte Carlo (Longstaff-Schwartz) | 55 |

## Repository files

| File | Description |
|---|---|
| `project1.ipynb` | **Final source code** for all four problems |
| `FE535_Project1_Report.docx` | Written report (submitted as PDF) |

## Setup

Python 3.10 or newer (developed on 3.12).

```bash
pip install yfinance pandas numpy matplotlib scipy jupyter
jupyter notebook project1.ipynb
```

Run the notebook from top to bottom with **Restart → Run All**. Cells depend on variables from earlier cells, so running them out of order can give errors or wrong results.

* An internet connection is required: prices are downloaded from Yahoo Finance.
* A full run takes about a minute; Problem 4.3 (100-stage game) is the slowest cell.
* Random seeds are fixed (535 for Problem 3; 2026 and 2027 for Problem 4), so simulation results are reproducible.
* Yahoo Finance occasionally revises historical data, so figures can differ slightly depending on the download date.

## Data

* **Source:** Yahoo Finance via `yfinance`, using adjusted closing prices (`Adj Close`, `auto_adjust=False`), so dividends and fund distributions are included.
* **Returns:** log returns, r_t = ln(P_t / P_{t−1}). The risk-free rate is zero throughout, as the handout specifies.

| Problem | Tickers | Frequency | Sample |
|---|---|---|---|
| 1 | 19 funds (FSPTX, FSELX, FSCSX, FSPHX, FBIOX, FSMEX, FPHAX, FIDSX, FSRBX, FSPCX, FSENX, FSDAX, FSRFX, FSCPX, FSAVX, FSRPX, FDFAX, FSTCX, FWRLX), SPY, XLK | Daily | Jan 2015 – Sep 2026 (2,952 returns) |
| 2 | FSPTX, FIDSX, FSCPX | Monthly | Feb 2015 – Sep 2026, split 70 / 70 months |
| 3 | SPY | Monthly | Jan 1994 – Sep 2026 (393 prices, 392 returns) |
| 4 | None (simulated X_t ~ U(0, 10)) | – | – |

SPY and XLK are kept in separate series (`market_log_returns`, `xlk_log_returns`) so they are never counted among the 19 funds.

## Methodology

### Problem 1: Performance summary

Daily figures are annualized with 252 trading days. For fund *i* and benchmark *M*:

| Measure | Definition |
|---|---|
| Mean return | 252 · mean(r_i) |
| Volatility | √252 · sd(r_i) |
| Sharpe ratio | mean / volatility |
| Beta | Cov(r_i, r_M) / Var(r_M) |
| Jensen's alpha | μ_i − β · μ_M |
| Treynor ratio | μ_i / β |
| Tracking error | √252 · sd(r_i − r_M) |
| Information ratio | (μ_i − μ_M) / tracking error |

Each summary table reports min, Q1, mean, median, Q3 and max across the 19 funds. Part 5 repeats the relative measures with XLK as the benchmark.

### Problem 2: Mean-variance efficient frontier

Monthly returns are split into two equal halves. For each half we estimate μ and Σ and form the two funds in closed form (no optimization package):

* GMV portfolio: w₀ = Σ⁻¹1 / (1ᵀΣ⁻¹1)
* Aggressive fund: w₁ = Bμ = Σ⁻¹μ − (1ᵀΣ⁻¹μ) · w₀
* Optimal portfolio for risk aversion A: w(A) = w₀ + (1/A) · w₁, with A = 1, …, 100

The Sharpe (tangency) portfolio is w(A_T) with A_T = 1ᵀΣ⁻¹μ. Three frontiers are evaluated with period-2 data: utopian (built with μ₂, Σ₂), out-of-sample (built with μ₁, Σ₁) and shrinkage (built with μ₁ and a diagonal Σ₁). Plots are annualized (mean × 12, volatility × √12).

### Problem 3: Risk modeling

* GBM calibration with Δt = 1/12: σ̂ = sd(r)/√Δt and μ̂ = mean(r)/Δt + σ̂²/2.
* 1,000 Monte Carlo paths over 392 months, starting at the Jan 1994 close, checked against E[S_T] = S₀e^{μT} and Var[S_T] = S₀²e^{2μT}(e^{σ²T} − 1).
* Closest simulated path to the actual SPY path by L2 (second-norm) distance.
* 99% VaR defined as mean − 1st percentile, following the handout.
* σ sweep from 0.10 to 0.50, with μ recalibrated so that μ − σ²/2 matches the data's mean log return. Monte Carlo results are compared with the closed form VaR(σ) = 100·S₀·(e^{mT + σ²T/2} − e^{mT + zσ√T}).
* Historical vs. parametric (simulated) monthly VaR, with skewness, kurtosis and a density comparison.

### Problem 4: Dynamic game via LSMC

Backward induction on 1,000,000 simulated paths. At each stage the continuation value is fitted by least squares on the basis [1, x, x²], and the player stops when X_t exceeds the fitted continuation value. The fitted rules are then re-evaluated on 4,000,000 fresh paths to remove in-sample bias, and compared with the analytical solution U_{k+1} = U_k²/20 + 5, starting from U₁ = 5.

## Headline results

**Problem 1 (benchmark SPY).** Only FSELX, FSPTX and FSCSX beat SPY's 12.9% mean return, each with higher volatility than SPY's 17.6%. The highest alphas are FSELX (6.4%) and FSPTX (4.7%). Expense ratios are negatively correlated with mean return (−0.55) and with alpha (−0.54). Against XLK, the technology funds' alphas largely disappear, so measured alpha depends heavily on the benchmark.

**Problem 2.** The two halves give very different weights: the GMV portfolio shifts its largest holding from FSCPX (66%) to FIDSX (63%). The utopian Sharpe portfolio reaches an annualized Sharpe ratio of 1.40, but the period-1 Sharpe portfolio achieves only 0.67 when evaluated in period 2. The diagonal-covariance portfolios are closer to the true frontier at low risk (GMV Sharpe 0.70 vs. 0.51 out-of-sample) but give up return at higher risk.

**Problem 3.** μ̂ = 0.113 and σ̂ = 0.150 per year. The 99% VaR of the 100-share position is about $96,000 (mean − 1st percentile), and even the 1st-percentile path ends with a gain. VaR rises non-linearly with σ, from about $68,000 at σ = 0.10 to about $2.6 million at σ = 0.50. Historical monthly VaR (0.120) exceeds the GBM VaR (0.100) because actual returns are negatively skewed (−0.72) and fat-tailed (excess kurtosis 1.27).

**Problem 4.** LSMC matches the analytical solution closely in all three games:

| Game | LSMC value (out-of-sample) | Analytical value | Threshold at t = 0 |
|---|---|---|---|
| 2-stage (t = 0, 1, 2) | 6.9526 | 6.9531 | 6.25 |
| 10-stage (t = 0, …, 10) | 8.7078 | 8.7075 | 8.61 |
| 100-stage (t = 0, …, 100) | 9.8139 | 9.8139 | 9.81 |

The stopping threshold falls as the game approaches its final stage, and the value of the option to wait grows with the number of stages at a diminishing rate.

## Team

Christopher Torello, Jinhong Chen, Riccardo Baccelliere
