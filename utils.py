"""
utils.py — Shared helper functions for factor timing analysis.

Used across notebooks 01–04.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Portfolio metrics ──────────────────────────────────────────────────────────

def sharpe(returns, annualize=True):
    """Annualized Sharpe ratio (assumes returns are already in excess of rf)."""
    r = returns.dropna()
    if r.std() < 1e-10:
        return np.nan
    s = r.mean() / r.std()
    return s * np.sqrt(12) if annualize else s


def make_ewp(df, factor_cols, start=None, end=None):
    """
    Equal-weight portfolio: row-wise mean of factor_cols each month.
    NaNs are ignored (graceful degradation for early months with missing factors).
    Returns a monthly return Series indexed by date.
    """
    sub = df[factor_cols]
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]
    return sub.mean(axis=1)


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown of a return series."""
    cum = (1 + returns.dropna()).cumprod()
    roll_max = cum.cummax()
    return ((cum - roll_max) / roll_max).min()


def alpha_tstat(portfolio_rets, benchmark_rets, nw_lags=6):
    """
    OLS alpha of portfolio_rets ~ benchmark_rets.
    t-stat uses Newey-West HAC standard errors.
    Returns (annualized_alpha, t_stat).
    """
    aligned = pd.concat([portfolio_rets, benchmark_rets], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan, np.nan
    y = aligned.iloc[:, 0].values
    x = sm.add_constant(aligned.iloc[:, 1].values)
    res = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    return res.params[0] * 12, res.tvalues[0]


def info_ratio(portfolio_rets, benchmark_rets):
    """Annualized information ratio = annualized active return / tracking error."""
    active = (portfolio_rets - benchmark_rets).dropna()
    if active.std() < 1e-10:
        return np.nan
    return active.mean() / active.std() * np.sqrt(12)


def factor_weight_turnover(wt_df):
    """Average monthly L1 turnover of the factor weight vector."""
    return wt_df.diff().abs().sum(axis=1).mean()


def evaluate(model_name, oos_rets, oos_wts, benchmark_rets):
    """Compute a full set of OOS performance metrics for a given model."""
    r = oos_rets.dropna()
    bench = benchmark_rets.reindex(r.index).dropna()
    r = r.reindex(bench.index)

    ann_alpha, t_alpha = alpha_tstat(r, bench)

    return {
        "Model":          model_name,
        "N months":       len(r),
        "Ann. Sharpe":    round(sharpe(r), 3),
        "EWP Sharpe":     round(sharpe(bench), 3),
        "Alpha (ann.)":   round(ann_alpha, 4) if not np.isnan(ann_alpha) else np.nan,
        "t(alpha)":       round(t_alpha, 2)   if not np.isnan(t_alpha)   else np.nan,
        "Info Ratio":     round(info_ratio(r, bench), 3),
        "Max Drawdown":   round(max_drawdown(r), 4),
        "Wt Turnover":    round(factor_weight_turnover(oos_wts), 4),
    }


# ── ML helpers ─────────────────────────────────────────────────────────────────

def information_coefficient(y_true, y_pred):
    """
    Rank correlation (Spearman-like IC) between predicted and actual returns.
    Returns NaN if either series has zero variance or fewer than 2 observations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return np.nan
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return np.nan
    return np.corrcoef(y_pred, y_true)[0, 1]
