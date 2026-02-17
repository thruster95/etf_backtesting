import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backtest_rebel import (
    fetch_range,
    make_price_panel,
    map_monthly_weights_to_daily_rebal_days,
    backtest_monthly_rebal_with_dynamic_weights,
    buy_and_hold,
    summarize,
)


def compute_stock_bond_momentum_weights_monthly(
    px: pd.DataFrame,
    lookback_months: int,
) -> pd.DataFrame:
    """
    ETF1(주식)의 단일 n개월 모멘텀으로 월말 타깃 비중 생성.
    - ETF1 현재 월말 종가 > n개월 전 월말 종가 -> ETF1 100%, ETF2 0%
    - ETF1 현재 월말 종가 <= n개월 전 월말 종가 -> ETF1 0%, ETF2 100%
    - 결측은 ETF2 100%
    """
    mpx = px.resample("ME").last()
    stock_now = mpx["ETF1"]
    stock_prev = stock_now.shift(lookback_months)

    w = pd.DataFrame(index=mpx.index, columns=["ETF1", "ETF2"], data=0.0)
    w["ETF2"] = 1.0
    w.loc[stock_now > stock_prev, ["ETF1", "ETF2"]] = [1.0, 0.0]
    w.loc[stock_now <= stock_prev, ["ETF1", "ETF2"]] = [0.0, 1.0]

    tie_or_nan = stock_now.isna() | stock_prev.isna()
    w.loc[tie_or_nan, ["ETF1", "ETF2"]] = [0.0, 1.0]
    return w


def compute_average_momentum_score_weights_monthly(
    px: pd.DataFrame,
    score_months: int = 12,
) -> pd.DataFrame:
    """
    평균 모멘텀 스코어(1~score_months) 기반 월말 타깃 비중 생성.

    ETF1 스코어 정의:
      score = (I(P_t > P_{t-1}) + I(P_t > P_{t-2}) + ... + I(P_t > P_{t-score_months})) / score_months

    비중:
      w_etf1 = score
      w_etf2 = 1 - score

    주의: 과거 가격이 없어 비교 불가한 항은 0점으로 처리(분모는 고정 score_months).
    """
    mpx = px.resample("ME").last()
    stock_now = mpx["ETF1"]

    score_sum = pd.Series(0.0, index=mpx.index)
    for k in range(1, score_months + 1):
        signal = (stock_now > stock_now.shift(k)).fillna(False).astype(float)
        score_sum += signal

    score = (score_sum / float(score_months)).clip(lower=0.0, upper=1.0)

    w = pd.DataFrame(index=mpx.index, columns=["ETF1", "ETF2"], data=0.0)
    w["ETF1"] = score
    w["ETF2"] = 1.0 - score
    return w


def calc_ann_vol_from_daily_equity(equity: pd.Series) -> float:
    dr = equity.pct_change().dropna()
    if len(dr) < 2:
        return np.nan
    return float(dr.std(ddof=1) * np.sqrt(252))


def run_momentum_lookback_sweep(
    etf1_code: str,
    etf2_code: str,
    start: str,
    end: str,
    upd_stkpc_tp: str = "1",
    lookback_range=range(1, 13),
    score_months: int = 12,
):
    max_lb = max(max(lookback_range), score_months)

    start_main = pd.to_datetime(start)
    start_hist = (start_main - pd.DateOffset(months=max_lb + 2)).strftime("%Y-%m-%d")

    df1 = fetch_range(etf1_code, start_main.strftime("%Y-%m-%d"), end, upd_stkpc_tp=upd_stkpc_tp)
    df2 = fetch_range(etf2_code, start_main.strftime("%Y-%m-%d"), end, upd_stkpc_tp=upd_stkpc_tp)
    px = make_price_panel(df1, df2)

    df1_hist = fetch_range(etf1_code, start_hist, end, upd_stkpc_tp=upd_stkpc_tp)
    df2_hist = fetch_range(etf2_code, start_hist, end, upd_stkpc_tp=upd_stkpc_tp)
    px_hist = make_price_panel(df1_hist, df2_hist)

    weights_monthly = {}
    weights_rebal = {}
    equities = {}
    stat_rows = []

    for n in lookback_range:
        w_monthly = compute_stock_bond_momentum_weights_monthly(px_hist, lookback_months=n)
        w_rebal = map_monthly_weights_to_daily_rebal_days(px, w_monthly)
        name = f"Momentum {n}M"
        eq = backtest_monthly_rebal_with_dynamic_weights(px, w_rebal, initial=1.0, name=name)

        weights_monthly[n] = w_monthly
        weights_rebal[n] = w_rebal
        equities[n] = eq

        row = summarize(eq, name)
        row["Lookback(M)"] = n
        row["AnnVol(%)"] = round(calc_ann_vol_from_daily_equity(eq) * 100, 2)
        stat_rows.append(row)

    stats = pd.DataFrame(stat_rows).set_index("Lookback(M)").sort_index()

    # 평균 모멘텀 스코어 전략(1~score_months)
    w_score_monthly = compute_average_momentum_score_weights_monthly(px_hist, score_months=score_months)
    w_score_rebal = map_monthly_weights_to_daily_rebal_days(px, w_score_monthly)
    eq_score = backtest_monthly_rebal_with_dynamic_weights(
        px,
        w_score_rebal,
        initial=1.0,
        name=f"Avg Momentum Score 1~{score_months}M",
    )

    score_stats = summarize(eq_score, f"Avg Momentum Score 1~{score_months}M")
    score_stats["AnnVol(%)"] = round(calc_ann_vol_from_daily_equity(eq_score) * 100, 2)

    eq_stock = buy_and_hold(px["ETF1"], initial=1.0)
    eq_bond = buy_and_hold(px["ETF2"], initial=1.0)

    etf1_stats = summarize(eq_stock, f"Buy&Hold ETF1 ({etf1_code})")
    etf1_stats["AnnVol(%)"] = round(calc_ann_vol_from_daily_equity(eq_stock) * 100, 2)
    etf2_stats =summarize(eq_bond, f"Buy&Hold ETF2 ({etf2_code})")
    etf2_stats["AnnVol(%)"] = round(calc_ann_vol_from_daily_equity(eq_bond) * 100, 2)


    best_by_cagr = stats["CAGR(%)"].idxmax()
    best_by_nav = stats["Final(NAV)"].idxmax()

    print("\n[모멘텀 n(1~12) 성과 요약]")
    print(stats[["전략", "CAGR(%)", "MDD(%)", "AnnVol(%)", "Final(NAV)"]])
    print(f"\n[Best by CAGR] n={best_by_cagr}M, CAGR={stats.loc[best_by_cagr, 'CAGR(%)']}%")
    print(f"[Best by Final NAV] n={best_by_nav}M, Final(NAV)={stats.loc[best_by_nav, 'Final(NAV)']}")

    print("\n[평균 모멘텀 스코어 전략 성과]")
    print(pd.DataFrame([score_stats]).set_index("전략")[["CAGR(%)", "MDD(%)", "AnnVol(%)", "Final(NAV)"]])

    print("\n[BUY AND HOLD 전략 성과]")
    print(pd.DataFrame([etf1_stats]).set_index("전략")[["CAGR(%)", "MDD(%)", "AnnVol(%)", "Final(NAV)"]])
    print(pd.DataFrame([etf2_stats]).set_index("전략")[["CAGR(%)", "MDD(%)", "AnnVol(%)", "Final(NAV)"]])

    # 1) 모든 모멘텀 곡선 + 평균 모멘텀 스코어 + B&H
    plt.figure(figsize=(13, 6))
    for n in lookback_range:
        plt.plot(equities[n].index, equities[n].values, label=f"Mom {n}M", alpha=0.5)
    plt.plot(eq_score.index, eq_score.values, label=f"Avg Score 1~{score_months}M", linewidth=2.4)
    plt.plot(eq_stock.index, eq_stock.values, label=f"B&H ETF1 ({etf1_code})", linewidth=2.2, linestyle="--")
    plt.plot(eq_bond.index, eq_bond.values, label=f"B&H ETF2 ({etf2_code})", linewidth=2.2, linestyle=":")
    plt.title("Momentum Lookback Sweep + Average Momentum Score (Monthly Rebalancing)")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()

    # 2) CAGR 바 차트 + 평균 모멘텀 스코어 수평선
    plt.figure(figsize=(11, 4))
    plt.bar(stats.index.astype(str), stats["CAGR(%)"].values)
    plt.axhline(score_stats["CAGR(%)"], color="red", linestyle="--", label=f"Avg Score 1~{score_months}M")
    plt.title("CAGR by Momentum Lookback (Months)")
    plt.xlabel("Lookback (Months)")
    plt.ylabel("CAGR (%)")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) 베스트 n vs 평균 모멘텀 스코어 vs B&H 드로우다운
    eq_best = equities[best_by_cagr]
    dd_best = eq_best / eq_best.cummax() - 1.0
    dd_score = eq_score / eq_score.cummax() - 1.0
    dd_stock = eq_stock / eq_stock.cummax() - 1.0
    dd_bond = eq_bond / eq_bond.cummax() - 1.0

    plt.figure(figsize=(12, 4))
    plt.plot(dd_best.index, dd_best.values, label=f"Best Momentum {best_by_cagr}M")
    plt.plot(dd_score.index, dd_score.values, label=f"Avg Score 1~{score_months}M")
    plt.plot(dd_stock.index, dd_stock.values, label=f"B&H ETF1 ({etf1_code})")
    plt.plot(dd_bond.index, dd_bond.values, label=f"B&H ETF2 ({etf2_code})")
    plt.title("Drawdown: Best Momentum / Avg Score / Buy&Hold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "prices": px,
        "weights_monthly": weights_monthly,
        "weights_rebal_days": weights_rebal,
        "equities": equities,
        "stats": stats,
        "best_by_cagr": best_by_cagr,
        "best_by_nav": best_by_nav,
        "score_monthly": w_score_monthly,
        "score_rebal_days": w_score_rebal,
        "score_equity": eq_score,
        "score_stats": score_stats,
    }


if __name__ == "__main__":
    ETF1 = "069500"
    ETF2 = "148070"  # 나스닥 133690, 키움국고채10년 148070
    START = "2012-11-01"
    END = "2026-02-09"

    run_momentum_lookback_sweep(
        etf1_code=ETF1,
        etf2_code=ETF2,
        start=START,
        end=END,
        upd_stkpc_tp="1",
        lookback_range=range(1, 13),
        score_months=12,
    )
