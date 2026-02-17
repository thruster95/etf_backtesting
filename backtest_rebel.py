import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kiwoom_data import fetch_ohlcv_kiwoom_paged

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def _estimate_days_needed(start: str, end: str, buffer_days: int = 120) -> int:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    if e < s:
        raise ValueError("end가 start보다 빠릅니다.")
    cal_days = (e - s).days + 1
    approx_trading_days = int(cal_days * 0.75)
    return max(120, approx_trading_days + buffer_days)


def fetch_range(stk_cd: str, start: str, end: str, upd_stkpc_tp: str = "1") -> pd.DataFrame:
    days_needed = _estimate_days_needed(start, end, buffer_days=150)
    df = fetch_ohlcv_kiwoom_paged(stk_cd=stk_cd, days=days_needed, upd_stkpc_tp=upd_stkpc_tp)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    df = df[(df["Date"] >= s) & (df["Date"] <= e)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"{stk_cd}: start~end 구간 데이터가 없습니다. (start={start}, end={end})")

    return df


def make_price_panel(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    a = df1[["Date", "Close"]].rename(columns={"Close": "ETF1"}).copy()
    b = df2[["Date", "Close"]].rename(columns={"Close": "ETF2"}).copy()
    a["Date"] = pd.to_datetime(a["Date"])
    b["Date"] = pd.to_datetime(b["Date"])
    px = pd.merge(a, b, on="Date", how="inner").sort_values("Date").set_index("Date").dropna()
    if px.empty:
        raise RuntimeError("두 ETF의 공통 날짜 구간이 없습니다.")
    return px


def first_trading_day_each_month(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=idx)
    first_days = s.groupby([idx.year, idx.month]).apply(lambda x: x.index.min()).values
    return pd.to_datetime(first_days)


def buy_and_hold(price: pd.Series, initial=1.0) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    return initial * (1.0 + r).cumprod()


def compute_monthly_returns(px: pd.DataFrame) -> pd.DataFrame:
    mpx = px.resample("ME").last()
    return mpx.pct_change()


def compute_equal_weights_monthly(px: pd.DataFrame, **kwargs) -> pd.DataFrame:
    mpx = px.resample("ME").last()
    w = pd.DataFrame(index=mpx.index, columns=["ETF1", "ETF2"], data=0.5)
    return w


def compute_inverse_vol_weights_monthly(
    px: pd.DataFrame,
    lookback_months: int = 12,
    min_vol: float = 1e-8,
    **kwargs,
) -> pd.DataFrame:
    mret = compute_monthly_returns(px)
    vol = mret.rolling(window=lookback_months, min_periods=lookback_months).std(ddof=1)
    inv_vol = 1.0 / vol.clip(lower=min_vol)
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return w.fillna(0.5)


def compute_momentum_weights_monthly(
    px: pd.DataFrame,
    lookback_months: int = 12,
    long_only: bool = True,
    **kwargs,
) -> pd.DataFrame:
    mpx = px.resample("ME").last()
    mom = mpx.pct_change(lookback_months)

    if long_only:
        mom = mom.clip(lower=0.0)

#    denom = mom.sum(axis=1)
#    w = mom.div(denom.replace(0, np.nan), axis=0)
#    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.5)

    # 두 ETF 모멘텀을 비교해 높은 쪽에 100% 배분(승자독식)
    winner = mom.idxmax(axis=1)
    w = pd.DataFrame(index=mom.index, columns=["ETF1", "ETF2"], data=0.0)
    w.loc[winner == "ETF1", "ETF1"] = 1.0
    w.loc[winner == "ETF2", "ETF2"] = 1.0

    # 동률 또는 NaN(초기 구간 등)은 50:50으로 처리
    tie_or_nan = (mom["ETF1"] == mom["ETF2"]) | mom.isna().any(axis=1)
    w.loc[tie_or_nan, ["ETF1", "ETF2"]] = 0.5

    return w


def map_monthly_weights_to_daily_rebal_days(px: pd.DataFrame, w_monthly: pd.DataFrame) -> pd.DataFrame:
    rebal_days = first_trading_day_each_month(px.index)
    out = []

    for d in rebal_days:
        prev_month_end = (d - pd.offsets.MonthEnd(1)).normalize() + pd.offsets.MonthEnd(0)
        if prev_month_end in w_monthly.index:
            w = w_monthly.loc[prev_month_end, ["ETF1", "ETF2"]].values.astype(float)
        else:
            w = np.array([0.5, 0.5], dtype=float)

        if np.isfinite(w).all() and w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.array([0.5, 0.5], dtype=float)

        out.append([d, w[0], w[1]])

    return pd.DataFrame(out, columns=["Date", "ETF1", "ETF2"]).set_index("Date")


def backtest_monthly_rebal_with_dynamic_weights(
    px: pd.DataFrame,
    w_rebal: pd.DataFrame,
    initial: float = 1.0,
    name: str = "Rebal_Dynamic",
) -> pd.Series:
    rets = px.pct_change().fillna(0.0)
    rebal_days = set(w_rebal.index)

    equity = initial
    w = np.array([0.5, 0.5], dtype=float)
    out = []

    for d in px.index:
        if d in rebal_days:
            tgt = w_rebal.loc[d, ["ETF1", "ETF2"]].values.astype(float)
            if np.isfinite(tgt).all() and tgt.sum() > 0:
                w = tgt / tgt.sum()

        r = float(np.dot(w, rets.loc[d].values))
        equity *= (1.0 + r)

        vals = w * (1.0 + rets.loc[d].values)
        if vals.sum() != 0:
            w = vals / vals.sum()

        out.append(equity)

    return pd.Series(out, index=px.index, name=name)


def calc_cagr(equity: pd.Series) -> float:
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def calc_mdd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calc_ann_vol_from_daily_equity(equity: pd.Series) -> float:
    dr = equity.pct_change().dropna()
    if len(dr) < 2:
        return np.nan
    return float(dr.std(ddof=1) * np.sqrt(252))


def summarize(eq: pd.Series, name: str) -> dict:
    return {
        "전략": name,
        "CAGR(%)": round(calc_cagr(eq) * 100, 2),
        "MDD(%)": round(calc_mdd(eq) * 100, 2),
        "AnnVol(%)": round(calc_ann_vol_from_daily_equity(eq) * 100, 2),
        "Final(NAV)": round(float(eq.iloc[-1]), 4),
    }


def calc_yearly_returns_from_equity(equity: pd.Series) -> pd.Series:
    y_end = equity.resample("YE").last()
    y_ret = y_end.pct_change() * 100.0
    y_ret.index = y_ret.index.year
    return y_ret


def build_yearly_return_table(equity_dict: dict) -> pd.DataFrame:
    cols = {}
    for name, eq in equity_dict.items():
        cols[name] = calc_yearly_returns_from_equity(eq)

    yearly = pd.DataFrame(cols)
    yearly = yearly.sort_index()
    return yearly.round(2)


def plot_equity(eq_list, labels, title="Equity Curve (Normalized to 1.0)"):
    plt.figure(figsize=(12, 6))
    for eq, lb in zip(eq_list, labels):
        plt.plot(eq.index, eq.values, label=lb)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_drawdown(eq_list, labels, title="Drawdown"):
    plt.figure(figsize=(12, 4))
    for eq, lb in zip(eq_list, labels):
        dd = eq / eq.cummax() - 1.0
        plt.plot(dd.index, dd.values, label=lb)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def _build_strategy_configs(lookback_months: int = 12) -> list:
    return [
        {
            "key": "rebal_50_50",
            "label": "Rebal 50/50 (Monthly)",
            "weight_func": compute_equal_weights_monthly,
            "params": {},
        },
        {
            "key": "invvol",
            "label": f"Rebal InverseVol ({lookback_months}M monthly std)",
            "weight_func": compute_inverse_vol_weights_monthly,
            "params": {"lookback_months": lookback_months},
        },
        {
            "key": "momentum",
            "label": f"Rebal Momentum ({lookback_months}M, long-only)",
            "weight_func": compute_momentum_weights_monthly,
            "params": {"lookback_months": lookback_months, "long_only": True},
        },
    ]


def run_pair_backtest(
    etf1_code: str,
    etf2_code: str,
    start: str,
    end: str,
    upd_stkpc_tp: str = "1",
    lookback_months: int = 12,
):
    strategy_configs = _build_strategy_configs(lookback_months=lookback_months)

    max_lookback = max(
        [cfg["params"].get("lookback_months", 1) for cfg in strategy_configs],
        default=1,
    )

    start_main = pd.to_datetime(start)
    start_hist = (start_main - pd.DateOffset(months=max_lookback + 2)).strftime("%Y-%m-%d")

    df1 = fetch_range(etf1_code, start_main.strftime("%Y-%m-%d"), end, upd_stkpc_tp=upd_stkpc_tp)
    df2 = fetch_range(etf2_code, start_main.strftime("%Y-%m-%d"), end, upd_stkpc_tp=upd_stkpc_tp)
    px = make_price_panel(df1, df2)

    df1_hist = fetch_range(etf1_code, start_hist, end, upd_stkpc_tp=upd_stkpc_tp)
    df2_hist = fetch_range(etf2_code, start_hist, end, upd_stkpc_tp=upd_stkpc_tp)
    px_hist = make_price_panel(df1_hist, df2_hist)

    weights_monthly = {}
    weights_rebal_days = {}
    equities = {}

    for cfg in strategy_configs:
        w_monthly = cfg["weight_func"](px_hist, **cfg["params"])
        w_rebal = map_monthly_weights_to_daily_rebal_days(px, w_monthly)
        eq = backtest_monthly_rebal_with_dynamic_weights(px, w_rebal, initial=1.0, name=cfg["label"])

        weights_monthly[cfg["key"]] = w_monthly
        weights_rebal_days[cfg["key"]] = w_rebal
        equities[cfg["key"]] = eq

    eq1 = buy_and_hold(px["ETF1"], initial=1.0)
    eq2 = buy_and_hold(px["ETF2"], initial=1.0)

    stats_rows = [summarize(equities[cfg["key"]], cfg["label"]) for cfg in strategy_configs]
    stats_rows.extend([
        summarize(eq1, f"Buy&Hold ETF1 ({etf1_code})"),
        summarize(eq2, f"Buy&Hold ETF2 ({etf2_code})"),
    ])
    stats = pd.DataFrame(stats_rows).set_index("전략")

    yearly_inputs = {cfg["label"]: equities[cfg["key"]] for cfg in strategy_configs}
    yearly_inputs[f"B&H {etf1_code}"] = eq1
    yearly_inputs[f"B&H {etf2_code}"] = eq2
    yearly_table = build_yearly_return_table(yearly_inputs)

    print("\n[성과 요약]")
    print(stats)

    print("\n[연도별 수익률(%)]")
    print(yearly_table)

    print("\n[월별 타깃 비중 샘플 - month end 기준(다음 달 첫 거래일 적용)]")
    for cfg in strategy_configs:
        print(f"\n- {cfg['label']}")
        print(weights_monthly[cfg["key"]].tail(12).round(4))

    plot_eq_list = [equities[cfg["key"]] for cfg in strategy_configs] + [eq1, eq2]
    plot_labels = [cfg["label"] for cfg in strategy_configs] + [f"B&H {etf1_code}", f"B&H {etf2_code}"]

    plot_equity(plot_eq_list, plot_labels)
    plot_drawdown(plot_eq_list, plot_labels)

    return {
        "prices": px,
        "weights_monthly": weights_monthly,
        "weights_rebal_days": weights_rebal_days,
        "equity": {**equities, "buy_hold_etf1": eq1, "buy_hold_etf2": eq2},
        "stats": stats,
        "yearly": yearly_table,
    }


if __name__ == "__main__":
    ETF1 = "069500"
    ETF2 = "133690" #나스닥 133690, 키움국고채10년 148070
    START = "2012-11-01"
    END = "2026-02-09"

    run_pair_backtest(
        etf1_code=ETF1,
        etf2_code=ETF2,
        start=START,
        end=END,
        upd_stkpc_tp="1",
        lookback_months=6,
    )
