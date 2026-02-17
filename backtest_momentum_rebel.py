import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kiwoom_data import fetch_ohlcv_kiwoom_paged


def _estimate_days_needed(start: str, end: str, buffer_days: int = 120) -> int:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    if e < s:
        raise ValueError("end가 start보다 빠릅니다.")
    cal_days = (e - s).days + 1
    approx_trading_days = int(cal_days * 0.75)
    return max(120, approx_trading_days + buffer_days)


def fetch_range(stk_cd: str, start: str, end: str, upd_stkpc_tp: str = "1") -> pd.DataFrame:
    days_needed = _estimate_days_needed(start, end, buffer_days=180)
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


def buy_and_hold(price: pd.Series, initial: float = 1.0) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    return initial * (1.0 + r).cumprod()


def backtest_monthly_rebal_5050(px: pd.DataFrame, initial: float = 1.0) -> pd.Series:
    rets = px.pct_change().fillna(0.0)
    rebal_days = set(first_trading_day_each_month(px.index))

    equity = initial
    w = np.array([0.5, 0.5], dtype=float)
    out = []

    for d in px.index:
        if d in rebal_days:
            w = np.array([0.5, 0.5], dtype=float)

        r = float(np.dot(w, rets.loc[d].values))
        equity *= (1.0 + r)

        vals = w * (1.0 + rets.loc[d].values)
        if vals.sum() > 0:
            w = vals / vals.sum()

        out.append(equity)

    return pd.Series(out, index=px.index, name="Rebal_50_50")


def compute_momentum_signals(px: pd.DataFrame, lookback_months: int = 12) -> pd.Series:
    """
    모멘텀 시그널(월말):
    - ETF1(주식) 월말 종가 > n개월 전 월말 종가 => +1 (주식 100 / 채권 0)
    - ETF1(주식) 월말 종가 < n개월 전 월말 종가 => -1 (주식 0 / 채권 100)
    - 동일(=0)인 경우 => 직전 포지션 유지

    룩어헤드 방지를 위해 월말에 계산한 시그널은 '다음 달 첫 거래일'에 적용.
    """
    mpx = px.resample("ME").last()
    mom = mpx["ETF1"] / mpx["ETF1"].shift(lookback_months) - 1.0

    signal = pd.Series(index=mpx.index, dtype=float)
    signal[mom > 0] = 1.0
    signal[mom < 0] = -1.0

    signal = signal.ffill().fillna(1.0)
    return signal


def map_signals_to_rebal_weights(px: pd.DataFrame, signal_month_end: pd.Series) -> pd.DataFrame:
    rebal_days = first_trading_day_each_month(px.index)
    out = []

    for d in rebal_days:
        prev_month_end = (d - pd.offsets.MonthEnd(1)).normalize() + pd.offsets.MonthEnd(0)

        sig = signal_month_end.get(prev_month_end, np.nan)
        if np.isnan(sig):
            w = np.array([0.5, 0.5], dtype=float)
        elif sig > 0:
            w = np.array([1.0, 0.0], dtype=float)
        else:
            w = np.array([0.0, 1.0], dtype=float)

        out.append([d, w[0], w[1]])

    return pd.DataFrame(out, columns=["Date", "ETF1", "ETF2"]).set_index("Date")


def backtest_monthly_rebal_with_weights(
    px: pd.DataFrame,
    w_rebal: pd.DataFrame,
    initial: float = 1.0,
    name: str = "Rebal_Momentum"
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
        if vals.sum() > 0:
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


def summarize(eq: pd.Series, name: str) -> dict:
    return {
        "전략": name,
        "CAGR(%)": round(calc_cagr(eq) * 100, 2),
        "MDD(%)": round(calc_mdd(eq) * 100, 2),
        "Final(NAV)": round(float(eq.iloc[-1]), 4),
    }


def plot_equity(eq_list, labels, title="Equity Curve (Normalized to 1.0)"):
    plt.figure(figsize=(12, 6))
    for eq, lb in zip(eq_list, labels):
        plt.plot(eq.index, eq.values, label=lb)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def run_pair_backtest_with_momentum(
    etf1_code: str,
    etf2_code: str,
    start: str,
    end: str,
    upd_stkpc_tp: str = "1",
    lookback_months: int = 12,
):
    # 모멘텀 계산을 위해 n개월 과거 데이터 포함해 fetch
    start_with_buffer = (pd.to_datetime(start) - pd.DateOffset(months=lookback_months + 2)).strftime("%Y-%m-%d")

    df1_full = fetch_range(etf1_code, start_with_buffer, end, upd_stkpc_tp=upd_stkpc_tp)
    df2_full = fetch_range(etf2_code, start_with_buffer, end, upd_stkpc_tp=upd_stkpc_tp)
    px_full = make_price_panel(df1_full, df2_full)

    # 실제 백테스트 구간
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    px = px_full[(px_full.index >= s) & (px_full.index <= e)].copy()

    signal_month_end = compute_momentum_signals(px_full, lookback_months=lookback_months)
    w_rebal_days = map_signals_to_rebal_weights(px, signal_month_end)

    eq_mom = backtest_monthly_rebal_with_weights(
        px,
        w_rebal_days,
        initial=1.0,
        name=f"Rebal_Momentum_{lookback_months}M",
    )
    eq_rebal_5050 = backtest_monthly_rebal_5050(px, initial=1.0)
    eq1 = buy_and_hold(px["ETF1"], initial=1.0)
    eq2 = buy_and_hold(px["ETF2"], initial=1.0)

    stats = pd.DataFrame([
        summarize(eq_mom, f"Momentum ({lookback_months}M)"),
        summarize(eq_rebal_5050, "Rebal 50/50 (Monthly)"),
        summarize(eq1, f"Buy&Hold ETF1 ({etf1_code})"),
        summarize(eq2, f"Buy&Hold ETF2 ({etf2_code})"),
    ]).set_index("전략")

    print("\n[성과 요약]")
    print(stats)

    print("\n[월말 모멘텀 시그널 샘플 (+1:주식, -1:채권)]")
    print(signal_month_end.tail(24))

    print("\n[리밸런싱일 타깃 비중 샘플]")
    print(w_rebal_days.tail(24).round(4))

    plot_equity(
        [eq_mom, eq_rebal_5050, eq1, eq2],
        [f"Momentum {lookback_months}M", "Rebal 50/50", f"B&H {etf1_code}", f"B&H {etf2_code}"],
    )

    return {
        "prices": px,
        "signal_month_end": signal_month_end,
        "weights_rebal_days": w_rebal_days,
        "equity": (eq_mom, eq_rebal_5050, eq1, eq2),
        "stats": stats,
    }


if __name__ == "__main__":
    ETF1 = "069500"  # 주식 ETF
    ETF2 = "148070"  # 채권 ETF
    START = "2012-01-01"
    END = "2026-02-09"

    run_pair_backtest_with_momentum(
        etf1_code=ETF1,
        etf2_code=ETF2,
        start=START,
        end=END,
        upd_stkpc_tp="1",
        lookback_months=12,
    )
