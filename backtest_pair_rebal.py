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


def backtest_monthly_rebal(px: pd.DataFrame, w_target=(0.5, 0.5), initial=1.0) -> pd.Series:
    w_target = np.array(w_target, dtype=float)
    w_target = w_target / w_target.sum()

    rets = px.pct_change().fillna(0.0)
    rebal_days = set(first_trading_day_each_month(px.index))

    equity = initial
    w = w_target.copy()
    out = []

    for d in px.index:
        if d in rebal_days:
            w = w_target.copy()

        r = float(np.dot(w, rets.loc[d].values))
        equity *= (1.0 + r)

        # 비중 드리프트 반영
        vals = w * (1.0 + rets.loc[d].values)
        if vals.sum() != 0:
            w = vals / vals.sum()

        out.append(equity)

    return pd.Series(out, index=px.index, name="Rebal_50_50")


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


def plot_equity(eq_rebal: pd.Series, eq1: pd.Series, eq2: pd.Series, code1: str, code2: str):
    plt.figure(figsize=(12, 6))
    plt.plot(eq_rebal.index, eq_rebal.values, label="Rebal 50/50 (Monthly)")
    plt.plot(eq1.index, eq1.values, label=f"Buy&Hold ETF1 ({code1})")
    plt.plot(eq2.index, eq2.values, label=f"Buy&Hold ETF2 ({code2})")
    plt.title("Equity Curve (Normalized to 1.0)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_drawdown(eq_rebal: pd.Series, eq1: pd.Series, eq2: pd.Series, code1: str, code2: str):
    dd_rebal = eq_rebal / eq_rebal.cummax() - 1.0
    dd1 = eq1 / eq1.cummax() - 1.0
    dd2 = eq2 / eq2.cummax() - 1.0

    plt.figure(figsize=(12, 4))
    plt.plot(dd_rebal.index, dd_rebal.values, label="Rebal DD")
    plt.plot(dd1.index, dd1.values, label=f"ETF1 DD ({code1})")
    plt.plot(dd2.index, dd2.values, label=f"ETF2 DD ({code2})")
    plt.title("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_pair_backtest(etf1_code: str, etf2_code: str, start: str, end: str, upd_stkpc_tp: str = "1"):
    df1 = fetch_range(etf1_code, start, end, upd_stkpc_tp=upd_stkpc_tp)
    df2 = fetch_range(etf2_code, start, end, upd_stkpc_tp=upd_stkpc_tp)

    px = make_price_panel(df1, df2)

    eq_rebal = backtest_monthly_rebal(px, w_target=(0.5, 0.5), initial=1.0)
    eq1 = buy_and_hold(px["ETF1"], initial=1.0)
    eq2 = buy_and_hold(px["ETF2"], initial=1.0)

    stats = pd.DataFrame([
        summarize(eq_rebal, "Rebal 50/50 (Monthly)"),
        summarize(eq1, f"Buy&Hold ETF1 ({etf1_code})"),
        summarize(eq2, f"Buy&Hold ETF2 ({etf2_code})"),
    ]).set_index("전략")

    print("\n[성과 요약]")
    print(stats)

    plot_equity(eq_rebal, eq1, eq2, etf1_code, etf2_code)
    plot_drawdown(eq_rebal, eq1, eq2, etf1_code, etf2_code)

    return {"prices": px, "equity": (eq_rebal, eq1, eq2), "stats": stats}


if __name__ == "__main__":
    # 예시 (원하는 ETF로 바꿔)
    ETF1 = "379800"   # 지수 ETF 예: KODEX200: 069500 / 레버리지: 122630 / S&P500: 379800
    ETF2 = "148070"   # 채권 ETF 예: KODEX국고채3년: 114260 / 키움국고채10년: 148070 /

    START = "2011-01-01"
    END   = "2026-01-31"

    run_pair_backtest(ETF1, ETF2, START, END, upd_stkpc_tp="1")
