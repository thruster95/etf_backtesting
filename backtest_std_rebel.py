import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kiwoom_data import fetch_ohlcv_kiwoom_paged

pd.set_option("display.max_rows", None)        # 모든 행 표시
pd.set_option("display.max_columns", None)     # 모든 열 표시
pd.set_option("display.width", None)           # 줄바꿈 폭 자동
pd.set_option("display.max_colwidth", None)    # 컬럼 너비 제한 해제

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

        vals = w * (1.0 + rets.loc[d].values)
        if vals.sum() != 0:
            w = vals / vals.sum()

        out.append(equity)

    return pd.Series(out, index=px.index, name="Rebal_50_50")


# -----------------------------
# 변동성 타깃(역변동성 비중) 전략
# -----------------------------
def compute_monthly_returns(px: pd.DataFrame) -> pd.DataFrame:
    """
    단순 월수익률 (로그 X)
    month-end 종가 기준: R_m = P_m / P_{m-1} - 1
    """
    mpx = px.resample("ME").last()
    mret = mpx.pct_change()
    return mret


def compute_inverse_vol_weights_monthly(
    px: pd.DataFrame,
    lookback_months: int = 12,
    min_vol: float = 1e-8
) -> pd.DataFrame:
    """
    - 최근 lookback_months의 '월수익률 표준편차(표본표준편차, ddof=1)' 계산
    - 역변동성 비중: w_i ∝ 1/vol_i
    - 룩어헤드 방지: t월 리밸런싱 비중은 (t-1)월말까지 데이터로 계산
      => rolling std 계산 후 1개월 shift => X
    # t월말까지 확정된 월수익률로 변동성 계산,
    # 해당 값은 다음 달 첫 거래일 리밸런싱에 사용
    """
    mret = compute_monthly_returns(px)  # index: month-end
    vol = mret.rolling(window=lookback_months, min_periods=lookback_months).std(ddof=1)
    #vol = vol.shift(1)  # 직전월까지 정보만 사용

    inv_vol = 1.0 / vol.clip(lower=min_vol)
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # NaN인 초기 구간은 동등가중으로 대체
    w = w.fillna(0.5)
    return w  # month-end index, columns ETF1/ETF2


def map_monthly_weights_to_daily_rebal_days(
    px: pd.DataFrame,
    w_monthly: pd.DataFrame
) -> pd.DataFrame:
    """
    month-end로 계산된 월별 비중을 각 월 첫 거래일 리밸런싱에 적용
    예) 2020-05-31 비중 -> 2020-06월 첫 거래일 적용
    """
    rebal_days = first_trading_day_each_month(px.index)
    out = []

    for d in rebal_days:
        prev_month_end = (d - pd.offsets.MonthEnd(1)).normalize() + pd.offsets.MonthEnd(0)
        # w_monthly index는 month-end timestamp (시각 00:00:00)
        if prev_month_end in w_monthly.index:
            w = w_monthly.loc[prev_month_end, ["ETF1", "ETF2"]].values.astype(float)
        else:
            w = np.array([0.5, 0.5], dtype=float)
        # 안전 정규화
        if np.isfinite(w).all() and w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.array([0.5, 0.5], dtype=float)

        out.append([d, w[0], w[1]])

    w_daily_rebal = pd.DataFrame(out, columns=["Date", "ETF1", "ETF2"]).set_index("Date")
    return w_daily_rebal


def backtest_monthly_rebal_with_dynamic_weights(
    px: pd.DataFrame,
    w_rebal: pd.DataFrame,  # index=rebal day, cols ETF1/ETF2
    initial: float = 1.0,
    name: str = "Rebal_InverseVol_12M"
) -> pd.Series:
    """
    매월 첫 거래일에 w_rebal의 목표비중으로 리밸런싱.
    월중에는 드리프트.
    """
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
    """
    equity(일별 누적 NAV) -> 연도별 수익률(%)
    계산: 연말 NAV 기준 pct_change
    """
    y_end = equity.resample("YE").last()     # 연말 기준
    y_ret = y_end.pct_change() * 100.0       # %
    y_ret.index = y_ret.index.year           # 인덱스를 YYYY로
    return y_ret


def build_yearly_return_table(equity_dict: dict) -> pd.DataFrame:
    """
    equity_dict 예:
      {
        "InvVol 12M": eq_invvol,
        "Rebal 50/50": eq_rebal_5050,
        "B&H 069500": eq1,
        "B&H 148070": eq2,
      }
    반환: 연도 x 전략 수익률(%)
    """
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


def run_pair_backtest_with_invvol(
    etf1_code: str,
    etf2_code: str,
    start: str,
    end: str,
    upd_stkpc_tp: str = "1",
    lookback_months: int = 12
):
    df1 = fetch_range(etf1_code, start, end, upd_stkpc_tp=upd_stkpc_tp)
    df2 = fetch_range(etf2_code, start, end, upd_stkpc_tp=upd_stkpc_tp)
    px = make_price_panel(df1, df2)

   #변동성을 구하고 싶어서 1년전 데이터도 주가 뽑기
    start_prev_1y = (pd.to_datetime(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    df1_2 = fetch_range(etf1_code, start_prev_1y, end, upd_stkpc_tp=upd_stkpc_tp)
    df2_2 = fetch_range(etf2_code, start_prev_1y, end, upd_stkpc_tp=upd_stkpc_tp)
    px_2 = make_price_panel(df1_2, df2_2)

    # 기존 비교 전략
    eq_rebal_5050 = backtest_monthly_rebal(px, w_target=(0.5, 0.5), initial=1.0)
    eq1 = buy_and_hold(px["ETF1"], initial=1.0)
    eq2 = buy_and_hold(px["ETF2"], initial=1.0)

    # 신규: 12개월 월간변동성 역수 비중
    w_monthly = compute_inverse_vol_weights_monthly(px_2, lookback_months=lookback_months)
    w_rebal_days = map_monthly_weights_to_daily_rebal_days(px, w_monthly)
    eq_invvol = backtest_monthly_rebal_with_dynamic_weights(
        px, w_rebal_days, initial=1.0, name=f"Rebal_InvVol_{lookback_months}M"
    )

    stats = pd.DataFrame([
        summarize(eq_invvol, f"Rebal InverseVol ({lookback_months}M monthly std)"),
        summarize(eq_rebal_5050, "Rebal 50/50 (Monthly)"),
        summarize(eq1, f"Buy&Hold ETF1 ({etf1_code})"),
        summarize(eq2, f"Buy&Hold ETF2 ({etf2_code})"),
    ]).set_index("전략")

    yearly_table = build_yearly_return_table({
        f"InvVol {lookback_months}M": eq_invvol,
        "Rebal 50/50": eq_rebal_5050,
        f"B&H {etf1_code}": eq1,
        f"B&H {etf2_code}": eq2,
    })

    print("\n[성과 요약]")
    print(stats)

    print("\n[연도별 수익률(%)]")
    print(yearly_table)

    # 참고: 최근 12개월 기준으로 계산된 월별 비중 일부 출력
    print("\n[월별 타깃 비중 샘플 - month end 기준(다음 달 첫 거래일 적용)]")
    print(w_monthly.tail(300).round(4))

    plot_equity(
        [eq_invvol, eq_rebal_5050, eq1, eq2],
        [f"InvVol {lookback_months}M", "Rebal 50/50", f"B&H {etf1_code}", f"B&H {etf2_code}"]
    )
    plot_drawdown(
        [eq_invvol, eq_rebal_5050, eq1, eq2],
        [f"InvVol {lookback_months}M", "Rebal 50/50", f"B&H {etf1_code}", f"B&H {etf2_code}"]
    )

    return {
        "prices": px,
        "weights_monthly": w_monthly,
        "weights_rebal_days": w_rebal_days,
        "equity": (eq_invvol, eq_rebal_5050, eq1, eq2),
        "stats": stats,
    }


if __name__ == "__main__":
    ETF1 = "069500"   # 예: 지수 ETF 예: KODEX200: 069500(2002~ / 레버리지: 122630 / S&P500: 379800
    ETF2 = "132030"   # 예: KODEX국고채3년: 114260 / 키움국고채10년: 148070(2011/10~) /
    START = "2012-11-01"
    END   = "2026-02-09"

    run_pair_backtest_with_invvol(
        etf1_code=ETF1,
        etf2_code=ETF2,
        start=START,
        end=END,
        upd_stkpc_tp="1",
        lookback_months=12
    )
