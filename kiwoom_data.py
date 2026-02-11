import time
import datetime as dt
import pandas as pd
import os

from kiwoom_client import KIWOOM


# =========================
# 숫자 파서 (네가 쓰던 것 그대로)
# =========================
def _to_price(x) -> int:
    """가격용: 부호 제거 후 절대값"""
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    s = s.replace(",", "").replace("+", "").replace("-", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except Exception:
            return 0


def _to_signed(x) -> int:
    """수급/증감용: 부호 유지"""
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    s = s.replace(",", "").replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except Exception:
            return 0


# =========================
# 키움 응답에서 "리스트(배열)" 자동 추출
# =========================
def _extract_first_list(data: dict) -> list:
    candidates = []
    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append((k, v))
    if candidates:
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        return candidates[0][1]
    return []


# =========================
# (A) OHLCV - 단발(최근 N일)
# =========================
def fetch_ohlcv_kiwoom(stk_cd: str, days: int = 130, upd_stkpc_tp: str = "1") -> pd.DataFrame:
    """
    ✅ 키움 REST 기반 일봉 OHLCV (단발)
    - 성공하면 최근 N일(오래된->최신) 정렬 DataFrame 반환
    """
    base_dt = dt.datetime.now().strftime("%Y%m%d")
    body = {"stk_cd": stk_cd, "base_dt": base_dt, "upd_stkpc_tp": upd_stkpc_tp}

    try:
        data, meta = KIWOOM.post(path="/api/dostk/chart", api_id="ka10081", body=body, cont_yn="N", next_key="")
        items = _extract_first_list(data)

        if not items:
            raise RuntimeError("ka10081 : items empty")

        rows = []
        for it in items[: max(days, 60)]:
            d = it.get("dt") or it.get("date") or it.get("bsop_date") or it.get("stck_bsop_date")
            if not d:
                continue

            close = it.get("cur_prc") or it.get("close") or it.get("close_prc") or it.get("cls_prc")
            openp = it.get("open_pric") or it.get("open") or it.get("opn_prc")
            high  = it.get("high_pric") or it.get("high") or it.get("hig_prc")
            low   = it.get("low_pric")  or it.get("low")  or it.get("low_prc")

            vol = it.get("trde_qty") or it.get("volume") or it.get("acc_trde_qty") or it.get("vol")

            rows.append(
                {"Date": d, "Open": _to_price(openp), "High": _to_price(high), "Low": _to_price(low),
                 "Close": _to_price(close), "Volume": _to_price(vol)}
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("ka10081 :: parsed df empty")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").tail(days).reset_index(drop=True)
        return df

    except Exception as e:
        raise RuntimeError(f"OHLCV 조회 실패(ka10081 실패): {e}")


# =========================
# (B) OHLCV - 연속조회 버전 (장기 백테스트용)
# =========================
def fetch_ohlcv_kiwoom_paged(
    stk_cd: str,
    days: int,
    upd_stkpc_tp: str = "1",
    api_id_primary: str = "ka10081",
    api_id_fallback: str = "ka10080",
    max_pages: int = 300,
    sleep_sec: float = 0.05,
) -> pd.DataFrame:
    """
    ✅ 키움 REST 기반 일봉 OHLCV (연속조회 cont_yn='Y' + next_key)
    - 너의 KIWOOM.post()가 resp_meta['next_key']를 "응답 헤더"에서 추출해주므로
      meta['next_key']만 쓰면 됨.
    - 반환: Date/Open/High/Low/Close/Volume, 오래된->최신 정렬, tail(days)
    """

    base_dt = dt.datetime.now().strftime("%Y%m%d")
    body = {"stk_cd": stk_cd, "base_dt": base_dt, "upd_stkpc_tp": upd_stkpc_tp}

    target = max(days, 60)

    def _run(api_id: str) -> pd.DataFrame:
        cont_yn = "N"
        next_key = ""
        seen = set()
        rows = []

        for _ in range(max_pages):
            data, meta = KIWOOM.post(path="/api/dostk/chart", api_id=api_id, body=body, cont_yn=cont_yn, next_key=next_key)
            items = _extract_first_list(data)
            if not items:
                break

            for it in items:
                d = it.get("dt") or it.get("date") or it.get("bsop_date") or it.get("stck_bsop_date")
                if not d or d in seen:
                    continue
                seen.add(d)

                close = it.get("cur_prc") or it.get("close") or it.get("close_prc") or it.get("cls_prc")
                openp = it.get("open_pric") or it.get("open") or it.get("opn_prc")
                high  = it.get("high_pric") or it.get("high") or it.get("hig_prc")
                low   = it.get("low_pric")  or it.get("low")  or it.get("low_prc")
                vol   = it.get("trde_qty")  or it.get("volume") or it.get("acc_trde_qty") or it.get("vol")

                rows.append(
                    {"Date": d, "Open": _to_price(openp), "High": _to_price(high), "Low": _to_price(low),
                     "Close": _to_price(close), "Volume": _to_price(vol)}
                )

            nk = (meta or {}).get("next_key", "") or ""
            if not nk:
                break

            cont_yn = "Y"
            next_key = nk

            if sleep_sec and sleep_sec > 0:
                time.sleep(sleep_sec)

            # 목표치만큼 확보되면 여기서 멈춤(더 과거 필요하면 이 break 삭제)
            if len(rows) >= target:
                break

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"{api_id}: parsed df empty")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        return df

    # primary 먼저
    try:
        return _run(api_id_primary)
    except Exception:
        # fallback
        return _run(api_id_fallback)


# =========================
# (C) 수급 예시 (너가 쓰던 것 유지)
# =========================
def ka10060_investor_chart(stk_cd: str, dt_yyyymmdd: str, amt_qty_tp: str, trde_tp: str, unit_tp: str = "1"):
    body = {
        "dt": dt_yyyymmdd,
        "stk_cd": stk_cd,
        "amt_qty_tp": amt_qty_tp,
        "trde_tp": trde_tp,
        "unit_tp": unit_tp,
    }
    data, meta = KIWOOM.post(path="/api/dostk/chart", api_id="ka10060", body=body, cont_yn="N", next_key="")
    items = data.get("stk_invsr_orgn_chart") or _extract_first_list(data)
    return items or []

# =========================
# (D) ETF 정보 조회 (ka40004)
# =========================
def fetch_etf_info_ka40004(
    txon_type: str = "0",
    navpre: str = "0",
    mngmcomp: str = "0000",
    txon_yn: str = "0",
    trace_idex : str = "0",
    stex_tp : str = "3",
    save_csv: bool = True,
    api_id: str = "ka40004",
    max_pages: int = 50,
    sleep_sec: float = 0.05,
) -> pd.DataFrame:

    body = {
        "txon_type": txon_type,      # 과세유형 0:전체, 1:비과세, 2:보유기간과세, 3:회사형, 4:외국, 5:비과세해외(보유기간관세)
        "navpre": navpre,            # NAV대비 0:전체, 1:NAV > 전일종가, 2:NAV < 전일종가
        "mngmcomp": mngmcomp,        # 운용사 0000:전체, 3020:KODEX(삼성), 3027:KOSEF(키움), 3191:TIGER(미래에셋), 3228:KINDEX(한국투자), 3023:KStar(KB), 3022:아리랑(한화), 9999:기타운용사
        "txon_yn": txon_yn,          # 과세여부 0:전체, 1:과세, 2:비과세
        "trace_idex": trace_idex,    # 추적지수 0:전체
        "stex_tp": stex_tp,          # 거래소구분 1:KRX, 2:NXT, 3:통합
    }

    # 응답 키가 환경별로 다를 수 있으므로 유연하게 처리
    cont_yn = "N"
    next_key = ""
    all_rows = []

    for _ in range(max_pages):
        data, meta = KIWOOM.post(
            path="/api/dostk/etf",
            api_id=api_id,
            body=body,
            cont_yn=cont_yn,
            next_key=next_key
        )

        # 가장 그럴듯한 리스트 추출
        items = _extract_first_list(data)
        if not items:
            # 혹시 dict 단건이면 1건으로 처리
            if isinstance(data, dict) and data:
                # list가 아닌 dict 내부 단건 구조 대응
                # ex) {"etf_info": {...}}
                dict_candidates = [v for v in data.values() if isinstance(v, dict)]
                if dict_candidates:
                    items = dict_candidates
                else:
                    break
            else:
                break

        # 각 row를 평탄화(중첩 dict 방어)
        for it in items:
            row = {}
            for k, v in it.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        row[f"{k}.{kk}"] = vv
                else:
                    row[k] = v
            all_rows.append(row)

        nk = (meta or {}).get("next_key", "") or ""
        if not nk:
            break

        cont_yn = "Y"
        next_key = nk

        if sleep_sec and sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows).drop_duplicates().reset_index(drop=True)

    # 날짜형 컬럼 자동 파싱(가능한 경우만)
    for col in df.columns:
        if "date" in col.lower() or col.lower().endswith("_dt") or col.lower() in ("dt", "base_dt"):
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass

    if save_csv:
        os.makedirs("output_data", exist_ok=True)
        now = dt.datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = f"output_data/etf_info_ka40004_{now}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return df

# =========================
# (E) 테마 정보
# =========================
def fetch_theme_info_ka90001(
    qry_tp: str = "0",
    stk_cd: str = "",
    date_tp: str = "1",         #날짜구분:n일전 (1일 ~ 99일 날짜입력)
    theme_nm: str = "",
    flu_pl_amt_tp: str = "1",   # 1:상위기간수익률, 2:하위기간수익률, 3:상위등락률, 4:하위등락률
    stex_tp: str = "3",         # 1:KRX, 2:NXT, 3:통합
    max_pages: int = 50,
    sleep_sec: float = 0.05,
    save_csv: bool = True,
) -> pd.DataFrame:

    body = {
        "qry_tp": qry_tp,
        "stk_cd": stk_cd,
        "date_tp": date_tp,
        "thema_nm": theme_nm,   # API 키명 유지
        "flu_pl_amt_tp": flu_pl_amt_tp,
        "stex_tp": stex_tp,
    }

    cont_yn = "N"
    next_key = ""
    all_rows = []

    for _ in range(max_pages):
        data, meta = KIWOOM.post(
            path="/api/dostk/thme",
            api_id="ka90001",
            body=body,
            cont_yn=cont_yn,
            next_key=next_key
        )

        items = _extract_first_list(data)
        if not items:
            break

        all_rows.extend(items)

        nk = (meta or {}).get("next_key", "") or ""
        if not nk:
            break

        cont_yn = "Y"
        next_key = nk
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows).drop_duplicates().reset_index(drop=True)

    if save_csv:
        os.makedirs("output_data", exist_ok=True)
        now = dt.datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = f"output_data/theme_info_ka90001_{now}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return df