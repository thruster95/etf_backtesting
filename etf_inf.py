# 사용 예시 (함수는 이미 위에서 정의되어 있다고 가정)

from kiwoom_data import fetch_etf_info_ka40004

# 1) 전체 ETF 조회 + CSV 저장
df_all = fetch_etf_info_ka40004(
    txon_type="0",
    navpre="0",
    mngmcomp="0000",
    txon_yn="0",
    trace_idex="0",
    stex_tp="3",
    save_csv=True,
)

print("전체 건수:", len(df_all))
print(df_all.head())
print(df_all.columns.tolist())
