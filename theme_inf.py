# 사용 예시 (함수는 이미 위에서 정의되어 있다고 가정)

from kiwoom_data import fetch_theme_info_ka90001

# 1) 전체 ETF 조회 + CSV 저장
df_all = fetch_theme_info_ka90001(
    date_tp = "1",                 #날짜구분:n일전 (1일 ~ 99일 날짜입력)
)

print("전체 건수:", len(df_all))
print(df_all.head())
print(df_all.columns.tolist())
