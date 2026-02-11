import os
import datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()

KIWOOM_APPKEY = os.getenv("KIWOOM_APPKEY")
KIWOOM_SECRETKEY = os.getenv("KIWOOM_SECRETKEY")
KIWOOM_BASE = os.getenv("KIWOOM_BASE", "https://api.kiwoom.com")

if not KIWOOM_APPKEY or not KIWOOM_SECRETKEY:
    raise RuntimeError("KIWOOM_APPKEY / KIWOOM_SECRETKEY 를 .env에 설정해줘.")


class KiwoomRest:
    def __init__(self, base: str, appkey: str, secretkey: str):
        self.base = base.rstrip("/")
        self.appkey = appkey
        self.secretkey = secretkey
        self._auth_header = None
        self._expires_dt = None

    def _need_refresh(self) -> bool:
        if not self._auth_header or not self._expires_dt:
            return True
        try:
            exp = dt.datetime.strptime(self._expires_dt, "%Y%m%d%H%M%S")
            return dt.datetime.now() >= (exp - dt.timedelta(minutes=2))
        except Exception:
            return True

    def get_token(self) -> str:
        if not self._need_refresh():
            return self._auth_header

        url = f"{self.base}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.appkey,
            "secretkey": self.secretkey,
        }
        h = {"Content-Type": "application/json;charset=UTF-8"}
        r = requests.post(url, json=payload, headers=h, timeout=15)
        r.raise_for_status()
        data = r.json()

        if data.get("return_code", 0) != 0:
            raise RuntimeError(f"토큰 발급 실패: {data.get('return_msg')} ({data.get('return_code')})")

        token = data.get("token")
        token_type = data.get("token_type", "bearer").lower()
        self._expires_dt = data.get("expires_dt")
        self._auth_header = f"{'Bearer' if token_type == 'bearer' else token_type} {token}"
        return self._auth_header

    def post(self, path: str, api_id: str, body: dict, cont_yn: str = "N", next_key: str = "") -> tuple[dict, dict]:
        """
        /api/dostk/chart 공통 호출
        - 헤더: authorization, api-id, cont-yn, next-key
        - 반환: (data_json, resp_meta)  resp_meta는 "응답 헤더"에서 cont/next_key를 읽어서 돌려줌
        """
        url = f"{self.base}{path}"
        auth = self.get_token()
        h = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": auth,
            "api-id": api_id,
            "cont-yn": cont_yn,
            "next-key": next_key,
        }
        r = requests.post(url, json=body, headers=h, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("return_code", 0) != 0:
            raise RuntimeError(f"{api_id} 호출 실패: {data.get('return_msg')} ({data.get('return_code')})")

        resp_meta = {
            "cont_yn": r.headers.get("cont-yn", "N"),
            "next_key": r.headers.get("next-key", ""),
            "api_id": r.headers.get("api-id", api_id),
        }
        return data, resp_meta


    def post_chart(self, api_id: str, body: dict, cont_yn: str = "N", next_key: str = "") -> tuple[dict, dict]:
        """
        /api/dostk/chart 공통 호출
        - 헤더: authorization, api-id, cont-yn, next-key
        - 반환: (data_json, resp_meta)  resp_meta는 "응답 헤더"에서 cont/next_key를 읽어서 돌려줌
        """
        url = f"{self.base}/api/dostk/chart"
        auth = self.get_token()
        h = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": auth,
            "api-id": api_id,
            "cont-yn": cont_yn,
            "next-key": next_key,
        }
        r = requests.post(url, json=body, headers=h, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("return_code", 0) != 0:
            raise RuntimeError(f"{api_id} 호출 실패: {data.get('return_msg')} ({data.get('return_code')})")

        resp_meta = {
            "cont_yn": r.headers.get("cont-yn", "N"),
            "next_key": r.headers.get("next-key", ""),
            "api_id": r.headers.get("api-id", api_id),
        }
        return data, resp_meta



# 전역 인스턴스(다른 파일에서 import 해서 그대로 씀)
KIWOOM = KiwoomRest(KIWOOM_BASE, KIWOOM_APPKEY, KIWOOM_SECRETKEY)
