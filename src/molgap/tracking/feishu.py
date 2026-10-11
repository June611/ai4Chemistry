"""Direct Feishu OpenAPI access with token refresh and durable record identity."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

from molgap.tracking.records import SCHEMA, read_json, utc_now, write_json


class FeishuError(RuntimeError):
    """An API failure whose message contains no credential or response body."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass(repr=False)
class Settings:
    app_id: str
    app_secret: str
    app_token: str
    table_id: str

    @classmethod
    def from_env(cls) -> Settings:
        app_token = os.environ.get("FEISHU_APP_TOKEN", "")
        table_id = os.environ.get("FEISHU_TABLE_ID", "")
        if value := os.environ.get("FEISHU_BITABLE_URL"):
            parsed = urlparse(value)
            parts = parsed.path.strip("/").split("/")
            if not parsed.hostname or not parsed.hostname.endswith(".feishu.cn"):
                raise ValueError("FEISHU_BITABLE_URL must be a domestic Feishu Base URL")
            if len(parts) != 2 or parts[0] != "base":
                raise ValueError(
                    "Use a /base/<app_token> URL; a /wiki/ node token is not app_token"
                )
            app_token = app_token or parts[1]
            table_id = table_id or parse_qs(parsed.query).get("table", [""])[0]
        values = {
            "FEISHU_APP_ID": os.environ.get("FEISHU_APP_ID", ""),
            "FEISHU_APP_SECRET": os.environ.get("FEISHU_APP_SECRET", ""),
            "FEISHU_APP_TOKEN": app_token,
            "FEISHU_TABLE_ID": table_id,
        }
        missing = [name for name, value in values.items() if not value.strip()]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        return cls(*values.values())


class FeishuClient:
    def __init__(self, settings: Settings, timeout: float = 10):
        self.settings = settings
        self.timeout = timeout
        self._token = ""
        self._expires_at = 0.0
        self._validated = False
        self.base = "/bitable/v1/apps/{}/tables/{}".format(
            quote(settings.app_token, safe=""), quote(settings.table_id, safe="")
        )

    def _http(self, method: str, path: str, body=None, query=None, token="") -> dict:
        url = "https://open.feishu.cn/open-apis" + path
        if query:
            url += "?" + urlencode(query)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = "Bearer " + token
        encoded = (
            json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            if body is not None
            else None
        )
        request = Request(url, data=encoded, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise FeishuError(f"Feishu HTTP {exc.code}; check network/API permissions") from None
        except (OSError, URLError):
            raise FeishuError("Feishu connection failed; local data is preserved") from None
        except (ValueError, TypeError):
            raise FeishuError("Invalid Feishu JSON response") from None
        if not isinstance(result, dict) or "code" not in result:
            raise FeishuError("Invalid Feishu response envelope")
        if result["code"] != 0:
            code = result["code"]
            hint = "check API permissions, published app version and document app edit access"
            raise FeishuError(f"Feishu API code {code}; {hint}", code=code)
        return result

    def _access_token(self) -> str:
        if time.monotonic() >= self._expires_at:
            result = self._http(
                "POST",
                "/auth/v3/tenant_access_token/internal",
                {
                    "app_id": self.settings.app_id,
                    "app_secret": self.settings.app_secret,
                },
            )
            self._token = result["tenant_access_token"]
            self._expires_at = time.monotonic() + max(0, result["expire"] - 60)
        return self._token

    def request(self, method: str, path: str, body=None, query=None) -> dict:
        try:
            result = self._http(method, path, body, query, self._access_token())
        except FeishuError as exc:
            if exc.code not in {99991661, 99991663, 99991664, 99991668}:
                raise
            self._expires_at = 0
            result = self._http(method, path, body, query, self._access_token())
        return result.get("data", {})

    def list_fields(self) -> dict[str, int]:
        fields, page_token = {}, ""
        while True:
            query = {"page_size": 100}
            if page_token:
                query["page_token"] = page_token
            data = self.request("GET", self.base + "/fields", query=query)
            fields.update({item["field_name"]: item["type"] for item in data.get("items", [])})
            if not data.get("has_more"):
                return fields
            next_token = data.get("page_token")
            if not next_token or next_token == page_token:
                raise FeishuError("Invalid Feishu pagination token")
            page_token = next_token

    def ensure_schema(self, create: bool = False) -> None:
        existing = self.list_fields()
        incompatible = [
            name for name, kind in SCHEMA.items() if name in existing and existing[name] != kind
        ]
        if incompatible:
            raise FeishuError("Incompatible field types: " + ", ".join(incompatible))
        missing = [name for name in SCHEMA if name not in existing]
        if missing and not create:
            raise FeishuError("Missing fields; run init-table: " + ", ".join(missing))
        for name in missing:
            self.request("POST", self.base + "/fields", {"field_name": name, "type": SCHEMA[name]})
            time.sleep(0.1)
        self._validated = True

    def find_record(self, run_id: str) -> str | None:
        data = self.request(
            "POST",
            self.base + "/records/search",
            {
                "field_names": ["运行ID"],
                "filter": {
                    "conjunction": "and",
                    "conditions": [
                        {"field_name": "运行ID", "operator": "is", "value": [run_id]},
                    ],
                },
            },
            {"page_size": 2},
        )
        items = data.get("items", [])
        if len(items) > 1 or data.get("has_more"):
            raise FeishuError("Duplicate 运行ID rows; resolve duplicates before synchronizing")
        return items[0]["record_id"] if items else None

    def sync(self, run_dir: Path, record: dict) -> None:
        directory = run_dir / "tracking"
        directory.mkdir(parents=True, exist_ok=True)
        # Serialize manual backfills and the background worker for this local run.
        with (directory / ".sync.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self._validated:
                self.ensure_schema()
            saved = directory / "feishu_sync.json"
            receipt = read_json(saved)
            identity = [self.settings.app_token, self.settings.table_id, record["run_id"]]
            if receipt.get("identity") != identity:
                receipt = {"identity": identity, "create_token": str(uuid.uuid4())}
            record_id = receipt.get("record_id") or self.find_record(record["run_id"])
            # Save the same UUID before creation, including when a response is lost.
            write_json(saved, receipt)
            if record_id:
                self.request(
                    "PUT",
                    self.base + "/records/" + quote(record_id, safe=""),
                    {"fields": record["fields"]},
                )
            else:
                result = self.request(
                    "POST",
                    self.base + "/records",
                    {"fields": record["fields"]},
                    {"client_token": receipt["create_token"]},
                )
                record_id = result["record"]["record_id"]
                # An idempotent creation may return an earlier snapshot. Apply the latest.
                self.request(
                    "PUT",
                    self.base + "/records/" + quote(record_id, safe=""),
                    {"fields": record["fields"]},
                )
            write_json(saved, {**receipt, "record_id": record_id, "synced_at": utc_now()})
