"""Feishu user-identity CLI backend; credentials remain in lark-cli's encrypted store."""

import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

from molgap.tracking.feishu import FeishuError
from molgap.tracking.records import SCHEMA, read_json, utc_now, write_json


def cli(arguments, *, timeout=90):
    try:
        result = subprocess.run(
            ["lark-cli", *arguments, "--as", "user"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FeishuError(f"Lark CLI unavailable: {type(exc).__name__}") from None
    try:
        payload = json.loads(result.stdout if result.returncode == 0 else result.stderr)
    except ValueError:
        raise FeishuError(f"Lark CLI exit {result.returncode}; no valid JSON envelope") from None
    if result.returncode != 0 or payload.get("ok") is not True:
        error = payload.get("error", {})
        # Never echo arbitrary raw CLI output (tokens/authorization links may appear).
        raise FeishuError(
            f"Lark CLI exit {result.returncode}: "
            f"{error.get('type', 'API error')} code={error.get('code')}"
        )
    return payload.get("data", {})


def table_fields():
    fields = []
    for name, kind in SCHEMA.items():
        field = {"name": name, "type": "number" if kind == 2 else "text"}
        if kind == 2 and (name == "学习率" or any(m in name for m in ("MAE", "MSE", "RMSE", "R2"))):
            # CLI Base v3 supports 0..4 decimal places; full values remain in JSON/CSV.
            field["style"] = {"type": "plain", "precision": 4}
        fields.append(field)
    return fields + [{"name": "预测结果附件", "type": "attachment"}]


HARDWARE_FIELDS = [
    {"name": "采样ID", "type": "text"},
    {"name": "采样时间", "type": "text"},
    {"name": "队列", "type": "text"},
    {"name": "GPU型号", "type": "text"},
    *[
        {"name": name, "type": "number"}
        for name in ("GPU利用率_pct", "整卡显存_MiB", "系统可用内存_MiB", "数据盘剩余_GiB")
    ],
]


class LarkCLIClient:
    def __init__(self, destination: dict, cache: Path):
        self.destination, self.cache = destination, cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self._validated = False

    def args(self, table=None):
        return [
            "--base-token",
            self.destination["base_token"],
            "--table-id",
            table or self.destination["table_id"],
        ]

    def ensure_schema(self):
        fields = cli(["base", "+field-list", *self.args()])["fields"]
        actual = {f["name"]: f["type"] for f in fields}
        if any(
            actual.get(name) != ("number" if kind == 2 else "text") for name, kind in SCHEMA.items()
        ):
            raise FeishuError("Training table schema mismatch; check the configured destination")
        if actual.get("预测结果附件") != "attachment":
            raise FeishuError("Missing prediction attachment field")
        self._validated = True

    def find(self, key, value, table=None):
        data = cli(
            [
                "base",
                "+record-list",
                *self.args(table),
                "--field-id",
                key,
                "--limit",
                "2",
                "--json",
                "--filter-json",
                json.dumps(
                    {"logic": "and", "conditions": [[key, "==", value]]}, ensure_ascii=False
                ),
            ]
        )
        ids = data.get("record_id_list", [])
        if len(ids) > 1 or data.get("has_more"):
            raise FeishuError(f"Duplicate {key} rows; resolve before syncing")
        return ids[0] if ids else None

    def upsert(self, fields, key, table=None):
        identity = ":".join(
            (self.destination["base_token"], table or self.destination["table_id"], fields[key])
        )
        receipt_path = self.cache / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
        with receipt_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = read_json(receipt_path)
            comparable = {k: v for k, v in fields.items() if k != "更新时间"}
            digest = hashlib.sha256(
                json.dumps(comparable, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if receipt.get("record_id") and receipt.get("fields_hash") == digest:
                return receipt["record_id"], receipt_path
            # Lookup recovers a successful creation whose response was lost.
            identifier = receipt.get("record_id") or self.find(key, fields[key], table)
            command = [
                "base",
                "+record-upsert",
                *self.args(table),
                "--json",
                json.dumps(fields, ensure_ascii=False, allow_nan=False),
            ]
            if identifier:
                command += ["--record-id", identifier]
            result = cli(command)
            identifier = (
                identifier
                or result.get("record_id")
                or next(iter(result.get("record_id_list", [])), None)
            )
            if not identifier:
                identifier = self.find(key, fields[key], table)
            if not identifier:
                raise FeishuError("Write succeeded but record identity was not returned")
            write_json(
                receipt_path,
                {
                    **receipt,
                    "record_id": identifier,
                    "run_id": fields[key],
                    "synced_at": utc_now(),
                    "fields_hash": digest,
                },
            )
            return identifier, receipt_path

    def sync_record(self, record):
        if not self._validated:
            self.ensure_schema()
        return self.upsert(record["fields"], "运行ID")

    def attach_prediction(self, record_id, receipt_path, prediction):
        receipt = read_json(receipt_path)
        if receipt.get("prediction_uploaded"):
            return
        # A lost upload response is recovered by reading the real attachment cell.
        data = (
            cli(
                [
                    "base",
                    "+record-list",
                    *self.args(),
                    "--field-id",
                    "预测结果附件",
                    "--field-id",
                    "运行ID",
                    "--limit",
                    "2",
                    "--json",
                    "--filter-json",
                    json.dumps(
                        {
                            "logic": "and",
                            "conditions": [
                                ["运行ID", "==", read_json(receipt_path).get("run_id", "")]
                            ],
                        }
                    ),
                ]
            )
            if read_json(receipt_path).get("run_id")
            else None
        )
        if data and any(row[0] for row in data.get("data", [])):
            write_json(receipt_path, {**receipt, "prediction_uploaded": True})
            return
        cli(
            [
                "base",
                "+record-upload-attachment",
                *self.args(),
                "--record-id",
                record_id,
                "--field-id",
                "预测结果附件",
                "--file",
                str(prediction),
            ],
            timeout=180,
        )
        write_json(receipt_path, {**receipt, "prediction_uploaded": True})


def create_destination(path: Path):
    if path.exists():
        return read_json(path)
    result = cli(
        [
            "base",
            "+base-create",
            "--name",
            "QM9 模型训练与资源监测",
            "--table-name",
            "模型训练记录",
            "--time-zone",
            "Asia/Shanghai",
            "--fields",
            json.dumps(table_fields(), ensure_ascii=False),
        ]
    )
    write_json(path.with_suffix(".creation.json"), result)
    # The typed shortcut returns the created Base plus initial table metadata.
    base = result.get("base", result.get("app", result))
    token = base.get("base_token") or base.get("app_token") or base.get("token")
    if not token:
        raise FeishuError("Base created; inspect the saved creation receipt before retrying")
    tables = cli(["base", "+table-list", "--base-token", token])
    items = tables.get("tables", tables.get("items", []))
    table = next(t for t in items if t.get("name") == "模型训练记录")
    hardware = cli(
        [
            "base",
            "+table-create",
            "--base-token",
            token,
            "--name",
            "硬件采样",
            "--fields",
            json.dumps(HARDWARE_FIELDS, ensure_ascii=False),
        ]
    )
    write_json(path.with_suffix(".hardware-creation.json"), hardware)
    hardware_table = hardware.get("table", hardware)
    destination = {
        "base_token": token,
        "table_id": table.get("id", table.get("table_id")),
        "hardware_table_id": hardware_table.get("id", hardware_table.get("table_id")),
        "url": base.get("url") or f"https://my.feishu.cn/base/{token}",
    }
    write_json(path, destination)
    return destination
