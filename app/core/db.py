import os
import re
import json
import yaml
import aiosqlite
from pathlib import Path

from app.core.config import apply_env_aliases

MAX_USAGE_ROWS = 300_000


class Database:
    def __init__(self, path: str):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self):
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def _execute(self, sql: str, params=()):
        return await self._conn.execute(sql, params)

    async def _fetchall(self, sql: str, params=()):
        c = await self._conn.execute(sql, params)
        return await c.fetchall()

    async def _fetchone(self, sql: str, params=()):
        c = await self._conn.execute(sql, params)
        return await c.fetchone()

    async def _column_names(self, table_name: str) -> set[str]:
        rows = await self._fetchall(f"PRAGMA table_info({table_name})")
        return {row["name"] for row in rows}

    async def _ensure_column(self, table_name: str, column_name: str, ddl: str):
        columns = await self._column_names(table_name)
        if column_name not in columns:
            await self._execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")

    # ── migration ──

    async def migrate(self):
        await self._execute("""
            CREATE TABLE IF NOT EXISTS providers (
                name TEXT PRIMARY KEY,
                base_url TEXT NOT NULL,
                health_url TEXT DEFAULT '',
                timeout INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 1,
                key_strategy TEXT DEFAULT 'round_robin',
                enabled INTEGER DEFAULT 1
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS provider_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT NOT NULL REFERENCES providers(name) ON DELETE CASCADE,
                key_value TEXT NOT NULL,
                rpm INTEGER DEFAULT 60,
                tpm INTEGER,
                weight INTEGER DEFAULT 1,
                note TEXT DEFAULT ''
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT NOT NULL REFERENCES providers(name) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                routing_key TEXT NOT NULL UNIQUE,
                prompt_price REAL DEFAULT 0,
                completion_price REAL DEFAULT 0,
                enabled INTEGER DEFAULT 1
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_value TEXT NOT NULL UNIQUE,
                note TEXT DEFAULT '',
                rate_limit_rpm INTEGER DEFAULT 60,
                allowed_models TEXT DEFAULT '*',
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS routing_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                config_json TEXT DEFAULT '{}',
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS routing_profile_api_keys (
                api_key_id INTEGER PRIMARY KEY REFERENCES api_keys(id) ON DELETE CASCADE,
                routing_profile_id INTEGER NOT NULL REFERENCES routing_profiles(id) ON DELETE CASCADE,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS routing_profile_users (
                user_id TEXT PRIMARY KEY,
                routing_profile_id INTEGER NOT NULL REFERENCES routing_profiles(id) ON DELETE CASCADE,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key_id INTEGER DEFAULT 0,
                subject_type TEXT DEFAULT 'api_key',
                subject_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                routing_profile_id TEXT DEFAULT '',
                request_id TEXT DEFAULT '',
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT DEFAULT 'success',
                error TEXT
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS keyword_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                routing_profile_id INTEGER REFERENCES routing_profiles(id) ON DELETE CASCADE,
                keyword TEXT NOT NULL,
                match_mode TEXT DEFAULT 'contains',
                target_routing_key TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS cascade_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                routing_profile_id INTEGER REFERENCES routing_profiles(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                source_routing_key TEXT NOT NULL,
                escalation_chain TEXT NOT NULL,
                min_content_length INTEGER DEFAULT 10,
                check_refusal INTEGER DEFAULT 1,
                refusal_patterns TEXT DEFAULT '[]',
                check_finish_reason INTEGER DEFAULT 1,
                local_judge_model TEXT,
                local_judge_threshold REAL DEFAULT 0.7,
                local_judge_system_prompt TEXT,
                custom_validator_url TEXT,
                custom_validator_timeout INTEGER DEFAULT 5000,
                max_escalations INTEGER DEFAULT 3,
                cost_threshold REAL DEFAULT 0.0,
                priority INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS cascade_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key_id INTEGER DEFAULT 0,
                subject_type TEXT DEFAULT 'api_key',
                subject_id TEXT DEFAULT '',
                user_id TEXT DEFAULT '',
                routing_profile_id TEXT DEFAULT '',
                request_id TEXT DEFAULT '',
                original_model TEXT NOT NULL,
                models_attempted TEXT NOT NULL,
                final_model TEXT NOT NULL,
                total_escalations INTEGER DEFAULT 0,
                validation_reasons TEXT,
                total_cost REAL DEFAULT 0,
                total_latency_ms INTEGER DEFAULT 0,
                status TEXT DEFAULT 'success'
            )
        """)
        await self._execute("""
            CREATE TABLE IF NOT EXISTS provider_health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider_name TEXT NOT NULL REFERENCES providers(name) ON DELETE CASCADE,
                source_url TEXT NOT NULL,
                status TEXT DEFAULT 'unknown',
                healthy INTEGER,
                score REAL,
                latency_ms REAL DEFAULT 0,
                message TEXT DEFAULT '',
                raw_json TEXT DEFAULT ''
            )
        """)
        await self._ensure_column("providers", "health_url", "health_url TEXT DEFAULT ''")
        await self._ensure_column("usage_logs", "subject_type", "subject_type TEXT DEFAULT 'api_key'")
        await self._ensure_column("usage_logs", "subject_id", "subject_id TEXT DEFAULT ''")
        await self._ensure_column("usage_logs", "user_id", "user_id TEXT DEFAULT ''")
        await self._ensure_column("usage_logs", "routing_profile_id", "routing_profile_id TEXT DEFAULT ''")
        await self._ensure_column("usage_logs", "request_id", "request_id TEXT DEFAULT ''")
        await self._ensure_column("cascade_stats", "subject_type", "subject_type TEXT DEFAULT 'api_key'")
        await self._ensure_column("cascade_stats", "subject_id", "subject_id TEXT DEFAULT ''")
        await self._ensure_column("cascade_stats", "user_id", "user_id TEXT DEFAULT ''")
        await self._ensure_column("cascade_stats", "routing_profile_id", "routing_profile_id TEXT DEFAULT ''")
        await self._ensure_column("cascade_stats", "request_id", "request_id TEXT DEFAULT ''")
        await self._ensure_column("keyword_rules", "routing_profile_id", "routing_profile_id INTEGER REFERENCES routing_profiles(id) ON DELETE CASCADE")
        await self._ensure_column("cascade_rules", "routing_profile_id", "routing_profile_id INTEGER REFERENCES routing_profiles(id) ON DELETE CASCADE")
        await self._ensure_column("cascade_rules", "local_judge_model", "local_judge_model TEXT")
        await self._ensure_column("cascade_rules", "local_judge_threshold", "local_judge_threshold REAL DEFAULT 0.7")
        await self._ensure_column("cascade_rules", "local_judge_system_prompt", "local_judge_system_prompt TEXT")
        await self._conn.commit()

    # ── seed ──

    async def seed_from_yaml(self, yaml_path: str = "config/providers.yaml"):
        apply_env_aliases()
        config_path = Path(yaml_path)
        if not config_path.exists():
            return
        raw = config_path.read_text(encoding="utf-8")
        raw = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ""), raw)
        data = yaml.safe_load(raw)
        if not data:
            return

        for name, pconf in data.get("providers", {}).items():
            base_url = (pconf.get("base_url") or "").strip()
            keys = [k for k in (pconf.get("keys") or []) if (k.get("key") or "").strip()]
            if not keys or not base_url:
                continue
            await self._execute(
                "INSERT OR IGNORE INTO providers (name,base_url,health_url,timeout,max_retries,key_strategy) VALUES (?,?,?,?,?,?)",
                (
                    name,
                    base_url,
                    (pconf.get("health_url") or "").strip(),
                    pconf.get("timeout", 0),
                    pconf.get("max_retries", 1),
                    pconf.get("key_strategy", "round_robin"),
                ),
            )
            for k in keys:
                await self._execute(
                    "INSERT INTO provider_keys (provider_name,key_value,rpm,tpm,weight) VALUES (?,?,?,?,?)",
                    (name, k["key"], k.get("rpm", 60), k.get("tpm"), k.get("weight", 1)),
                )
            for m in pconf.get("models") or []:
                pricing = m.get("pricing") or {}
                for rk in m.get("routing_keys") or []:
                    await self._execute(
                        "INSERT OR IGNORE INTO models (provider_name,model_name,routing_key,prompt_price,completion_price) VALUES (?,?,?,?,?)",
                        (name, m["name"], rk, pricing.get("prompt", 0), pricing.get("completion", 0)),
                    )

        for rule in data.get("keyword_rules") or []:
            await self._execute(
                "INSERT INTO keyword_rules (keyword,match_mode,target_routing_key,priority,note) VALUES (?,?,?,?,?)",
                (rule["keyword"], rule.get("match_mode", "contains"), rule["target_routing_key"], rule.get("priority", 0), rule.get("note", "")),
            )

        for rule in data.get("cascade_rules") or []:
            import json
            await self._execute(
                """INSERT INTO cascade_rules 
                (name,source_routing_key,escalation_chain,min_content_length,check_refusal,
                 refusal_patterns,check_finish_reason,local_judge_model,local_judge_threshold,
                 local_judge_system_prompt,custom_validator_url,custom_validator_timeout,
                 max_escalations,cost_threshold,priority,note) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rule["name"],
                    rule["source_routing_key"],
                    json.dumps(rule["escalation_chain"]),
                    rule.get("min_content_length", 10),
                    1 if rule.get("check_refusal", True) else 0,
                    json.dumps(rule.get("refusal_patterns", [])),
                    1 if rule.get("check_finish_reason", True) else 0,
                    rule.get("local_judge_model"),
                    rule.get("local_judge_threshold", 0.7),
                    rule.get("local_judge_system_prompt"),
                    rule.get("custom_validator_url"),
                    rule.get("custom_validator_timeout", 5000),
                    rule.get("max_escalations", 3),
                    rule.get("cost_threshold", 0.0),
                    rule.get("priority", 0),
                    rule.get("note", ""),
                ),
            )

        await self._conn.commit()

    # ── provider CRUD ──

    async def list_providers(self, enabled_only: bool = True):
        sql = "SELECT * FROM providers"
        params = ()
        if enabled_only:
            sql += " WHERE enabled=1"
        return await self._fetchall(sql, params)

    async def get_provider(self, name: str):
        return await self._fetchone("SELECT * FROM providers WHERE name=?", (name,))

    async def create_provider(self, name: str, base_url: str, timeout: int = 0,
                              max_retries: int = 3, key_strategy: str = "round_robin",
                              health_url: str = ""):
        await self._execute(
            "INSERT INTO providers (name,base_url,health_url,timeout,max_retries,key_strategy) VALUES (?,?,?,?,?,?)",
            (name, base_url, health_url, timeout, max_retries, key_strategy),
        )
        await self._conn.commit()

    async def update_provider(self, name: str, **kwargs):
        allowed = {"base_url", "health_url", "timeout", "max_retries", "key_strategy", "enabled"}
        pairs = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not pairs:
            return
        set_clause = ", ".join(f"{k}=?" for k, _ in pairs)
        vals = [v for _, v in pairs] + [name]
        await self._execute(f"UPDATE providers SET {set_clause} WHERE name=?", vals)
        await self._conn.commit()

    async def delete_provider(self, name: str):
        await self._execute("DELETE FROM providers WHERE name=?", (name,))
        await self._conn.commit()

    async def insert_provider_health_check(
        self,
        provider_name: str,
        source_url: str,
        status: str = "unknown",
        healthy: bool | None = None,
        score: float | None = None,
        latency_ms: float = 0,
        message: str = "",
        raw: dict | list | str | None = None,
    ):
        raw_json = ""
        if raw is not None:
            raw_json = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
        await self._execute(
            """INSERT INTO provider_health_checks
            (timestamp,provider_name,source_url,status,healthy,score,latency_ms,message,raw_json)
            VALUES (datetime('now'),?,?,?,?,?,?,?,?)""",
            (
                provider_name,
                source_url,
                status,
                None if healthy is None else (1 if healthy else 0),
                score,
                latency_ms,
                message,
                raw_json,
            ),
        )
        await self._conn.commit()

    async def list_provider_health_checks(self, provider_name: str | None = None, limit: int = 20):
        sql = "SELECT * FROM provider_health_checks"
        params = []
        if provider_name:
            sql += " WHERE provider_name=?"
            params.append(provider_name)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return await self._fetchall(sql, tuple(params))

    # ── provider keys CRUD ──

    async def list_provider_keys(self, provider_name: str):
        return await self._fetchall(
            "SELECT id,provider_name,key_value,rpm,tpm,weight,note,substr(key_value,-4) AS key_suffix FROM provider_keys WHERE provider_name=?",
            (provider_name,),
        )

    async def create_provider_key(self, provider_name: str, key_value: str, rpm: int = 60,
                                  tpm: int | None = None, weight: int = 1, note: str = ""):
        await self._execute(
            "INSERT INTO provider_keys (provider_name,key_value,rpm,tpm,weight,note) VALUES (?,?,?,?,?,?)",
            (provider_name, key_value, rpm, tpm, weight, note),
        )
        await self._conn.commit()

    async def delete_provider_key(self, key_id: int):
        await self._execute("DELETE FROM provider_keys WHERE id=?", (key_id,))
        await self._conn.commit()

    # ── routing profiles CRUD ──

    async def list_routing_profiles(self, enabled_only: bool = False):
        sql = "SELECT * FROM routing_profiles"
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY id ASC"
        return await self._fetchall(sql)

    async def get_routing_profile(self, profile_id: int):
        return await self._fetchone("SELECT * FROM routing_profiles WHERE id=?", (profile_id,))

    async def create_routing_profile(self, name: str, description: str = "", config_json: str = "{}") -> int:
        cursor = await self._execute(
            """INSERT INTO routing_profiles (name,description,config_json,updated_at)
            VALUES (?,?,?,datetime('now'))""",
            (name, description, config_json),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def update_routing_profile(self, profile_id: int, **kwargs):
        allowed = {"name", "description", "config_json", "enabled"}
        pairs = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not pairs:
            return
        set_clause = ", ".join(f"{k}=?" for k, _ in pairs)
        vals = [v for _, v in pairs]
        vals += [profile_id]
        await self._execute(
            f"UPDATE routing_profiles SET {set_clause}, updated_at=datetime('now') WHERE id=?",
            vals,
        )
        await self._conn.commit()

    async def delete_routing_profile(self, profile_id: int):
        await self._execute("DELETE FROM routing_profiles WHERE id=?", (profile_id,))
        await self._conn.commit()

    async def bind_api_key_routing_profile(self, api_key_id: int, routing_profile_id: int):
        await self._execute(
            """INSERT INTO routing_profile_api_keys (api_key_id,routing_profile_id)
            VALUES (?,?)
            ON CONFLICT(api_key_id) DO UPDATE SET routing_profile_id=excluded.routing_profile_id""",
            (api_key_id, routing_profile_id),
        )
        await self._conn.commit()

    async def unbind_api_key_routing_profile(self, api_key_id: int):
        await self._execute("DELETE FROM routing_profile_api_keys WHERE api_key_id=?", (api_key_id,))
        await self._conn.commit()

    async def get_routing_profile_for_api_key(self, api_key_id: int):
        return await self._fetchone(
            """SELECT rp.*
            FROM routing_profile_api_keys AS bind
            JOIN routing_profiles AS rp ON rp.id = bind.routing_profile_id
            WHERE bind.api_key_id=?""",
            (api_key_id,),
        )

    async def bind_user_routing_profile(self, user_id: str, routing_profile_id: int):
        await self._execute(
            """INSERT INTO routing_profile_users (user_id,routing_profile_id)
            VALUES (?,?)
            ON CONFLICT(user_id) DO UPDATE SET routing_profile_id=excluded.routing_profile_id""",
            (user_id, routing_profile_id),
        )
        await self._conn.commit()

    async def unbind_user_routing_profile(self, user_id: str):
        await self._execute("DELETE FROM routing_profile_users WHERE user_id=?", (user_id,))
        await self._conn.commit()

    async def get_routing_profile_for_user(self, user_id: str):
        return await self._fetchone(
            """SELECT rp.*
            FROM routing_profile_users AS bind
            JOIN routing_profiles AS rp ON rp.id = bind.routing_profile_id
            WHERE bind.user_id=?""",
            (user_id,),
        )

    # ── model CRUD ──

    async def list_models(self, enabled_only: bool = True):
        sql = "SELECT * FROM models"
        params = ()
        if enabled_only:
            sql += " WHERE enabled=1"
        return await self._fetchall(sql, params)

    async def get_models_by_provider(self, provider_name: str, enabled_only: bool = True):
        sql = "SELECT * FROM models WHERE provider_name=?"
        params = [provider_name]
        if enabled_only:
            sql += " AND enabled=1"
        return await self._fetchall(sql, tuple(params))

    async def create_model(self, provider_name: str, model_name: str, routing_key: str,
                           prompt_price: float = 0, completion_price: float = 0):
        await self._execute(
            "INSERT INTO models (provider_name,model_name,routing_key,prompt_price,completion_price) VALUES (?,?,?,?,?)",
            (provider_name, model_name, routing_key, prompt_price, completion_price),
        )
        await self._conn.commit()

    async def sync_provider_models(self, provider_name: str, model_names: list[str]):
        cleaned = []
        seen = set()
        for model_name in model_names:
            value = (model_name or "").strip()
            if value and value not in seen:
                seen.add(value)
                cleaned.append(value)

        if not cleaned:
            return {"created": 0, "deleted": 0, "preserved_aliases": 0}

        rows = await self.get_models_by_provider(provider_name, enabled_only=False)
        stale_row_ids = {
            row["id"]
            for row in rows
            if row["model_name"] not in seen or row["routing_key"] != row["model_name"]
        }
        created = 0

        for row_id in stale_row_ids:
            await self._execute("DELETE FROM models WHERE id=?", (row_id,))

        current_rows = await self.get_models_by_provider(provider_name, enabled_only=False)
        rows_by_model = {row["model_name"]: row for row in current_rows if row["routing_key"] == row["model_name"]}

        for model_name in cleaned:
            canonical = rows_by_model.get(model_name)
            if canonical:
                await self._execute(
                    "UPDATE models SET enabled=1 WHERE id=?",
                    (canonical["id"],),
                )
                continue

            prompt_price = 0
            completion_price = 0
            await self._execute(
                "INSERT INTO models (provider_name,model_name,routing_key,prompt_price,completion_price,enabled) VALUES (?,?,?,?,?,1)",
                (provider_name, model_name, model_name, prompt_price, completion_price),
            )
            created += 1

        await self._conn.commit()
        return {
            "created": created,
            "deleted": len(stale_row_ids),
            "preserved_aliases": 0,
        }

    async def update_model(self, model_id: int, **kwargs):
        allowed = {"model_name", "routing_key", "prompt_price", "completion_price", "enabled"}
        pairs = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not pairs:
            return
        set_clause = ", ".join(f"{k}=?" for k, _ in pairs)
        vals = [v for _, v in pairs] + [model_id]
        await self._execute(f"UPDATE models SET {set_clause} WHERE id=?", vals)
        await self._conn.commit()

    async def delete_model(self, model_id: int):
        await self._execute("DELETE FROM models WHERE id=?", (model_id,))
        await self._conn.commit()

    # ── keyword rules CRUD ──

    async def list_keyword_rules(
        self,
        enabled_only: bool = False,
        routing_profile_id: str | int | None = None,
        include_global: bool = True,
    ):
        conditions = []
        params = []
        if enabled_only:
            conditions.append("enabled=1")
        if routing_profile_id is None:
            pass
        elif str(routing_profile_id).isdigit():
            if include_global:
                conditions.append("(routing_profile_id=? OR routing_profile_id IS NULL)")
                params.append(int(routing_profile_id))
            else:
                conditions.append("routing_profile_id=?")
                params.append(int(routing_profile_id))
        elif include_global:
            conditions.append("routing_profile_id IS NULL")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM keyword_rules{where}
            ORDER BY
                CASE WHEN routing_profile_id IS NULL THEN 0 ELSE 1 END DESC,
                priority DESC,
                id ASC
        """
        return await self._fetchall(sql, tuple(params))

    async def create_keyword_rule(self, keyword: str, target_routing_key: str,
                                  match_mode: str = "contains", priority: int = 0,
                                  note: str = "", routing_profile_id: int | None = None):
        await self._execute(
            """INSERT INTO keyword_rules
            (routing_profile_id,keyword,match_mode,target_routing_key,priority,note)
            VALUES (?,?,?,?,?,?)""",
            (routing_profile_id, keyword, match_mode, target_routing_key, priority, note),
        )
        await self._conn.commit()

    async def update_keyword_rule(self, rule_id: int, **kwargs):
        allowed = {"routing_profile_id", "keyword", "match_mode", "target_routing_key", "priority", "enabled", "note"}
        pairs = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not pairs:
            return
        set_clause = ", ".join(f"{k}=?" for k, _ in pairs)
        vals = [v for _, v in pairs] + [rule_id]
        await self._execute(f"UPDATE keyword_rules SET {set_clause} WHERE id=?", vals)
        await self._conn.commit()

    async def delete_keyword_rule(self, rule_id: int):
        await self._execute("DELETE FROM keyword_rules WHERE id=?", (rule_id,))
        await self._conn.commit()

    # ── cascade rules CRUD ──

    async def list_cascade_rules(
        self,
        enabled_only: bool = False,
        routing_profile_id: str | int | None = None,
        include_global: bool = True,
    ):
        conditions = []
        params = []
        if enabled_only:
            conditions.append("enabled=1")
        if routing_profile_id is None:
            pass
        elif str(routing_profile_id).isdigit():
            if include_global:
                conditions.append("(routing_profile_id=? OR routing_profile_id IS NULL)")
                params.append(int(routing_profile_id))
            else:
                conditions.append("routing_profile_id=?")
                params.append(int(routing_profile_id))
        elif include_global:
            conditions.append("routing_profile_id IS NULL")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM cascade_rules{where}
            ORDER BY
                CASE WHEN routing_profile_id IS NULL THEN 0 ELSE 1 END DESC,
                priority DESC,
                id ASC
        """
        return await self._fetchall(sql, tuple(params))

    async def get_cascade_rule(self, source_routing_key: str, routing_profile_id: str | int | None = None):
        conditions = ["source_routing_key=?", "enabled=1"]
        params = [source_routing_key]
        if routing_profile_id is not None and str(routing_profile_id).isdigit():
            conditions.append("(routing_profile_id=? OR routing_profile_id IS NULL)")
            params.append(int(routing_profile_id))
        else:
            conditions.append("routing_profile_id IS NULL")
        where = " AND ".join(conditions)
        return await self._fetchone(
            f"""SELECT * FROM cascade_rules
            WHERE {where}
            ORDER BY CASE WHEN routing_profile_id IS NULL THEN 0 ELSE 1 END DESC, priority DESC, id ASC
            LIMIT 1""",
            tuple(params),
        )

    async def create_cascade_rule(self, name: str, source_routing_key: str, escalation_chain: str,
                                  min_content_length: int = 10, check_refusal: bool = True,
                                  refusal_patterns: str = "[]", check_finish_reason: bool = True,
                                  local_judge_model: str | None = None, local_judge_threshold: float = 0.7,
                                  local_judge_system_prompt: str | None = None,
                                  custom_validator_url: str | None = None, custom_validator_timeout: int = 5000,
                                  max_escalations: int = 3, cost_threshold: float = 0.0,
                                  priority: int = 0, note: str = "", routing_profile_id: int | None = None):
        await self._execute(
            """INSERT INTO cascade_rules 
            (routing_profile_id,name,source_routing_key,escalation_chain,min_content_length,check_refusal,
             refusal_patterns,check_finish_reason,local_judge_model,local_judge_threshold,
             local_judge_system_prompt,custom_validator_url,custom_validator_timeout,
             max_escalations,cost_threshold,priority,note) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (routing_profile_id, name, source_routing_key, escalation_chain, min_content_length, check_refusal,
             refusal_patterns, check_finish_reason, local_judge_model, local_judge_threshold,
             local_judge_system_prompt, custom_validator_url, custom_validator_timeout,
             max_escalations, cost_threshold, priority, note),
        )
        await self._conn.commit()

    async def update_cascade_rule(self, rule_id: int, **kwargs):
        allowed = {
            "routing_profile_id", "name", "source_routing_key", "escalation_chain", "min_content_length",
            "check_refusal", "refusal_patterns", "check_finish_reason",
            "local_judge_model", "local_judge_threshold", "local_judge_system_prompt",
            "custom_validator_url", "custom_validator_timeout", "max_escalations",
            "cost_threshold", "priority", "enabled", "note"
        }
        pairs = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not pairs:
            return
        set_clause = ", ".join(f"{k}=?" for k, _ in pairs)
        vals = [v for _, v in pairs] + [rule_id]
        await self._execute(f"UPDATE cascade_rules SET {set_clause} WHERE id=?", vals)
        await self._conn.commit()

    async def delete_cascade_rule(self, rule_id: int):
        await self._execute("DELETE FROM cascade_rules WHERE id=?", (rule_id,))
        await self._conn.commit()

    # ── cascade stats ──

    async def insert_cascade_stat(self, row: dict):
        await self._execute(
            """INSERT INTO cascade_stats 
            (timestamp,api_key_id,subject_type,subject_id,user_id,routing_profile_id,request_id,
             original_model,models_attempted,final_model,total_escalations,validation_reasons,
             total_cost,total_latency_ms,status) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["timestamp"], row.get("api_key_id", 0), row.get("subject_type", "api_key"),
                row.get("subject_id", ""), row.get("user_id"), row.get("routing_profile_id"),
                row.get("request_id"), row["original_model"],
                row["models_attempted"], row["final_model"], row.get("total_escalations", 0),
                row.get("validation_reasons"), row.get("total_cost", 0),
                row.get("total_latency_ms", 0), row.get("status", "success"),
            ),
        )
        await self._conn.commit()

    async def query_cascade_stats(self, from_date: str = None, to_date: str = None,
                                  model: str = None, limit: int = 100, offset: int = 0):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        if model:
            conditions.append("(original_model = ? OR final_model = ?)")
            params.extend([model, model])
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM cascade_stats{where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return await self._fetchall(sql, tuple(params))

    async def cascade_stats_summary(self, from_date: str = None, to_date: str = None):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(total_escalations),0) AS total_escalations,
                COALESCE(SUM(total_cost),0) AS total_cost,
                COALESCE(AVG(total_latency_ms),0) AS avg_latency_ms
            FROM cascade_stats{where}
        """
        return await self._fetchone(sql, tuple(params))

    async def cascade_stats_daily(self, from_date: str = None, to_date: str = None):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT DATE(timestamp) AS day, final_model,
                COUNT(*) AS requests,
                COALESCE(SUM(total_escalations),0) AS escalations,
                COALESCE(SUM(total_cost),0) AS cost,
                COALESCE(AVG(total_latency_ms),0) AS avg_latency
            FROM cascade_stats{where}
            GROUP BY DATE(timestamp), final_model
            ORDER BY day DESC
        """
        return await self._fetchall(sql, tuple(params))

    # ── settings CRUD ──

    async def get_setting(self, key: str, default: str = ""):
        row = await self._fetchone("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str):
        await self._execute(
            "INSERT INTO settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        await self._conn.commit()

    async def get_all_settings(self) -> dict:
        rows = await self._fetchall("SELECT key, value FROM settings")
        return {r["key"]: r["value"] for r in rows}

    async def get_effective_routing_settings(
        self,
        routing_profile_id: str | None = None,
        defaults: dict | None = None,
    ) -> dict:
        defaults = defaults or {}
        settings = await self.get_all_settings()
        result = {
            "keyword_routing_enabled": settings.get(
                "keyword_routing_enabled",
                "true" if defaults.get("keyword_routing_enabled", False) else "false",
            ) == "true",
            "semantic_routing_enabled": settings.get(
                "semantic_routing_enabled",
                "true" if defaults.get("semantic_routing_enabled", False) else "false",
            ) == "true",
            "cascade_enabled": settings.get(
                "cascade_enabled",
                "true" if defaults.get("cascade_enabled", False) else "false",
            ) == "true",
        }
        if routing_profile_id and str(routing_profile_id).isdigit():
            profile = await self.get_routing_profile(int(routing_profile_id))
            if profile and profile["enabled"]:
                try:
                    profile_config = json.loads(profile["config_json"] or "{}")
                except json.JSONDecodeError:
                    profile_config = {}
                for key in ("keyword_routing_enabled", "semantic_routing_enabled", "cascade_enabled"):
                    if key in profile_config:
                        result[key] = bool(profile_config[key])
        return result

    async def update_routing_profile_settings(self, profile_id: int, updates: dict):
        profile = await self.get_routing_profile(profile_id)
        if not profile:
            raise ValueError(f"Routing profile {profile_id} not found")
        try:
            config = json.loads(profile["config_json"] or "{}")
        except json.JSONDecodeError:
            config = {}
        config.update(updates)
        await self.update_routing_profile(profile_id, config_json=json.dumps(config, ensure_ascii=False))

    # ── api_keys CRUD (user-facing) ──

    async def list_api_keys(self):
        return await self._fetchall(
            "SELECT id,note,rate_limit_rpm,allowed_models,enabled,created_at,substr(key_value,-4) AS key_suffix FROM api_keys"
        )

    async def get_api_key(self, key_value: str):
        return await self._fetchone("SELECT * FROM api_keys WHERE key_value=? AND enabled=1", (key_value,))

    async def create_api_key(self, key_value: str, note: str = "",
                             rate_limit_rpm: int = 60, allowed_models: str = "*"):
        await self._execute(
            "INSERT INTO api_keys (key_value,note,rate_limit_rpm,allowed_models) VALUES (?,?,?,?)",
            (key_value, note, rate_limit_rpm, allowed_models),
        )
        await self._conn.commit()

    async def delete_api_key(self, key_id: int):
        await self._execute("UPDATE api_keys SET enabled=0 WHERE id=?", (key_id,))
        await self._conn.commit()

    # ── usage logs ──

    async def insert_usage(self, row: dict):
        await self._execute(
            """INSERT INTO usage_logs
            (timestamp,api_key_id,subject_type,subject_id,user_id,routing_profile_id,request_id,
             model,provider,prompt_tokens,completion_tokens,total_tokens,cost,latency_ms,status,error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["timestamp"], row.get("api_key_id", 0), row.get("subject_type", "api_key"),
                row.get("subject_id", ""), row.get("user_id"), row.get("routing_profile_id"),
                row.get("request_id"), row["model"], row["provider"],
                row["prompt_tokens"], row["completion_tokens"], row["total_tokens"],
                row["cost"], row["latency_ms"], row["status"], row.get("error"),
            ),
        )
        await self._conn.commit()
        await self._maybe_clean_usage()

    async def _maybe_clean_usage(self):
        c = await self._conn.execute("SELECT COUNT(*) FROM usage_logs")
        count = (await c.fetchone())[0]
        if count <= MAX_USAGE_ROWS:
            return
        excess = count - MAX_USAGE_ROWS
        await self._execute(
            "DELETE FROM usage_logs WHERE id IN (SELECT id FROM usage_logs ORDER BY id LIMIT ?)",
            (excess,),
        )
        await self._conn.commit()

    async def query_usage(self, from_date: str = None, to_date: str = None,
                          model: str = None, status: str = None, limit: int = 100, offset: int = 0):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM usage_logs{where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return await self._fetchall(sql, tuple(params))

    async def usage_summary(self, from_date: str = None, to_date: str = None):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(total_tokens),0) AS total_tokens,
                COALESCE(SUM(prompt_tokens),0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens),0) AS total_completion_tokens,
                COALESCE(SUM(cost),0) AS total_cost,
                COALESCE(SUM(CASE WHEN status='error' THEN 1 ELSE 0 END),0) AS total_errors,
                COALESCE(AVG(latency_ms),0) AS avg_latency_ms
            FROM usage_logs{where}
        """
        return await self._fetchone(sql, tuple(params))

    async def usage_daily(self, from_date: str = None, to_date: str = None):
        conditions = []
        params = []
        if from_date:
            conditions.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("timestamp <= ?")
            params.append(to_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT DATE(timestamp) AS day, model,
                COUNT(*) AS requests,
                COALESCE(SUM(total_tokens),0) AS tokens,
                COALESCE(SUM(cost),0) AS cost,
                COALESCE(AVG(latency_ms),0) AS avg_latency
            FROM usage_logs{where}
            GROUP BY DATE(timestamp), model
            ORDER BY day DESC
        """
        return await self._fetchall(sql, tuple(params))
