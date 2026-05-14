import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import edge_tts
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

load_dotenv()

# ============================================
# АДАПТАЦИЯ ДЛЯ RENDER: ПОСТОЯННОЕ ХРАНИЛИЩЕ
# ============================================
RENDER_DATA_DIR = os.getenv("RENDER_DATA_DIR", "/opt/render/project/src/data")
os.makedirs(RENDER_DATA_DIR, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(RENDER_DATA_DIR) / "tutor.db"
INDEX_PATH = BASE_DIR / "index.html"

# Перенос существующей БД при первом запуске
OLD_DB_PATH = BASE_DIR / "tutor.db"
if OLD_DB_PATH.exists() and not DB_PATH.exists():
    shutil.copy2(OLD_DB_PATH, DB_PATH)
    print(f"✅ База данных перенесена в {DB_PATH}")
# ============================================


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_csv(name: str, default: str = "") -> List[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
    YOOKASSA_SHOP_ID = os.getenv("YOOKASSA_SHOP_ID", "").strip()
    YOOKASSA_SECRET_KEY = os.getenv("YOOKASSA_SECRET_KEY", "").strip()
    BASE_SERVER_URL = os.getenv("BASE_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
    YOOKASSA_API_URL = "https://api.yookassa.ru/v3/payments"
    OLLAMA_DIRECT_API_URL = os.getenv("OLLAMA_DIRECT_API_URL", "https://ollama.com/api/chat")

    MODEL_BASIC = os.getenv("MODEL_BASIC", "gpt-oss:120b-cloud")
    MODEL_PREMIUM = os.getenv("MODEL_PREMIUM", "deepseek-v3.1:671b-cloud")

    PRICE_BASIC = int(os.getenv("PRICE_BASIC", "300"))
    PRICE_PREMIUM = int(os.getenv("PRICE_PREMIUM", "500"))
    SESSIONS_PER_MONTH = int(os.getenv("SESSIONS_PER_MONTH", "5"))
    SESSION_DURATION_MINUTES = int(os.getenv("SESSION_DURATION_MINUTES", "120"))
    MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", "20"))
    MAX_TTS_TEXT_LENGTH = int(os.getenv("MAX_TTS_TEXT_LENGTH", "100000000"))

    ALLOWED_ORIGINS = env_csv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,null"
    )
    TRUST_PROXY_HEADERS = env_flag("TRUST_PROXY_HEADERS", False)
    ENABLE_TEST_PAYMENTS = env_flag("ENABLE_TEST_PAYMENTS", True)

    @classmethod
    def yookassa_enabled(cls) -> bool:
        placeholder_markers = ("shop_id", "secret_key", "ваш", "your_")
        shop_id = cls.YOOKASSA_SHOP_ID.lower()
        secret_key = cls.YOOKASSA_SECRET_KEY.lower()
        return (
            bool(cls.YOOKASSA_SHOP_ID and cls.YOOKASSA_SECRET_KEY)
            and not any(marker in shop_id for marker in placeholder_markers)
            and not any(marker in secret_key for marker in placeholder_markers)
        )


config = Config()


class Database:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Оптимизации SQLite
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-2000")
        conn.execute("PRAGMA temp_store=MEMORY")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def execute(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                return conn.execute(query, params)

    async def fetch_one(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchone()

    async def fetch_all(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()

    async def init(self):
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    password_hash TEXT,
                    user_api_key TEXT UNIQUE NOT NULL,
                    plan TEXT DEFAULT 'basic',
                    subscription_end TEXT,
                    sessions_used_this_month INTEGER DEFAULT 0,
                    current_session_start TEXT,
                    last_session_month TEXT,
                    total_questions INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            required = {
                "password_hash": "TEXT",
                "user_api_key": "TEXT",
                "plan": "TEXT DEFAULT 'basic'",
                "subscription_end": "TEXT",
                "sessions_used_this_month": "INTEGER DEFAULT 0",
                "current_session_start": "TEXT",
                "last_session_month": "TEXT",
                "total_questions": "INTEGER DEFAULT 0",
            }
            for column_name, column_type in required.items():
                if column_name not in columns:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    amount INTEGER NOT NULL,
                    plan TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    payment_id TEXT UNIQUE,
                    yookassa_id TEXT,
                    confirmation_url TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    paid_at TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip TEXT PRIMARY KEY,
                    requests_count INTEGER DEFAULT 1,
                    first_request TEXT,
                    blocked_until TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temp_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tts_sessions (
                    id TEXT PRIMARY KEY,
                    completed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


db = Database()


class RegisterRequest(BaseModel):
    name: str
    password: str
    plan: str = "basic"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip().upper()
        if len(value) < 2:
            raise ValueError("Имя слишком короткое")
        if len(value) > 30:
            raise ValueError("Имя слишком длинное")
        return value

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, value):
        if value not in {"basic", "premium"}:
            raise ValueError("Неверный тариф")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value):
        value = value.strip()
        if len(value) < 4:
            raise ValueError("Пароль должен быть минимум 4 символа")
        if len(value) > 128:
            raise ValueError("Пароль слишком длинный")
        return value


class LoginRequest(BaseModel):
    name: str
    password: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        return RegisterRequest.validate_name(value)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value):
        return RegisterRequest.validate_password(value)


class ChatRequest(BaseModel):
    messages: list
    tts_id: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        if not isinstance(value, list) or not value:
            raise ValueError("Сообщения не переданы")
        if len(value) > config.MAX_CHAT_MESSAGES:
            raise ValueError(f"Слишком длинный диалог, максимум {config.MAX_CHAT_MESSAGES} сообщений")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Каждое сообщение должно быть объектом")
            if item.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Некорректная роль сообщения")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Пустое сообщение")
        return value


class PaymentRequest(BaseModel):
    user_api_key: str
    plan: str

    @field_validator("user_api_key")
    @classmethod
    def validate_api_key(cls, value):
        value = value.strip()
        if not value.startswith("lenya_"):
            raise ValueError("Некорректный API ключ")
        return value

    @field_validator("plan")
    @classmethod
    def validate_payment_plan(cls, value):
        if value not in {"basic", "premium"}:
            raise ValueError("Неверный тариф")
        return value


class TTSRequest(BaseModel):
    text: str
    voice: str = "ru-RU-DmitryNeural"

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Пустой текст")
        if len(value) > config.MAX_TTS_TEXT_LENGTH:
            raise ValueError(f"Текст слишком длинный, максимум {config.MAX_TTS_TEXT_LENGTH} символов")
        return value


class SecurityUtils:
    @staticmethod
    def get_client_ip(request: Request) -> str:
        if config.TRUST_PROXY_HEADERS:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()
        return request.client.host if request.client else "127.0.0.1"

    @staticmethod
    def generate_api_key() -> str:
        return f"lenya_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return f"{salt.hex()}${hashed.hex()}"

    @staticmethod
    def verify_password(password: str, password_hash: str | None) -> bool:
        if not password_hash or "$" not in password_hash:
            return False
        salt_hex, hash_hex = password_hash.split("$", 1)
        try:
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(hash_hex)
        except ValueError:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(actual, expected)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def subscription_is_active(subscription_end: str | None) -> bool:
    parsed = parse_datetime(subscription_end)
    return bool(parsed and parsed >= datetime.utcnow())


class SessionManager:
    @staticmethod
    async def reset_monthly_sessions(user_id: int):
        current_month = datetime.utcnow().strftime("%Y-%m")
        user = await db.fetch_one(
            "SELECT last_session_month FROM users WHERE id = ?",
            (user_id,)
        )
        if user and user["last_session_month"] != current_month:
            await db.execute(
                "UPDATE users SET sessions_used_this_month = 0, last_session_month = ? WHERE id = ?",
                (current_month, user_id)
            )
            return True
        return False

    @staticmethod
    async def finish_expired_session(user_id: int, started_at: str):
        await db.execute(
            """
            UPDATE users
            SET sessions_used_this_month = sessions_used_this_month + 1,
                current_session_start = NULL
            WHERE id = ? AND current_session_start = ?
            """,
            (user_id, started_at)
        )

    @staticmethod
    async def validate_and_prepare_session(user: Dict[str, Any]) -> Dict[str, Any]:
        user_id = user["id"]
        await SessionManager.reset_monthly_sessions(user_id)
        user_row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        user = dict(user_row)

        if not subscription_is_active(user.get("subscription_end")):
            if user.get("subscription_end"):
                raise HTTPException(403, "Подписка истекла. Продлите тариф.")
            raise HTTPException(403, "Нет активной подписки. Сначала оплатите тариф.")

        current_session_start = user.get("current_session_start")
        if current_session_start:
            session_start = parse_datetime(current_session_start)
            if session_start and datetime.utcnow() - session_start > timedelta(minutes=config.SESSION_DURATION_MINUTES):
                await SessionManager.finish_expired_session(user_id, current_session_start)
                user_row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
                user = dict(user_row)

        if user.get("sessions_used_this_month", 0) >= config.SESSIONS_PER_MONTH:
            raise HTTPException(403, f"Лимит исчерпан: {config.SESSIONS_PER_MONTH} занятий в месяц.")

        return user


class RateLimiter:
    @staticmethod
    async def check_and_update(ip: str, limit: int = 30, window: int = 60, block_time: int = 300):
        now = datetime.utcnow()
        rate = await db.fetch_one("SELECT * FROM rate_limits WHERE ip = ?", (ip,))

        if not rate:
            await db.execute(
                "INSERT INTO rate_limits (ip, first_request) VALUES (?, ?)",
                (ip, now.isoformat())
            )
            return

        blocked_until = parse_datetime(rate["blocked_until"])
        if blocked_until and blocked_until > now:
            seconds_left = int((blocked_until - now).total_seconds())
            raise HTTPException(429, f"Слишком много запросов. Попробуйте через {seconds_left} сек.")

        first_request = parse_datetime(rate["first_request"])
        if not first_request or (now - first_request).total_seconds() > window:
            await db.execute(
                "UPDATE rate_limits SET requests_count = 1, first_request = ?, blocked_until = NULL WHERE ip = ?",
                (now.isoformat(), ip)
            )
            return

        new_count = rate["requests_count"] + 1
        if new_count > limit:
            blocked_until = (now + timedelta(seconds=block_time)).isoformat()
            await db.execute(
                "UPDATE rate_limits SET requests_count = ?, blocked_until = ? WHERE ip = ?",
                (new_count, blocked_until, ip)
            )
            raise HTTPException(429, f"IP заблокирован на {block_time} секунд")

        await db.execute(
            "UPDATE rate_limits SET requests_count = ? WHERE ip = ?",
            (new_count, ip)
        )


async def verify_user(x_api_key: str = Header(...)):
    user = await db.fetch_one(
        "SELECT * FROM users WHERE user_api_key = ?",
        (x_api_key,)
    )
    if not user:
        raise HTTPException(401, "Неверный API ключ")
    return dict(user)


async def verify_active_session(user: dict = Depends(verify_user)):
    user = await SessionManager.validate_and_prepare_session(user)
    if not user.get("current_session_start"):
        raise HTTPException(403, "Нет активного занятия. Нажмите «Начать занятие».")
    return user


def render_payment_result(title: str, message: str, accent: str = "#4caf50") -> HTMLResponse:
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{
                    font-family: Inter, Arial, sans-serif;
                    background: linear-gradient(135deg, #1a1a2e, #16213e);
                    color: white;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .card {{
                    width: min(440px, 100%);
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 24px;
                    padding: 32px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                }}
                h1 {{ color: {accent}; margin-bottom: 16px; }}
                p {{ opacity: 0.92; line-height: 1.5; }}
                .btn {{
                    margin-top: 20px;
                    padding: 14px 24px;
                    border: none;
                    border-radius: 999px;
                    background: white;
                    color: #1a1a2e;
                    font-weight: 700;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>{title}</h1>
                <p>{message}</p>
                <button class="btn" onclick="window.close()">Закрыть окно</button>
            </div>
        </body>
        </html>
        """
    )


async def activate_payment(payment_id: str):
    payment = await db.fetch_one(
        "SELECT * FROM payments WHERE payment_id = ?",
        (payment_id,)
    )
    if not payment:
        return None, False
    if payment["status"] == "success":
        return payment, False

    subscription_end = (datetime.utcnow() + timedelta(days=30)).isoformat()
    current_month = datetime.utcnow().strftime("%Y-%m")

    await db.execute(
        """
        UPDATE users
        SET plan = ?,
            subscription_end = ?,
            sessions_used_this_month = 0,
            last_session_month = ?,
            current_session_start = NULL
        WHERE id = ?
        """,
        (payment["plan"], subscription_end, current_month, payment["user_id"])
    )
    await db.execute(
        """
        UPDATE payments
        SET status = 'success', paid_at = ?
        WHERE payment_id = ?
        """,
        (datetime.utcnow().isoformat(), payment_id)
    )
    updated_payment = await db.fetch_one(
        "SELECT * FROM payments WHERE payment_id = ?",
        (payment_id,)
    )
    return updated_payment, True


def resolve_ollama_model(plan: str | None) -> str:
    model = config.MODEL_PREMIUM if plan == "premium" else config.MODEL_BASIC
    return model.removesuffix("-cloud")


def build_ai_unavailable_response(message: str, model: str, user: dict) -> dict:
    start = parse_datetime(user.get("current_session_start"))
    elapsed_minutes = int((datetime.utcnow() - start).total_seconds() / 60) if start else 0
    remaining = max(0, config.SESSION_DURATION_MINUTES - elapsed_minutes)
    return {
        "response": message,
        "remaining_minutes": remaining,
        "model_used": model,
        "temporary_error": True,
    }


async def send_ollama_chat(model: str, messages: list) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            config.OLLAMA_DIRECT_API_URL,
            json={"model": model, "messages": messages, "stream": False},
            headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"}
        )

    if response.status_code != 200:
        detail = response.text[:500]
        raise HTTPException(502, f"AI provider error {response.status_code}: {detail}")

    return response.json()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await db.init()
    await cleanup_old_temp_files()
    print("=" * 70)
    print("Лёня AI Tutor запущен на Render")
    print(f"URL: {config.BASE_SERVER_URL}")
    print(f"DB: {DB_PATH}")
    print("=" * 70)
    yield


app = FastAPI(
    title="Лёня AI Tutor",
    description="AI репетитор с голосом, подпиской и лимитами занятий",
    version="7.2.2-render",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS or ["http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-ID"],
)


async def cleanup_old_temp_files():
    try:
        old_files = await db.fetch_all(
            "SELECT file_path FROM temp_files WHERE created_at < datetime('now', '-1 hour')"
        )
        for file in old_files:
            file_path = file["file_path"]
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        await db.execute("DELETE FROM temp_files WHERE created_at < datetime('now', '-1 hour')")
        await db.execute("DELETE FROM tts_sessions WHERE created_at < datetime('now', '-1 hour')")
    except Exception:
        pass


async def delete_file_later(filepath: str, delay: int):
    await asyncio.sleep(delay)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        await db.execute("DELETE FROM temp_files WHERE file_path = ?", (filepath,))
    except Exception:
        pass


@app.get("/")
async def root():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return {
        "service": "Лёня AI Tutor",
        "version": app.version,
        "status": "frontend file missing",
    }


@app.get("/api")
async def api_info():
    return {
        "service": "Лёня AI Tutor",
        "version": app.version,
        "test_payments_enabled": config.ENABLE_TEST_PAYMENTS,
        "payments_mode": "live" if config.yookassa_enabled() else "test",
        "hosting": "Render",
        "db_path": str(DB_PATH),
    }


@app.get("/health")
async def health():
    try:
        await db.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "connected",
            "db_path": str(DB_PATH),
            "timestamp": datetime.utcnow().isoformat(),
            "ollama_configured": bool(config.OLLAMA_API_KEY),
            "payments_mode": "live" if config.yookassa_enabled() else "test",
        }
    except Exception:
        return {"status": "degraded", "database": "disconnected"}


@app.post("/register")
async def register(request: RegisterRequest):
    existing = await db.fetch_one(
        "SELECT * FROM users WHERE name = ?",
        (request.name,)
    )
    if existing:
        if existing["password_hash"]:
            if not SecurityUtils.verify_password(request.password, existing["password_hash"]):
                raise HTTPException(status_code=401, detail="Неверные имя или пароль")
        else:
            await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (SecurityUtils.hash_password(request.password), existing["id"])
            )
        return {
            "name": request.name,
            "user_api_key": existing["user_api_key"],
            "plan": existing["plan"] or "basic",
            "has_subscription": subscription_is_active(existing["subscription_end"]),
            "message": "Вход выполнен успешно",
        }

    user_api_key = SecurityUtils.generate_api_key()
    password_hash = SecurityUtils.hash_password(request.password)
    current_month = datetime.utcnow().strftime("%Y-%m")
    await db.execute(
        """
        INSERT INTO users (name, password_hash, user_api_key, plan, last_session_month)
        VALUES (?, ?, ?, ?, ?)
        """,
        (request.name, password_hash, user_api_key, request.plan, current_month)
    )
    return {
        "name": request.name,
        "user_api_key": user_api_key,
        "plan": request.plan,
        "has_subscription": False,
        "message": f"Регистрация успешна. Оплатите тариф {request.plan.upper()} для доступа.",
    }


@app.post("/login")
async def login(request: LoginRequest):
    user = await db.fetch_one(
        "SELECT * FROM users WHERE name = ?",
        (request.name,)
    )
    if not user:
        raise HTTPException(status_code=404, detail="Аккаунт не найден")

    if user["password_hash"]:
        if not SecurityUtils.verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Неверные имя или пароль")
    else:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (SecurityUtils.hash_password(request.password), user["id"])
        )
        user = await db.fetch_one(
            "SELECT * FROM users WHERE id = ?",
            (user["id"],)
        )

    return {
        "name": user["name"],
        "user_api_key": user["user_api_key"],
        "plan": user["plan"] or "basic",
        "has_subscription": subscription_is_active(user["subscription_end"]),
        "message": "Вход выполнен успешно",
    }


@app.post("/session/start")
async def start_session(user: dict = Depends(verify_user)):
    user = await SessionManager.validate_and_prepare_session(user)
    current_session_start = user.get("current_session_start")
    if current_session_start:
        session_start = parse_datetime(current_session_start)
        if session_start:
            elapsed = datetime.utcnow() - session_start
            if elapsed < timedelta(minutes=config.SESSION_DURATION_MINUTES):
                remaining = timedelta(minutes=config.SESSION_DURATION_MINUTES) - elapsed
                return {
                    "session_active": True,
                    "started_at": current_session_start,
                    "elapsed_minutes": int(elapsed.total_seconds() / 60),
                    "remaining_minutes": int(remaining.total_seconds() / 60),
                    "message": "Занятие уже активно",
                }

    now = datetime.utcnow().isoformat()
    await db.execute(
        "UPDATE users SET current_session_start = ? WHERE id = ?",
        (now, user["id"])
    )
    return {
        "session_active": True,
        "started_at": now,
        "duration_hours": config.SESSION_DURATION_MINUTES // 60,
        "remaining_minutes": config.SESSION_DURATION_MINUTES,
        "message": "Занятие начато",
    }


@app.get("/session/status")
async def session_status(user: dict = Depends(verify_user)):
    await SessionManager.reset_monthly_sessions(user["id"])
    user_row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user["id"],))
    user = dict(user_row)

    has_subscription = subscription_is_active(user.get("subscription_end"))
    sessions_used = user.get("sessions_used_this_month", 0)
    sessions_left = max(0, config.SESSIONS_PER_MONTH - sessions_used)

    if not user.get("current_session_start"):
        return {
            "session_active": False,
            "has_subscription": has_subscription,
            "plan": user.get("plan", "basic"),
            "sessions_used": sessions_used,
            "sessions_left": sessions_left,
        }

    start = parse_datetime(user["current_session_start"])
    if not start:
        return {
            "session_active": False,
            "has_subscription": has_subscription,
            "plan": user.get("plan", "basic"),
            "sessions_used": sessions_used,
            "sessions_left": sessions_left,
        }

    elapsed = datetime.utcnow() - start
    remaining = timedelta(minutes=config.SESSION_DURATION_MINUTES) - elapsed
    if remaining.total_seconds() <= 0:
        await SessionManager.finish_expired_session(user["id"], user["current_session_start"])
        sessions_used = min(config.SESSIONS_PER_MONTH, sessions_used + 1)
        return {
            "session_active": False,
            "message": "Занятие завершено",
            "has_subscription": has_subscription,
            "plan": user.get("plan", "basic"),
            "sessions_used": sessions_used,
            "sessions_left": max(0, config.SESSIONS_PER_MONTH - sessions_used),
        }

    return {
        "session_active": True,
        "has_subscription": has_subscription,
        "plan": user.get("plan", "basic"),
        "started_at": user["current_session_start"],
        "elapsed_minutes": int(elapsed.total_seconds() / 60),
        "remaining_minutes": int(remaining.total_seconds() / 60),
        "sessions_used": sessions_used,
        "sessions_left": sessions_left,
    }


@app.post("/chat")
async def chat(request: ChatRequest, req: Request, user: dict = Depends(verify_active_session)):
    if not config.OLLAMA_API_KEY:
        return build_ai_unavailable_response(
            "Сервис ИИ пока не настроен. Проверьте OLLAMA_API_KEY или подключите локальный Ollama.",
            "unconfigured",
            user
        )

    if request.tts_id and request.tts_id.strip():
        tts_session = await db.fetch_one(
            "SELECT * FROM tts_sessions WHERE id = ?",
            (request.tts_id,)
        )
        if tts_session and not tts_session["completed"]:
            raise HTTPException(
                425, 
                "Дождитесь окончания воспроизведения предыдущего ответа"
            )

    ip = SecurityUtils.get_client_ip(req)
    await RateLimiter.check_and_update(ip)
    model = resolve_ollama_model(user.get("plan"))

    try:
        data = await send_ollama_chat(model, request.messages)
        await db.execute(
            "UPDATE users SET total_questions = total_questions + 1 WHERE id = ?",
            (user["id"],)
        )

        start = parse_datetime(user["current_session_start"])
        elapsed_minutes = int((datetime.utcnow() - start).total_seconds() / 60) if start else 0
        remaining = max(0, config.SESSION_DURATION_MINUTES - elapsed_minutes)

        tts_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO tts_sessions (id, completed) VALUES (?, 0)",
            (tts_id,)
        )

        return {
            "response": data.get("message", {}).get("content", "Не удалось получить ответ"),
            "remaining_minutes": remaining,
            "model_used": model,
            "tts_id": tts_id,
        }
    except HTTPException:
        raise
    except httpx.HTTPError:
        return build_ai_unavailable_response(
            "Не удалось связаться с Ollama Cloud. Проверьте интернет, DNS или доступность ollama.com, либо переключитесь на локальный Ollama.",
            model,
            user
        )
    except Exception as exc:
        raise HTTPException(500, f"Internal chat error: {str(exc)}") from exc


@app.post("/tts")
async def tts_server(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        tts_id = str(uuid.uuid4())
        filename = f"tts_{tts_id}.mp3"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        communicate = edge_tts.Communicate(request.text, request.voice)
        await communicate.save(filepath)
        
        await db.execute(
            "INSERT INTO tts_sessions (id, completed) VALUES (?, 0)",
            (tts_id,)
        )
        
        await db.execute("INSERT INTO temp_files (file_path) VALUES (?)", (filepath,))
        background_tasks.add_task(delete_file_later, filepath, 300)
        
        response = FileResponse(filepath, media_type="audio/mpeg")
        response.headers["X-TTS-ID"] = tts_id
        response.headers["Access-Control-Expose-Headers"] = "X-TTS-ID"
        return response
    except Exception as exc:
        raise HTTPException(500, f"Ошибка синтеза речи: {str(exc)}") from exc


@app.post("/tts/complete/{tts_id}")
async def mark_tts_complete(tts_id: str):
    await db.execute(
        "UPDATE tts_sessions SET completed = 1 WHERE id = ?",
        (tts_id,)
    )
    return {"status": "ok", "tts_id": tts_id}


@app.get("/tts/status/{tts_id}")
async def get_tts_status(tts_id: str):
    tts_session = await db.fetch_one(
        "SELECT * FROM tts_sessions WHERE id = ?",
        (tts_id,)
    )
    if not tts_session:
        return {"tts_id": tts_id, "completed": True, "not_found": True}
    
    return {
        "tts_id": tts_id,
        "completed": bool(tts_session["completed"]),
        "created_at": tts_session["created_at"]
    }


@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "ru-RU-DmitryNeural", "name": "Дмитрий", "gender": "male", "description": "Спокойный мужской голос"},
            {"id": "ru-RU-SvetlanaNeural", "name": "Светлана", "gender": "female", "description": "Приятный женский голос"},
            {"id": "ru-RU-DariyaNeural", "name": "Дарья", "gender": "female", "description": "Молодой женский голос"},
        ]
    }


@app.get("/status")
async def get_status(user: dict = Depends(verify_user)):
    await SessionManager.reset_monthly_sessions(user["id"])
    user_row = await db.fetch_one("SELECT * FROM users WHERE id = ?", (user["id"],))
    user = dict(user_row)
    sessions_used = user.get("sessions_used_this_month", 0)
    return {
        "name": user["name"],
        "plan": user.get("plan", "basic"),
        "model": config.MODEL_PREMIUM if user.get("plan") == "premium" else config.MODEL_BASIC,
        "has_subscription": subscription_is_active(user.get("subscription_end")),
        "subscription_end": user.get("subscription_end"),
        "sessions_used": sessions_used,
        "sessions_left": max(0, config.SESSIONS_PER_MONTH - sessions_used),
        "total_questions": user.get("total_questions", 0),
        "current_session": {
            "active": user.get("current_session_start") is not None,
            "started_at": user.get("current_session_start")
        } if user.get("current_session_start") else None
    }


@app.post("/payment/create")
async def create_payment(request: PaymentRequest):
    user = await db.fetch_one(
        "SELECT * FROM users WHERE user_api_key = ?",
        (request.user_api_key,)
    )
    if not user:
        raise HTTPException(401, "Неверный API ключ")

    amount = config.PRICE_BASIC if request.plan == "basic" else config.PRICE_PREMIUM
    internal_payment_id = f"lenya_{uuid.uuid4().hex[:12]}"
    await db.execute(
        """
        INSERT INTO payments (user_id, amount, plan, payment_id, status)
        VALUES (?, ?, ?, ?, 'pending')
        """,
        (user["id"], amount, request.plan, internal_payment_id)
    )

    if config.yookassa_enabled():
        try:
            auth = base64.b64encode(
                f"{config.YOOKASSA_SHOP_ID}:{config.YOOKASSA_SECRET_KEY}".encode()
            ).decode()
            headers = {
                "Authorization": f"Basic {auth}",
                "Idempotence-Key": str(uuid.uuid4()),
                "Content-Type": "application/json",
            }
            payload = {
                "amount": {"value": str(amount), "currency": "RUB"},
                "confirmation": {
                    "type": "redirect",
                    "return_url": f"{config.BASE_SERVER_URL}/payment/success",
                },
                "capture": True,
                "description": f"Лёня AI Tutor - {request.plan.upper()}",
                "metadata": {"internal_payment_id": internal_payment_id},
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(config.YOOKASSA_API_URL, json=payload, headers=headers)
            if response.status_code != 200:
                raise HTTPException(400, f"Ошибка YooKassa: {response.text}")
            data = response.json()
            confirmation_url = data["confirmation"]["confirmation_url"]
            await db.execute(
                """
                UPDATE payments
                SET yookassa_id = ?, confirmation_url = ?
                WHERE payment_id = ?
                """,
                (data["id"], confirmation_url, internal_payment_id)
            )
            return {
                "payment_id": internal_payment_id,
                "amount": amount,
                "plan": request.plan,
                "confirmation_url": confirmation_url,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Ошибка создания платежа: {str(exc)}") from exc

    if not config.ENABLE_TEST_PAYMENTS:
        raise HTTPException(503, "Тестовые платежи отключены")

    test_url = f"{config.BASE_SERVER_URL}/payment/test?payment_id={internal_payment_id}"
    return {
        "payment_id": internal_payment_id,
        "amount": amount,
        "plan": request.plan,
        "confirmation_url": test_url,
        "test_mode": True,
    }


@app.post("/webhooks/yookassa")
async def yookassa_webhook(request: Request):
    try:
        event_json = await request.json()
        event_type = event_json.get("event")
        payment_info = event_json.get("object", {})
        metadata = payment_info.get("metadata", {})
        internal_payment_id = metadata.get("internal_payment_id")
        if not internal_payment_id:
            return {"status": "error", "message": "No payment id"}
        if event_type == "payment.succeeded":
            await activate_payment(internal_payment_id)
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/payment/success")
async def payment_success():
    return render_payment_result(
        "Оплата прошла успешно",
        "Подписка активирована на 30 дней. Вернитесь в приложение и обновите статус.",
    )


@app.get("/payment/test")
async def test_payment(payment_id: str):
    if not config.ENABLE_TEST_PAYMENTS:
        raise HTTPException(404, "Not found")

    payment, activated = await activate_payment(payment_id)
    if not payment:
        return render_payment_result("Платёж не найден", "Проверьте ссылку и попробуйте снова.", "#f44336")
    if not activated and payment["status"] == "success":
        return render_payment_result("Уже активировано", "Этот тестовый платёж уже был обработан.")

    return render_payment_result(
        "Тестовая оплата успешна",
        f"Тариф {payment['plan'].upper()} активирован. Сумма: {payment['amount']} RUB.",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    
    print("\n" + "=" * 70)
    print("Лёня AI Tutor - Render Edition")
    print("=" * 70)
    print(f"BASIC:   {config.PRICE_BASIC} RUB/month - {config.SESSIONS_PER_MONTH} sessions")
    print(f"PREMIUM: {config.PRICE_PREMIUM} RUB/month - {config.SESSIONS_PER_MONTH} sessions")
    print(f"Ollama API: {'loaded' if config.OLLAMA_API_KEY else 'missing'}")
    print(f"YooKassa: {'live' if config.yookassa_enabled() else 'test'}")
    print(f"Server: {config.BASE_SERVER_URL}")
    print(f"Database: {DB_PATH}")
    print(f"Data dir: {RENDER_DATA_DIR}")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        limit_concurrency=10,
        limit_max_requests=1000,
        timeout_keep_alive=30,
    )
