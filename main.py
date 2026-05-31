import asyncio
import base64
import hashlib
import hmac
import os
import secrets
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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

import psycopg2
import psycopg2.extras

load_dotenv()

# ============================================
# ПОДКЛЮЧЕНИЕ К SUPABASE (PostgreSQL)
# ============================================
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"
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
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
    YOOKASSA_SHOP_ID = os.getenv("YOOKASSA_SHOP_ID", "").strip()
    YOOKASSA_SECRET_KEY = os.getenv("YOOKASSA_SECRET_KEY", "").strip()
    BASE_SERVER_URL = os.getenv("BASE_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
    YOOKASSA_API_URL = "https://api.yookassa.ru/v3/payments"
    OLLAMA_DIRECT_API_URL = os.getenv("OLLAMA_DIRECT_API_URL", "https://ollama.com/api/chat")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    MODEL_BASIC = os.getenv("MODEL_BASIC", "gpt-oss:120b-cloud")
    MODEL_PREMIUM = os.getenv("MODEL_PREMIUM", "deepseek-v3.1:671b-cloud")
    MODEL_GEMINI = os.getenv("MODEL_GEMINI", "gemini-2.0-flash-exp")
    MODEL_LIGHT = os.getenv("MODEL_LIGHT", "meta-llama/llama-3.2-3b-instruct:free")

    PRICE_BASIC = int(os.getenv("PRICE_BASIC", "300"))
    PRICE_PREMIUM = int(os.getenv("PRICE_PREMIUM", "500"))
    PRICE_GEMINI = int(os.getenv("PRICE_GEMINI", "1500"))
    PRICE_LIGHT = int(os.getenv("PRICE_LIGHT", "0"))
    SESSIONS_PER_MONTH = int(os.getenv("SESSIONS_PER_MONTH", "5"))
    SESSION_DURATION_MINUTES = int(os.getenv("SESSION_DURATION_MINUTES", "120"))
    MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", "20"))
    MAX_TTS_TEXT_LENGTH = int(os.getenv("MAX_TTS_TEXT_LENGTH", "100000000"))

    ALLOWED_ORIGINS = env_csv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000"
    )
    TRUST_PROXY_HEADERS = env_flag("TRUST_PROXY_HEADERS", False)
    ENABLE_TEST_PAYMENTS = env_flag("ENABLE_TEST_PAYMENTS", True)

    @classmethod
    def yookassa_enabled(cls) -> bool:
        if not cls.YOOKASSA_SHOP_ID or not cls.YOOKASSA_SECRET_KEY:
            return False
        fake_ids = ("test_shop", "ваш_shop_id", "shop_id", "")
        fake_keys = ("test_secret", "ваш_secret_key", "secret_key", "")
        if cls.YOOKASSA_SHOP_ID.lower() in fake_ids:
            return False
        if cls.YOOKASSA_SECRET_KEY.lower() in fake_keys:
            return False
        return True


config = Config()


class Database:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_connection(self):
        conn = psycopg2.connect(SUPABASE_DB_URL)
        conn.autocommit = False
        return conn

    async def execute(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()
                    return cur

    async def fetch_one(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    return dict(result) if result else None

    async def fetch_all(self, query: str, params: tuple = ()):
        async with self._lock:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return [dict(row) for row in results]

    async def init(self):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        password_hash TEXT,
                        user_api_key TEXT UNIQUE,
                        plan TEXT DEFAULT 'basic',
                        subscription_end TEXT,
                        sessions_used_this_month INTEGER DEFAULT 0,
                        current_session_start TEXT,
                        last_session_month TEXT,
                        total_questions INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                conn.commit()
                
                for col, col_type in [
                    ("password_hash", "TEXT"),
                    ("user_api_key", "TEXT"),
                    ("plan", "TEXT DEFAULT 'basic'"),
                    ("subscription_end", "TEXT"),
                    ("sessions_used_this_month", "INTEGER DEFAULT 0"),
                    ("current_session_start", "TEXT"),
                    ("last_session_month", "TEXT"),
                    ("total_questions", "INTEGER DEFAULT 0"),
                ]:
                    try:
                        cur.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
                        conn.commit()
                    except:
                        conn.rollback()
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        amount INTEGER NOT NULL,
                        plan TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        payment_id TEXT UNIQUE,
                        yookassa_id TEXT,
                        confirmation_url TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        paid_at TEXT
                    )
                """)
                conn.commit()

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        ip TEXT PRIMARY KEY,
                        requests_count INTEGER DEFAULT 1,
                        first_request TEXT,
                        blocked_until TEXT
                    )
                """)
                conn.commit()

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS temp_files (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                conn.commit()

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tts_sessions (
                        id TEXT PRIMARY KEY,
                        completed INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                conn.commit()


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
        if value not in {"basic", "premium", "gemini", "light"}:
            raise ValueError("Неверный тариф. Доступны: basic, premium, gemini, light")
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
        if value not in {"basic", "premium", "gemini", "light"}:
            raise ValueError("Неверный тариф. Доступны: basic, premium, gemini, light")
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
            "SELECT last_session_month FROM users WHERE id = %s",
            (user_id,)
        )
        if user and user.get("last_session_month") != current_month:
            await db.execute(
                "UPDATE users SET sessions_used_this_month = 0, last_session_month = %s WHERE id = %s",
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
            WHERE id = %s AND current_session_start = %s
            """,
            (user_id, started_at)
        )

    @staticmethod
    async def validate_and_prepare_session(user: Dict[str, Any]) -> Dict[str, Any]:
        user_id = user["id"]
        await SessionManager.reset_monthly_sessions(user_id)
        user_row = await db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        user = dict(user_row) if user_row else {}

        if not user:
            raise HTTPException(403, "Пользователь не найден")

        if not subscription_is_active(user.get("subscription_end")):
            if user.get("subscription_end"):
                raise HTTPException(403, "Подписка истекла. Продлите тариф.")
            raise HTTPException(403, "Нет активной подписки. Сначала оплатите тариф.")

        current_session_start = user.get("current_session_start")
        if current_session_start:
            session_start = parse_datetime(current_session_start)
            if session_start and datetime.utcnow() - session_start > timedelta(minutes=config.SESSION_DURATION_MINUTES):
                await SessionManager.finish_expired_session(user_id, current_session_start)
                user_row = await db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
                user = dict(user_row) if user_row else {}

        if user.get("sessions_used_this_month", 0) >= config.SESSIONS_PER_MONTH:
            raise HTTPException(403, f"Лимит исчерпан: {config.SESSIONS_PER_MONTH} занятий в месяц.")

        return user


class RateLimiter:
    @staticmethod
    async def check_and_update(ip: str, limit: int = 30, window: int = 60, block_time: int = 300):
        now = datetime.utcnow()
        rate = await db.fetch_one("SELECT * FROM rate_limits WHERE ip = %s", (ip,))

        if not rate:
            await db.execute(
                "INSERT INTO rate_limits (ip, first_request) VALUES (%s, %s)",
                (ip, now.isoformat())
            )
            return

        blocked_until = parse_datetime(rate.get("blocked_until"))
        if blocked_until and blocked_until > now:
            seconds_left = int((blocked_until - now).total_seconds())
            raise HTTPException(429, f"Слишком много запросов. Попробуйте через {seconds_left} сек.")

        first_request = parse_datetime(rate.get("first_request"))
        if not first_request or (now - first_request).total_seconds() > window:
            await db.execute(
                "UPDATE rate_limits SET requests_count = 1, first_request = %s, blocked_until = NULL WHERE ip = %s",
                (now.isoformat(), ip)
            )
            return

        new_count = rate.get("requests_count", 0) + 1
        if new_count > limit:
            blocked_until = (now + timedelta(seconds=block_time)).isoformat()
            await db.execute(
                "UPDATE rate_limits SET requests_count = %s, blocked_until = %s WHERE ip = %s",
                (new_count, blocked_until, ip)
            )
            raise HTTPException(429, f"IP заблокирован на {block_time} секунд")

        await db.execute(
            "UPDATE rate_limits SET requests_count = %s WHERE ip = %s",
            (new_count, ip)
        )


async def verify_user(x_api_key: str = Header(...)):
    user = await db.fetch_one(
        "SELECT * FROM users WHERE user_api_key = %s",
        (x_api_key,)
    )
    if not user:
        raise HTTPException(401, "Неверный API ключ")
    return user


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
        "SELECT * FROM payments WHERE payment_id = %s",
        (payment_id,)
    )
    if not payment:
        return None, False
    if payment.get("status") == "success":
        return payment, False

    subscription_end = (datetime.utcnow() + timedelta(days=30)).isoformat()
    current_month = datetime.utcnow().strftime("%Y-%m")

    await db.execute(
        """
        UPDATE users
        SET plan = %s,
            subscription_end = %s,
            sessions_used_this_month = 0,
            last_session_month = %s,
            current_session_start = NULL
        WHERE id = %s
        """,
        (payment["plan"], subscription_end, current_month, payment["user_id"])
    )
    await db.execute(
        """
        UPDATE payments
        SET status = 'success', paid_at = %s
        WHERE payment_id = %s
        """,
        (datetime.utcnow().isoformat(), payment_id)
    )
    updated_payment = await db.fetch_one(
        "SELECT * FROM payments WHERE payment_id = %s",
        (payment_id,)
    )
    return updated_payment, True


def resolve_ai_model(plan: str | None) -> tuple[str, str]:
    if plan == "gemini":
        return config.MODEL_GEMINI, "google"
    elif plan == "light":
        return config.MODEL_LIGHT, "openrouter"
    elif plan == "premium":
        return config.MODEL_PREMIUM.removesuffix("-cloud"), "ollama"
    else:
        return config.MODEL_BASIC.removesuffix("-cloud"), "ollama"


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


async def send_gemini_chat(model: str, messages: list) -> dict:
    if not config.GOOGLE_API_KEY:
        raise HTTPException(502, "Google API key not configured")
    
    gemini_messages = []
    system_instruction = None
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_instruction = content
        elif role == "user":
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": content}]
            })
        elif role == "assistant":
            gemini_messages.append({
                "role": "model",
                "parts": [{"text": content}]
            })
    
    url = f"{config.GEMINI_API_URL}/{model}:generateContent?key={config.GOOGLE_API_KEY}"
    
    payload = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
            "topP": 0.95,
            "topK": 40
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    headers = {"Content-Type": "application/json"}
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        detail = response.text[:500]
        if response.status_code == 404 and model == config.MODEL_GEMINI:
            print(f"Gemini model {model} not found, trying gemini-1.5-flash")
            return await send_gemini_chat("gemini-1.5-flash", messages)
        raise HTTPException(502, f"Gemini API error {response.status_code}: {detail}")
    
    data = response.json()
    
    candidates = data.get("candidates", [])
    if not candidates:
        raise HTTPException(502, "No response from Gemini")
    
    content_parts = candidates[0].get("content", {}).get("parts", [])
    response_text = ""
    for part in content_parts:
        if "text" in part:
            response_text += part["text"]
    
    return {
        "message": {
            "content": response_text,
            "role": "assistant"
        },
        "model": model,
        "usage": data.get("usageMetadata", {})
    }


async def send_openrouter_chat(model: str, messages: list) -> dict:
    headers = {
        "HTTP-Referer": config.BASE_SERVER_URL,
        "X-Title": "Lenya AI Tutor",
        "Content-Type": "application/json"
    }
    
    if config.OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
    
    openrouter_messages = []
    for msg in messages:
        openrouter_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    payload = {
        "model": model,
        "messages": openrouter_messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            config.OPENROUTER_API_URL,
            json=payload,
            headers=headers
        )
    
    if response.status_code == 429:
        raise HTTPException(429, "OpenRouter free limit reached. Try again later.")
    
    if response.status_code != 200:
        detail = response.text[:500]
        raise HTTPException(502, f"OpenRouter API error {response.status_code}: {detail}")
    
    data = response.json()
    
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(502, "No response from OpenRouter")
    
    message = choices[0].get("message", {})
    content = message.get("content", "")
    
    return {
        "message": {
            "content": content,
            "role": "assistant"
        },
        "model": data.get("model", model),
        "usage": data.get("usage", {})
    }


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await db.init()
    await cleanup_old_temp_files()
    print("=" * 70)
    print("Лёня AI Tutor - Render + Supabase")
    print(f"URL: {config.BASE_SERVER_URL}")
    print(f"BASIC: {config.PRICE_BASIC} RUB | PREMIUM: {config.PRICE_PREMIUM} RUB")
    print(f"GEMINI: {config.PRICE_GEMINI} RUB | LIGHT: FREE")
    print(f"Сессий: {config.SESSIONS_PER_MONTH} | Длительность: {config.SESSION_DURATION_MINUTES} мин")
    print("=" * 70)
    yield


app = FastAPI(
    title="Лёня AI Tutor",
    description="AI репетитор с голосом, подпиской и лимитами занятий",
    version="8.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-ID"],
)


async def cleanup_old_temp_files():
    try:
        old_files = await db.fetch_all(
            "SELECT file_path FROM temp_files WHERE created_at < NOW() - INTERVAL '1 hour'"
        )
        for file in old_files:
            file_path = file.get("file_path")
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        await db.execute("DELETE FROM temp_files WHERE created_at < NOW() - INTERVAL '1 hour'")
        await db.execute("DELETE FROM tts_sessions WHERE created_at < NOW() - INTERVAL '1 hour'")
    except Exception:
        pass


async def delete_file_later(filepath: str, delay: int):
    await asyncio.sleep(delay)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        await db.execute("DELETE FROM temp_files WHERE file_path = %s", (filepath,))
    except Exception:
        pass


@app.get("/")
async def root():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return {"service": "Лёня AI Tutor", "version": app.version, "status": "OK"}


@app.get("/api")
async def api_info():
    return {
        "service": "Лёня AI Tutor",
        "version": app.version,
        "test_payments_enabled": config.ENABLE_TEST_PAYMENTS,
        "payments_mode": "live" if config.yookassa_enabled() else "test",
        "hosting": "Render + Supabase",
    }


@app.get("/health")
async def health():
    try:
        await db.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "Supabase (PostgreSQL)",
            "timestamp": datetime.utcnow().isoformat(),
            "ollama_configured": bool(config.OLLAMA_API_KEY),
            "gemini_configured": bool(config.GOOGLE_API_KEY),
            "openrouter_configured": True,
        }
    except Exception as e:
        return {"status": "degraded", "database": str(e)}


@app.post("/register")
async def register(request: RegisterRequest):
    existing = await db.fetch_one("SELECT * FROM users WHERE name = %s", (request.name,))
    if existing:
        if existing.get("password_hash"):
            if not SecurityUtils.verify_password(request.password, existing["password_hash"]):
                raise HTTPException(status_code=401, detail="Неверные имя или пароль")
        else:
            await db.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (SecurityUtils.hash_password(request.password), existing["id"])
            )
        return {
            "name": request.name,
            "user_api_key": existing["user_api_key"],
            "plan": existing.get("plan") or "basic",
            "has_subscription": subscription_is_active(existing.get("subscription_end")),
            "message": "Вход выполнен успешно",
        }

    user_api_key = SecurityUtils.generate_api_key()
    password_hash = SecurityUtils.hash_password(request.password)
    current_month = datetime.utcnow().strftime("%Y-%m")
    await db.execute(
        "INSERT INTO users (name, password_hash, user_api_key, plan, last_session_month) VALUES (%s, %s, %s, %s, %s)",
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
    user = await db.fetch_one("SELECT * FROM users WHERE name = %s", (request.name,))
    if not user:
        raise HTTPException(status_code=404, detail="Аккаунт не найден")
    if user.get("password_hash"):
        if not SecurityUtils.verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Неверные имя или пароль")
    return {
        "name": user["name"],
        "user_api_key": user["user_api_key"],
        "plan": user.get("plan") or "basic",
        "has_subscription": subscription_is_active(user.get("subscription_end")),
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
    await db.execute("UPDATE users SET current_session_start = %s WHERE id = %s", (now, user["id"]))
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
    user_row = await db.fetch_one("SELECT * FROM users WHERE id = %s", (user["id"],))
    user = dict(user_row) if user_row else {}

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
    model, provider = resolve_ai_model(user.get("plan"))
    
    if provider == "google" and not config.GOOGLE_API_KEY:
        return build_ai_unavailable_response("Google Gemini не настроен.", model, user)
    elif provider == "openrouter" and not config.OPENROUTER_API_KEY:
        # OpenRouter может работать без ключа для бесплатных моделей
        pass
    elif provider == "ollama" and not config.OLLAMA_API_KEY:
        return build_ai_unavailable_response("Ollama Cloud не настроен.", "unconfigured", user)

    if request.tts_id and request.tts_id.strip():
        tts_session = await db.fetch_one("SELECT * FROM tts_sessions WHERE id = %s", (request.tts_id,))
        if tts_session and not tts_session.get("completed"):
            raise HTTPException(425, "Дождитесь окончания воспроизведения предыдущего ответа")

    ip = SecurityUtils.get_client_ip(req)
    await RateLimiter.check_and_update(ip)

    try:
        if provider == "google":
            data = await send_gemini_chat(model, request.messages)
        elif provider == "openrouter":
            data = await send_openrouter_chat(model, request.messages)
        else:
            data = await send_ollama_chat(model, request.messages)
        
        await db.execute(
            "UPDATE users SET total_questions = total_questions + 1 WHERE id = %s",
            (user["id"],)
        )

        start = parse_datetime(user["current_session_start"])
        elapsed_minutes = int((datetime.utcnow() - start).total_seconds() / 60) if start else 0
        remaining = max(0, config.SESSION_DURATION_MINUTES - elapsed_minutes)

        tts_id = str(uuid.uuid4())
        await db.execute("INSERT INTO tts_sessions (id, completed) VALUES (%s, 0)", (tts_id,))

        return {
            "response": data.get("message", {}).get("content", "Не удалось получить ответ"),
            "remaining_minutes": remaining,
            "model_used": data.get("model", model),
            "tts_id": tts_id,
        }
    except HTTPException:
        raise
    except httpx.HTTPError:
        provider_names = {"google": "Google Gemini", "openrouter": "OpenRouter", "ollama": "Ollama Cloud"}
        return build_ai_unavailable_response(f"Не удалось связаться с {provider_names.get(provider, 'AI')}.", model, user)
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

        await db.execute("INSERT INTO tts_sessions (id, completed) VALUES (%s, 0)", (tts_id,))
        await db.execute("INSERT INTO temp_files (file_path) VALUES (%s)", (filepath,))
        background_tasks.add_task(delete_file_later, filepath, 300)

        response = FileResponse(filepath, media_type="audio/mpeg")
        response.headers["X-TTS-ID"] = tts_id
        response.headers["Access-Control-Expose-Headers"] = "X-TTS-ID"
        return response
    except Exception as exc:
        raise HTTPException(500, f"Ошибка синтеза речи: {str(exc)}") from exc


@app.post("/tts/complete/{tts_id}")
async def mark_tts_complete(tts_id: str):
    await db.execute("UPDATE tts_sessions SET completed = 1 WHERE id = %s", (tts_id,))
    return {"status": "ok", "tts_id": tts_id}


@app.get("/tts/status/{tts_id}")
async def get_tts_status(tts_id: str):
    tts_session = await db.fetch_one("SELECT * FROM tts_sessions WHERE id = %s", (tts_id,))
    if not tts_session:
        return {"tts_id": tts_id, "completed": True, "not_found": True}
    return {
        "tts_id": tts_id,
        "completed": bool(tts_session.get("completed")),
        "created_at": str(tts_session.get("created_at"))
    }


@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "ru-RU-DmitryNeural", "name": "Дмитрий", "gender": "male"},
            {"id": "ru-RU-SvetlanaNeural", "name": "Светлана", "gender": "female"},
            {"id": "ru-RU-DariyaNeural", "name": "Дарья", "gender": "female"},
        ]
    }


@app.get("/status")
async def get_status(user: dict = Depends(verify_user)):
    await SessionManager.reset_monthly_sessions(user["id"])
    user_row = await db.fetch_one("SELECT * FROM users WHERE id = %s", (user["id"],))
    user = dict(user_row) if user_row else {}
    sessions_used = user.get("sessions_used_this_month", 0)
    
    plan = user.get("plan", "basic")
    model_info = {
        "basic": (config.MODEL_BASIC, "Ollama Cloud"),
        "premium": (config.MODEL_PREMIUM, "Ollama Cloud"),
        "gemini": (config.MODEL_GEMINI, "Google AI"),
        "light": (config.MODEL_LIGHT, "OpenRouter (Free)")
    }
    model_name, provider = model_info.get(plan, (config.MODEL_BASIC, "Ollama Cloud"))
    
    return {
        "name": user.get("name"),
        "plan": plan,
        "model": model_name,
        "provider": provider,
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
    user = await db.fetch_one("SELECT * FROM users WHERE user_api_key = %s", (request.user_api_key,))
    if not user:
        raise HTTPException(401, "Неверный API ключ")

    prices = {
        "basic": config.PRICE_BASIC,
        "premium": config.PRICE_PREMIUM,
        "gemini": config.PRICE_GEMINI,
        "light": config.PRICE_LIGHT
    }
    amount = prices.get(request.plan, config.PRICE_BASIC)
    
    plan_names = {
        "basic": "Basic (GPT OSS 120B)",
        "premium": "Premium (DeepSeek V3.1)",
        "gemini": "Gemini Flash (Google AI)",
        "light": "Light (Free)"
    }
    
    # Бесплатный тариф - сразу активируем
    if amount == 0:
        internal_payment_id = f"lenya_free_{uuid.uuid4().hex[:12]}"
        await db.execute(
            "INSERT INTO payments (user_id, amount, plan, payment_id, status) VALUES (%s, %s, %s, %s, 'success')",
            (user["id"], 0, request.plan, internal_payment_id)
        )
        
        subscription_end = (datetime.utcnow() + timedelta(days=30)).isoformat()
        current_month = datetime.utcnow().strftime("%Y-%m")
        
        await db.execute(
            """
            UPDATE users
            SET plan = %s, subscription_end = %s, sessions_used_this_month = 0,
                last_session_month = %s, current_session_start = NULL
            WHERE id = %s
            """,
            (request.plan, subscription_end, current_month, user["id"])
        )
        
        return {
            "payment_id": internal_payment_id,
            "amount": 0,
            "plan": request.plan,
            "plan_name": "Light (Free)",
            "message": "Бесплатный тариф активирован!",
            "free": True
        }
    
    internal_payment_id = f"lenya_{uuid.uuid4().hex[:12]}"
    await db.execute(
        "INSERT INTO payments (user_id, amount, plan, payment_id, status) VALUES (%s, %s, %s, %s, 'pending')",
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
                "description": f"Лёня AI Tutor - {plan_names.get(request.plan, request.plan)}",
                "metadata": {"internal_payment_id": internal_payment_id},
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(config.YOOKASSA_API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                confirmation_url = data["confirmation"]["confirmation_url"]
                await db.execute(
                    "UPDATE payments SET yookassa_id = %s, confirmation_url = %s WHERE payment_id = %s",
                    (data["id"], confirmation_url, internal_payment_id)
                )
                return {
                    "payment_id": internal_payment_id,
                    "amount": amount,
                    "plan": request.plan,
                    "plan_name": plan_names.get(request.plan, request.plan),
                    "confirmation_url": confirmation_url,
                }
        except Exception:
            pass

    test_url = f"{config.BASE_SERVER_URL}/payment/test?payment_id={internal_payment_id}"
    return {
        "payment_id": internal_payment_id,
        "amount": amount,
        "plan": request.plan,
        "plan_name": plan_names.get(request.plan, request.plan),
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
    return render_payment_result("Оплата прошла успешно", "Подписка активирована на 30 дней.")


@app.get("/payment/test")
async def test_payment(payment_id: str):
    payment, activated = await activate_payment(payment_id)
    if not payment:
        return render_payment_result("Платёж не найден", "Проверьте ссылку.", "#f44336")
    if not activated and payment.get("status") == "success":
        return render_payment_result("Уже активировано", "Платёж уже обработан.")
    return render_payment_result(
        "Тестовая оплата успешна",
        f"Тариф {payment.get('plan', '').upper()} активирован. Сумма: {payment.get('amount')} RUB.",
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print("\n" + "=" * 70)
    print("Лёня AI Tutor - Render + Supabase")
    print(f"URL: {config.BASE_SERVER_URL}")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
