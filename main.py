import os
import hashlib
import httpx
import secrets
import string
import re
import base64
import binascii
import time
import pyotp
import uuid
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, unquote, urlparse, parse_qs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import auth_logic
import dashboard_manager
import google.generativeai as genai
import json

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

cors_allowed_origins = [
    origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if origin.strip()
]

supabase: Client = create_client(supabase_url, supabase_key)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- MODELS ---
class AICrackEstimateReq(BaseModel):
    entropyBits: float
    expectedGuesses: float

class UserAuth(BaseModel):
    email: EmailStr
    password: str

class VerifyOTP(BaseModel):
    email: str
    code: str

class MFAChallengeReq(BaseModel):
    user_id: str

class MFAVerifyReq(BaseModel):
    user_id: str
    code: str
    is_reauth: bool = False


class RecoveryVerifyReq(BaseModel):
    user_id: str
    recovery_key: str


class RecoveryResetReq(BaseModel):
    email: EmailStr
    recovery_key: str
    new_password: str

class NoteReq(BaseModel):
    user_id: str
    content: str


class NoteUpdateReq(BaseModel):
    user_id: str
    content: str

class PassReq(BaseModel):
    user_id: str
    site_name: str
    username_or_email: str
    password: str

class BookmarkReq(BaseModel):
    user_id: str
    title: str
    url: str

class AuthSeedReq(BaseModel):
    user_id: str
    issuer: str
    secret_key: str
    account_name: str | None = None


class UserScopeReq(BaseModel):
    user_id: str


class OneTimeSecretCreateReq(BaseModel):
    user_id: str
    content: str
    expires_in_minutes: int = 30


class ChatCreateReq(BaseModel):
    user_id: str
    invite_policy: str = "link"
    passcode: str | None = None
    approved_user_ids: list[str] | None = None


class ChatJoinReq(BaseModel):
    user_id: str | None = None
    passcode: str | None = None


class PasswordGeneratorReq(BaseModel):
    base_string: str
    length: int = 16
    include_uppercase: bool = True
    include_numbers: bool = True
    include_special: bool = True
    exclude_ambiguous: bool = True
    count: int = 5


class QrDecodeReq(BaseModel):
    image_data: str


class ChatSendReq(BaseModel):
    user_id: str
    session_id: str
    content: str


class ChatDeleteReq(BaseModel):
    user_id: str
    session_id: str
    deletion_reason: str | None = None


class ChatLeaveReq(BaseModel):
    user_id: str | None = None
    session_id: str


class ChatUpdateSettingsReq(BaseModel):
    user_id: str
    session_id: str
    invite_policy: str | None = None
    passcode: str | None = None
    clear_passcode: bool = False
    approved_user_ids: list[str] | None = None


class ChatApprovalReq(BaseModel):
    user_id: str
    session_id: str
    target_user_id: str
    action: str = "add"


class ChatTypingReq(BaseModel):
    user_id: str
    session_id: str
    is_typing: bool = True


def _is_valid_totp_secret(secret: str | None) -> bool:
    if not secret:
        return False
    try:
        normalized = secret.strip().replace(" ", "")
        if not normalized:
            return False
        padding = "=" * ((8 - (len(normalized) % 8)) % 8)
        base64.b32decode((normalized + padding).upper(), casefold=True)
        return True
    except (binascii.Error, ValueError):
        return False


ENC_PREFIX = "enc::"


def _build_data_encryption_key() -> bytes:
    configured = os.getenv("DATA_ENCRYPTION_KEY")
    if configured:
        token = configured.strip().encode("utf-8")
        try:
            Fernet(token)
            return token
        except Exception:
            pass

    seed = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_URL") or "ciphergate-default-seed"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


FERNET = Fernet(_build_data_encryption_key())


def _encrypt_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    if text.startswith(ENC_PREFIX):
        return text
    token = FERNET.encrypt(text.encode("utf-8")).decode("utf-8")
    return f"{ENC_PREFIX}{token}"


def _decrypt_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text or not text.startswith(ENC_PREFIX):
        return text
    token = text[len(ENC_PREFIX):]
    try:
        return FERNET.decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        return text


def _recovery_secret() -> str:
    return (
        os.getenv("RECOVERY_KEY_SECRET")
        or os.getenv("SUPABASE_ANON_KEY")
        or "ciphergate-recovery-secret"
    )


def _derive_user_recovery_key(user_id: str, email: str) -> str:
    seed = f"{user_id}:{email}:{_recovery_secret()}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    token = base64.b32encode(digest).decode("utf-8").replace("=", "")[:24]
    return "-".join(token[i:i + 4] for i in range(0, len(token), 4))


def _parse_otpauth_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "otpauth" or parsed.netloc.lower() != "totp":
        raise HTTPException(status_code=400, detail="QR does not contain a valid TOTP otpauth URI")

    label = unquote((parsed.path or "").lstrip("/"))
    issuer_from_label = None
    account_name = label
    if ":" in label:
        issuer_from_label, account_name = label.split(":", 1)

    query = parse_qs(parsed.query)
    secret = (query.get("secret", [""])[0] or "").strip()
    issuer = (query.get("issuer", [""])[0] or "").strip() or issuer_from_label or "CipherGate"

    if not _is_valid_totp_secret(secret):
        raise HTTPException(status_code=400, detail="QR code found, but secret key is missing/invalid")

    return {
        "issuer": issuer,
        "account_name": (account_name or "").strip(),
        "secret_key": secret,
        "otpauth_uri": uri,
    }


def _build_totp_payload(secret_key: str):
    normalized = (secret_key or "").strip().replace(" ", "")
    totp = pyotp.TOTP(normalized)
    period = int(getattr(totp, "interval", 30) or 30)
    current_ts = int(time.time())
    remaining = period - (current_ts % period)
    if remaining <= 0:
        remaining = period

    return {
        "code": totp.now(),
        "period": period,
        "seconds_remaining": remaining,
    }


CHAT_ROOMS: dict[str, dict] = {}
CHAT_STALE_SECONDS = 45
CHAT_SYSTEM_AUTHOR = "__system__"
SYSTEM_EVENT_PREFIX = "SYS_EVENT::"
CHAT_DEFAULT_INVITE_POLICY = "link"
CHAT_ROOM_SETTINGS: dict[str, dict] = {}
CHAT_DELETED_ROOMS: dict[str, dict] = {}
CHAT_AUDIT_LOGS: list[dict] = []
CHAT_ROOM_APPROVAL_REQUESTS: dict[str, list[dict]] = {}
CHAT_ROOM_TYPING: dict[str, dict[str, dict]] = {}
ONE_TIME_SECRETS: dict[str, dict] = {}
ONE_TIME_SECRET_MIN_MINUTES = 1
ONE_TIME_SECRET_MAX_MINUTES = 10080


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_one_time_secrets() -> None:
    now_ts = time.time()
    expired_tokens = [
        token
        for token, payload in ONE_TIME_SECRETS.items()
        if float(payload.get("expires_ts", 0)) <= now_ts or bool(payload.get("consumed"))
    ]
    for token in expired_tokens:
        ONE_TIME_SECRETS.pop(token, None)


def _random_display_name() -> str:
    adjectives = [
        "Silent", "Neon", "Swift", "Bright", "Misty", "Calm", "Shadow", "Crimson", "Solar", "Icy",
    ]
    nouns = [
        "Falcon", "Comet", "River", "Cipher", "Nova", "Fox", "Panda", "Eagle", "Wolf", "Otter",
    ]
    return f"{secrets.choice(adjectives)}{secrets.choice(nouns)}{secrets.randbelow(90) + 10}"


def _hash_passcode(passcode: str) -> str:
    normalized = str(passcode or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _verify_passcode(passcode_hash: str | None, provided: str | None) -> bool:
    if not passcode_hash:
        return True
    if not provided:
        return False
    return _hash_passcode(provided) == passcode_hash


def _normalize_invite_policy(policy: str | None) -> str:
    normalized = (policy or CHAT_DEFAULT_INVITE_POLICY).strip().lower()
    if normalized not in {"link", "approval"}:
        return CHAT_DEFAULT_INVITE_POLICY
    return normalized


def _default_room_settings() -> dict:
    return {
        "invite_policy": CHAT_DEFAULT_INVITE_POLICY,
        "passcode_hash": None,
        "approved_user_ids": [],
    }


def _make_system_event(kind: str, payload: dict) -> str:
    return f"{SYSTEM_EVENT_PREFIX}{json.dumps({'kind': kind, 'payload': payload}, separators=(',', ':'))}"


def _parse_system_event(content: str | None) -> dict | None:
    text = str(content or "")
    if not text.startswith(SYSTEM_EVENT_PREFIX):
        return None
    raw = text[len(SYSTEM_EVENT_PREFIX):]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _sanitize_room_settings(raw: dict | None) -> dict:
    base = _default_room_settings()
    if not isinstance(raw, dict):
        return base
    base["invite_policy"] = _normalize_invite_policy(raw.get("invite_policy"))
    passcode_hash = raw.get("passcode_hash")
    base["passcode_hash"] = passcode_hash if isinstance(passcode_hash, str) and passcode_hash else None
    approved = raw.get("approved_user_ids") or []
    if isinstance(approved, list):
        base["approved_user_ids"] = [str(x).strip() for x in approved if str(x).strip()]
    return base


def _sanitize_approval_requests(raw: list | None) -> list[dict]:
    if not isinstance(raw, list):
        return []
    clean = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        user_id = str(item.get("user_id") or "").strip()
        email = str(item.get("email") or "").strip()
        requested_at = str(item.get("requested_at") or "").strip() or _now_iso()
        if not user_id and not email:
            continue
        clean.append({
            "user_id": user_id or None,
            "email": email or None,
            "requested_at": requested_at,
        })
    return clean


def _lookup_user_email(user_id: str | None) -> str | None:
    uid = str(user_id or "").strip()
    if not uid:
        return None
    try:
        res = supabase.table("users").select("email").eq("id", uid).limit(1).execute().data or []
        if res:
            email = str(res[0].get("email") or "").strip()
            return email or None
    except Exception:
        pass
    return None


def _register_approval_request(room_id: str, user_id: str | None) -> list[dict]:
    current = _sanitize_approval_requests(CHAT_ROOM_APPROVAL_REQUESTS.get(room_id) or [])
    uid = str(user_id or "").strip() or None
    email = _lookup_user_email(uid)

    exists = False
    for req in current:
        if uid and req.get("user_id") == uid:
            exists = True
            break
        if email and req.get("email") == email:
            exists = True
            break

    if not exists:
        entry = {
            "user_id": uid,
            "email": email,
            "requested_at": _now_iso(),
        }
        current.append(entry)
        _persist_system_event(room_id, "approval_request", entry)

    CHAT_ROOM_APPROVAL_REQUESTS[room_id] = current
    return current


def _resolve_approval_request(room_id: str, target_user_id: str | None):
    uid = str(target_user_id or "").strip() or None
    if not uid:
        return
    current = _sanitize_approval_requests(CHAT_ROOM_APPROVAL_REQUESTS.get(room_id) or [])
    next_requests = [req for req in current if req.get("user_id") != uid]
    CHAT_ROOM_APPROVAL_REQUESTS[room_id] = next_requests
    _persist_system_event(room_id, "approval_request_resolved", {"user_id": uid, "resolved_at": _now_iso()})


def _persist_system_event(room_id: str, kind: str, payload: dict):
    try:
        supabase.table("chat_messages").insert({
            "id": uuid.uuid4().hex,
            "room_id": room_id,
            "session_id": "system",
            "author": CHAT_SYSTEM_AUTHOR,
            "content": _make_system_event(kind, payload),
            "created_at": _now_iso(),
        }).execute()
    except Exception:
        pass


def _audit_event(event_type: str, user_id: str | None = None, room_id: str | None = None, status: str = "success", meta: dict | None = None):
    entry = {
        "id": uuid.uuid4().hex,
        "event_type": event_type,
        "user_id": user_id,
        "room_id": room_id,
        "status": status,
        "meta": meta or {},
        "created_at": _now_iso(),
    }
    CHAT_AUDIT_LOGS.append(entry)
    if len(CHAT_AUDIT_LOGS) > 2000:
        del CHAT_AUDIT_LOGS[:500]

    # Optional persistence if a compatible table exists in Supabase.
    try:
        supabase.table("security_audit_logs").insert({
            "id": entry["id"],
            "event_type": entry["event_type"],
            "user_id": entry["user_id"],
            "room_id": entry["room_id"],
            "status": entry["status"],
            "meta": json.dumps(entry["meta"]),
            "created_at": entry["created_at"],
        }).execute()
    except Exception:
        pass


def _build_tombstone(room_id: str) -> dict | None:
    if room_id in CHAT_DELETED_ROOMS:
        return CHAT_DELETED_ROOMS[room_id]

    try:
        rows = supabase.table("chat_messages").select("content, created_at").eq("room_id", room_id).eq("author", CHAT_SYSTEM_AUTHOR).order("created_at", desc=True).limit(25).execute().data or []
    except Exception:
        rows = []

    for row in rows:
        evt = _parse_system_event(row.get("content"))
        if evt and evt.get("kind") == "room_deleted":
            payload = evt.get("payload") or {}
            tombstone = {
                "room_id": room_id,
                "deleted_at": payload.get("deleted_at") or row.get("created_at") or _now_iso(),
                "deleted_by": payload.get("deleted_by"),
                "reason": payload.get("reason") or "Deleted by the room owner",
            }
            CHAT_DELETED_ROOMS[room_id] = tombstone
            return tombstone
    return None


def _room_deleted_http(room_id: str) -> HTTPException:
    tombstone = _build_tombstone(room_id) or {
        "room_id": room_id,
        "deleted_at": _now_iso(),
        "deleted_by": None,
        "reason": "Deleted by the room owner",
    }
    return HTTPException(
        status_code=410,
        detail={
            "code": "ROOM_DELETED",
            "message": "This room has been deleted",
            **tombstone,
        },
    )


def _parse_room_stream(messages_data: list[dict]) -> tuple[dict, dict | None, list[dict], list[dict]]:
    settings = _default_room_settings()
    tombstone = None
    visible_messages = []
    approval_requests: list[dict] = []

    for m in messages_data:
        if m.get("author") == CHAT_SYSTEM_AUTHOR:
            evt = _parse_system_event(m.get("content"))
            if not evt:
                continue
            kind = evt.get("kind")
            payload = evt.get("payload") or {}
            if kind == "room_settings":
                settings = _sanitize_room_settings(payload)
            elif kind == "room_deleted":
                tombstone = {
                    "room_id": payload.get("room_id") or m.get("room_id"),
                    "deleted_at": payload.get("deleted_at") or m.get("created_at") or _now_iso(),
                    "deleted_by": payload.get("deleted_by"),
                    "reason": payload.get("reason") or "Deleted by the room owner",
                }
            elif kind == "approval_request":
                req = {
                    "user_id": payload.get("user_id"),
                    "email": payload.get("email"),
                    "requested_at": payload.get("requested_at") or m.get("created_at") or _now_iso(),
                }
                normalized = _sanitize_approval_requests([req])
                if normalized:
                    candidate = normalized[0]
                    if not any((candidate.get("user_id") and x.get("user_id") == candidate.get("user_id")) or (candidate.get("email") and x.get("email") == candidate.get("email")) for x in approval_requests):
                        approval_requests.append(candidate)
            elif kind == "approval_request_resolved":
                uid = str(payload.get("user_id") or "").strip()
                if uid:
                    approval_requests = [x for x in approval_requests if x.get("user_id") != uid]
            continue

        visible_messages.append({
            "id": m.get("id"),
            "session_id": m.get("session_id"),
            "author": m.get("author", "Anonymous"),
            "content": m.get("content"),
            "created_at": m.get("created_at"),
        })

    return settings, tombstone, visible_messages, approval_requests


def _get_room_or_404(room_id: str) -> dict:
    if _build_tombstone(room_id):
        raise _room_deleted_http(room_id)

    # Check in-memory cache first
    if room_id in CHAT_ROOMS:
        if room_id in CHAT_DELETED_ROOMS:
            raise _room_deleted_http(room_id)
        return CHAT_ROOMS[room_id]
    
    # Try to load from database if not in memory
    try:
        db_room = supabase.table("chat_rooms").select("*").eq("id", room_id).execute().data
        if not db_room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        room_data = db_room[0]
        
        # Load participants from database
        participants_data = supabase.table("chat_participants").select("*").eq("room_id", room_id).execute().data or []
        
        # Load only system events first (settings/tombstone/approvals). User messages are lazy-loaded.
        system_messages = (
            supabase
            .table("chat_messages")
            .select("id, room_id, session_id, author, content, created_at")
            .eq("room_id", room_id)
            .eq("author", CHAT_SYSTEM_AUTHOR)
            .order("created_at", desc=False)
            .execute()
            .data
            or []
        )

        settings, tombstone, _, approval_requests = _parse_room_stream(system_messages)
        if tombstone:
            CHAT_DELETED_ROOMS[room_id] = tombstone
            raise _room_deleted_http(room_id)
        
        # Initialize room in memory with data from database
        now_ts = time.time()
        participants = {}
        for p in participants_data:
            participants[p["session_id"]] = {
                "user_id": p.get("user_id"),
                "display_name": p.get("display_name", "Anonymous"),
                "joined_at": p.get("joined_at"),
                "last_seen": p.get("joined_at"),  # Default to joined_at for old sessions
                "last_seen_ts": now_ts,  # Treat as recently active since we're loading it
            }
        
        CHAT_ROOM_SETTINGS[room_id] = settings
        CHAT_ROOM_APPROVAL_REQUESTS[room_id] = approval_requests
        
        CHAT_ROOMS[room_id] = {
            "id": room_data.get("id"),
            "created_at": room_data.get("created_at"),
            "creator_session_id": None,  # Can't determine from DB, will be set on join
            "creator_user_id": room_data.get("creator_user_id"),
            "participants": participants,
            "messages": [],
            "messages_loaded": False,
            "settings": settings,
            "approval_requests": approval_requests,
        }
        
        return CHAT_ROOMS[room_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail="Chat room not found")


def _prune_stale_participants(room: dict):
    now = time.time()
    stale_ids = []
    for sid, member in room["participants"].items():
        if now - float(member.get("last_seen_ts", 0)) > CHAT_STALE_SECONDS:
            stale_ids.append(sid)
    for sid in stale_ids:
        room["participants"].pop(sid, None)


def _active_count(room: dict) -> int:
    _prune_stale_participants(room)
    return len(room["participants"])


def _active_typing_users(room_id: str, exclude_session_id: str | None = None) -> list[dict]:
    now_ts = time.time()
    room_typing = CHAT_ROOM_TYPING.get(room_id) or {}
    active = []
    stale_sessions = []

    for session_id, entry in room_typing.items():
        if exclude_session_id and session_id == exclude_session_id:
            continue
        if now_ts - float(entry.get("last_seen_ts", 0)) > 5:
            stale_sessions.append(session_id)
            continue
        active.append({
            "session_id": session_id,
            "user_id": entry.get("user_id"),
            "display_name": entry.get("display_name", "Anonymous"),
        })

    for session_id in stale_sessions:
        room_typing.pop(session_id, None)
    if room_typing:
        CHAT_ROOM_TYPING[room_id] = room_typing
    elif room_id in CHAT_ROOM_TYPING:
        CHAT_ROOM_TYPING.pop(room_id, None)

    return active

# --- AUTH ROUTES ---
@app.post("/auth/register")
async def register(user: UserAuth):
    hashed = auth_logic.hash_password(user.password)
    try:
        supabase.table("users").insert({
            "email": user.email, 
            "hashed_password": hashed,
            "otp_secret": None,
            "is_mfa_enabled": False,
        }).execute()
        return {"status": "ok"}
    except: raise HTTPException(status_code=400, detail="Email already registered")

@app.post("/auth/login")
async def login(user: UserAuth):
    res = supabase.table("users").select("*").eq("email", user.email).execute()
    if not res.data: raise HTTPException(status_code=404, detail="User not found")
    if auth_logic.verify_password(user.password, res.data[0]['hashed_password']):
        return {"user_id": res.data[0]['id'], "mfa_enabled": res.data[0]['is_mfa_enabled']}
    raise HTTPException(status_code=401, detail="Wrong password")

@app.post("/auth/2fa-verify")
async def verify_2fa(data: VerifyOTP):
    res = supabase.table("users").select("otp_secret").eq("email", data.email).execute()
    if not res.data or not res.data[0]['otp_secret']: raise HTTPException(status_code=400)
    if auth_logic.verify_totp_code(res.data[0]['otp_secret'], data.code):
        supabase.table("users").update({"is_mfa_enabled": True}).eq("email", data.email).execute()
        return {"status": "verified"}
    raise HTTPException(status_code=401)


@app.post("/auth/mfa/challenge")
async def mfa_challenge(req: MFAChallengeReq):
    res = supabase.table("users").select("id, email, otp_secret, is_mfa_enabled").eq("id", req.user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = res.data[0]
    existing_secret = user.get("otp_secret")
    secret = existing_secret if _is_valid_totp_secret(existing_secret) else None
    needs_setup = not bool(user.get("is_mfa_enabled")) or not secret

    if needs_setup and not secret:
        secret = auth_logic.generate_totp_secret()
        supabase.table("users").update({"otp_secret": secret}).eq("id", req.user_id).execute()

    if needs_setup:
        otpauth_uri = auth_logic.get_totp_uri(user["email"], secret)
        qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={quote(otpauth_uri, safe='')}"
        return {
            "needs_setup": True,
            "issuer": "CipherGate",
            "account_name": user["email"],
            "secret_key": secret,
            "otpauth_uri": otpauth_uri,
            "qr_code_url": qr_code_url,
        }

    return {"needs_setup": False}


@app.post("/auth/mfa/verify")
async def mfa_verify(req: MFAVerifyReq):
    res = supabase.table("users").select("id, otp_secret").eq("id", req.user_id).execute()
    if not res.data:
        _audit_event("mfa_verify", user_id=req.user_id, status="failed", meta={"reason": "user_not_found", "reauth": bool(req.is_reauth)})
        raise HTTPException(status_code=404, detail="User not found")

    secret = res.data[0].get("otp_secret")
    if not _is_valid_totp_secret(secret):
        _audit_event("mfa_verify", user_id=req.user_id, status="failed", meta={"reason": "mfa_not_initialized", "reauth": bool(req.is_reauth)})
        raise HTTPException(status_code=400, detail="MFA is not initialized for this account")

    if not auth_logic.verify_totp_code(secret, req.code):
        _audit_event("mfa_verify", user_id=req.user_id, status="failed", meta={"reason": "invalid_code", "reauth": bool(req.is_reauth)})
        raise HTTPException(status_code=401, detail="Invalid authenticator code")

    supabase.table("users").update({"is_mfa_enabled": True}).eq("id", req.user_id).execute()
    _audit_event("mfa_reauth" if req.is_reauth else "mfa_verify", user_id=req.user_id, status="success", meta={"reauth": bool(req.is_reauth)})
    return {"status": "verified"}


@app.get("/auth/recovery-key")
async def get_recovery_key(user_id: str):
    res = supabase.table("users").select("id, email").eq("id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = res.data[0]
    recovery_key = _derive_user_recovery_key(user_id=user["id"], email=user["email"])
    return {
        "user_id": user["id"],
        "email": user["email"],
        "recovery_key": recovery_key,
    }


@app.post("/auth/recovery/verify")
async def verify_recovery_key(req: RecoveryVerifyReq):
    res = supabase.table("users").select("id, email").eq("id", req.user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = res.data[0]
    expected = _derive_user_recovery_key(user_id=user["id"], email=user["email"])
    provided = (req.recovery_key or "").strip().upper().replace(" ", "")
    expected_normalized = expected.replace("-", "")
    provided_normalized = provided.replace("-", "")

    if provided_normalized != expected_normalized:
        raise HTTPException(status_code=401, detail="Invalid recovery key")

    return {"status": "verified"}


@app.post("/auth/recovery/reset")
async def reset_password_with_recovery(req: RecoveryResetReq):
    res = supabase.table("users").select("id, email").eq("email", req.email).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = res.data[0]
    expected = _derive_user_recovery_key(user_id=user["id"], email=user["email"])
    provided = (req.recovery_key or "").strip().upper().replace(" ", "")
    expected_normalized = expected.replace("-", "")
    provided_normalized = provided.replace("-", "")
    if provided_normalized != expected_normalized:
        raise HTTPException(status_code=401, detail="Invalid recovery key")

    if len((req.new_password or "").strip()) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    hashed = auth_logic.hash_password(req.new_password)
    supabase.table("users").update({"hashed_password": hashed}).eq("id", user["id"]).execute()
    return {"status": "password_reset"}

# --- VAULT DATA (Cleaned of File logic) ---
@app.get("/vault/all")
async def get_all_vault(user_id: str):
    notes = supabase.table("sticky_notes").select("*").eq("user_id", user_id).execute().data
    passwords = supabase.table("saved_passwords").select("*").eq("user_id", user_id).execute().data
    bookmarks = supabase.table("bookmarks").select("*").eq("user_id", user_id).execute().data
    auths = supabase.table("authenticator_seeds").select("*").eq("user_id", user_id).execute().data

    for note in notes:
        note["note_content"] = _decrypt_text(note.get("note_content"))

    for pwd in passwords:
        pwd["site_name"] = _decrypt_text(pwd.get("site_name"))
        pwd["username_or_email"] = _decrypt_text(pwd.get("username_or_email"))
        pwd["encrypted_password"] = _decrypt_text(pwd.get("encrypted_password"))

    for bm in bookmarks:
        bm["title"] = _decrypt_text(bm.get("title"))
        bm["url"] = _decrypt_text(bm.get("url"))
        bm["category"] = _decrypt_text(bm.get("category"))

    for auth in auths:
        auth["issuer"] = _decrypt_text(auth.get("issuer"))
        auth["account_name"] = _decrypt_text(auth.get("account_name"))
        auth["secret_key"] = _decrypt_text(auth.get("secret_key"))

    return {"notes": notes, "passwords": passwords, "bookmarks": bookmarks, "auths": auths}


@app.post("/vault/migrate-encryption")
async def migrate_vault_encryption(req: UserScopeReq):
    user_id = req.user_id

    notes = supabase.table("sticky_notes").select("id, note_content").eq("user_id", user_id).execute().data
    for note in notes:
        value = note.get("note_content")
        if value and not str(value).startswith(ENC_PREFIX):
            supabase.table("sticky_notes").update({"note_content": _encrypt_text(value)}).eq("id", note["id"]).execute()

    passwords = supabase.table("saved_passwords").select("id, site_name, username_or_email, encrypted_password").eq("user_id", user_id).execute().data
    for pwd in passwords:
        update_payload = {}
        for key in ["site_name", "username_or_email", "encrypted_password"]:
            value = pwd.get(key)
            if value and not str(value).startswith(ENC_PREFIX):
                update_payload[key] = _encrypt_text(value)
        if update_payload:
            supabase.table("saved_passwords").update(update_payload).eq("id", pwd["id"]).execute()

    bookmarks = supabase.table("bookmarks").select("id, title, url, category").eq("user_id", user_id).execute().data
    for bm in bookmarks:
        update_payload = {}
        for key in ["title", "url", "category"]:
            value = bm.get(key)
            if value and not str(value).startswith(ENC_PREFIX):
                update_payload[key] = _encrypt_text(value)
        if update_payload:
            supabase.table("bookmarks").update(update_payload).eq("id", bm["id"]).execute()

    auths = supabase.table("authenticator_seeds").select("id, issuer, account_name, secret_key").eq("user_id", user_id).execute().data
    for auth in auths:
        update_payload = {}
        for key in ["issuer", "account_name", "secret_key"]:
            value = auth.get(key)
            if value and not str(value).startswith(ENC_PREFIX):
                update_payload[key] = _encrypt_text(value)
        if update_payload:
            supabase.table("authenticator_seeds").update(update_payload).eq("id", auth["id"]).execute()

    return {"status": "ok", "message": "Vault rows migrated to encrypted storage"}

@app.post("/vault/notes")
async def add_note(n: NoteReq):
    supabase.table("sticky_notes").insert({
        "user_id": n.user_id,
        "note_content": _encrypt_text(n.content),
    }).execute()
    return {"status": "ok"}

@app.delete("/vault/notes/{id}")
async def del_note(id: str):
    supabase.table("sticky_notes").delete().eq("id", id).execute()
    return {"status": "ok"}


@app.put("/vault/notes/{id}")
async def update_note(id: str, n: NoteUpdateReq):
    existing = supabase.table("sticky_notes").select("id").eq("id", id).eq("user_id", n.user_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Note not found")

    supabase.table("sticky_notes").update({
        "note_content": _encrypt_text(n.content),
    }).eq("id", id).eq("user_id", n.user_id).execute()
    return {"status": "ok"}

@app.post("/vault/passwords")
async def add_pass(p: PassReq):
    supabase.table("saved_passwords").insert({
        "user_id": p.user_id,
        "site_name": _encrypt_text(p.site_name),
        "username_or_email": _encrypt_text(p.username_or_email),
        "encrypted_password": _encrypt_text(p.password),
    }).execute()
    return {"status": "ok"}

@app.delete("/vault/passwords/{id}")
async def del_pass(id: str):
    supabase.table("saved_passwords").delete().eq("id", id).execute()
    return {"status": "ok"}

@app.post("/vault/bookmarks")
async def add_bm(b: BookmarkReq):
    supabase.table("bookmarks").insert({
        "user_id": b.user_id,
        "title": _encrypt_text(b.title),
        "url": _encrypt_text(b.url),
        "category": _encrypt_text("General"),
    }).execute()
    return {"status": "ok"}

@app.delete("/vault/bookmarks/{id}")
async def del_bm(id: str):
    supabase.table("bookmarks").delete().eq("id", id).execute()
    return {"status": "ok"}

@app.post("/vault/authenticator")
async def add_auth(a: AuthSeedReq):
    supabase.table("authenticator_seeds").insert({
        "user_id": a.user_id,
        "issuer": _encrypt_text(a.issuer),
        "account_name": _encrypt_text(a.account_name),
        "secret_key": _encrypt_text(a.secret_key),
    }).execute()
    return {"status": "ok"}


@app.get("/vault/authenticator/{id}/qr")
async def auth_qr(id: str, user_id: str):
    res = supabase.table("authenticator_seeds").select("id, issuer, account_name, secret_key").eq("id", id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Authenticator entry not found")

    entry = res.data[0]
    issuer = _decrypt_text(entry.get("issuer")) or "CipherGate"
    account_name = _decrypt_text(entry.get("account_name")) or "user@ciphergate"
    secret_key = _decrypt_text(entry.get("secret_key"))

    if not _is_valid_totp_secret(secret_key):
        raise HTTPException(status_code=400, detail="Secret key is not a valid TOTP seed")

    otpauth_uri = auth_logic.get_totp_uri_custom(account_name=account_name, secret=secret_key, issuer=issuer)
    qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={quote(otpauth_uri, safe='')}"
    return {
        "id": id,
        "issuer": issuer,
        "account_name": account_name,
        "otpauth_uri": otpauth_uri,
        "qr_code_url": qr_code_url,
    }


@app.get("/vault/authenticator/codes")
async def auth_codes(user_id: str):
    res = supabase.table("authenticator_seeds").select("id, issuer, account_name, secret_key").eq("user_id", user_id).execute()

    items = []
    for entry in (res.data or []):
        issuer = _decrypt_text(entry.get("issuer")) or "Unknown"
        account_name = _decrypt_text(entry.get("account_name")) or ""
        secret_key = _decrypt_text(entry.get("secret_key"))
        if not _is_valid_totp_secret(secret_key):
            continue

        try:
            totp_payload = _build_totp_payload(secret_key)
        except Exception:
            continue

        items.append({
            "id": entry.get("id"),
            "issuer": issuer,
            "account_name": account_name,
            **totp_payload,
        })

    return {"items": items}


@app.post("/chat/rooms")
async def create_chat_room(req: ChatCreateReq):
    room_id = uuid.uuid4().hex[:10]
    session_id = uuid.uuid4().hex
    display_name = _random_display_name()
    now_iso = _now_iso()
    invite_policy = _normalize_invite_policy(req.invite_policy)
    approved = [str(x).strip() for x in (req.approved_user_ids or []) if str(x).strip()]
    if req.user_id and req.user_id not in approved:
        approved.append(req.user_id)
    settings = {
        "invite_policy": invite_policy,
        "passcode_hash": _hash_passcode(req.passcode) if (req.passcode or "").strip() else None,
        "approved_user_ids": approved,
    }

    # Save to database
    try:
        supabase.table("chat_rooms").insert({
            "id": room_id,
            "creator_user_id": req.user_id,
            "created_at": now_iso,
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat room: {str(e)}")

    # Save creator as participant
    try:
        supabase.table("chat_participants").insert({
            "room_id": room_id,
            "user_id": req.user_id,
            "session_id": session_id,
            "display_name": display_name,
            "joined_at": now_iso,
        }).execute()
    except Exception:
        pass

    # Also keep in-memory for real-time features
    now_ts = time.time()
    CHAT_ROOMS[room_id] = {
        "id": room_id,
        "created_at": now_iso,
        "creator_session_id": session_id,
        "creator_user_id": req.user_id,
        "participants": {
            session_id: {
                "user_id": req.user_id,
                "display_name": display_name,
                "joined_at": now_iso,
                "last_seen": now_iso,
                "last_seen_ts": now_ts,
            }
        },
        "messages": [],
        "settings": settings,
    }
    CHAT_ROOM_SETTINGS[room_id] = settings
    _persist_system_event(room_id, "room_settings", settings)
    _audit_event("chat_room_create", user_id=req.user_id, room_id=room_id, status="success", meta={"invite_policy": invite_policy, "has_passcode": bool(settings["passcode_hash"])})

    return {
        "room_id": room_id,
        "session_id": session_id,
        "display_name": display_name,
        "is_creator": True,
        "joined_at": now_iso,
        "room_settings": {
            "invite_policy": settings["invite_policy"],
            "has_passcode": bool(settings.get("passcode_hash")),
            "approved_count": len(settings.get("approved_user_ids") or []),
        },
    }


@app.post("/chat/rooms/{room_id}/join")
async def join_chat_room(room_id: str, req: ChatJoinReq):
    room = _get_room_or_404(room_id)
    _prune_stale_participants(room)

    session_id = uuid.uuid4().hex
    display_name = _random_display_name()
    now_ts = time.time()
    now_iso = _now_iso()
    settings = _sanitize_room_settings(room.get("settings") or CHAT_ROOM_SETTINGS.get(room_id) or _default_room_settings())
    room["settings"] = settings
    CHAT_ROOM_SETTINGS[room_id] = settings

    requester_id = (req.user_id or "").strip() or None
    creator_id = room.get("creator_user_id")
    approved_ids = set(settings.get("approved_user_ids") or [])

    if settings.get("invite_policy") == "approval":
        is_allowed = bool(requester_id and (requester_id == creator_id or requester_id in approved_ids))
        if not is_allowed:
            pending = _register_approval_request(room_id, requester_id)
            if room_id in CHAT_ROOMS:
                CHAT_ROOMS[room_id]["approval_requests"] = pending
            _audit_event("chat_room_join", user_id=requester_id, room_id=room_id, status="failed", meta={"reason": "approval_required"})
            raise HTTPException(status_code=403, detail={"code": "ROOM_APPROVAL_REQUIRED", "message": "This room requires owner approval"})

    if settings.get("passcode_hash") and requester_id != creator_id:
        if not (req.passcode or "").strip():
            _audit_event("chat_room_join", user_id=requester_id, room_id=room_id, status="failed", meta={"reason": "passcode_required"})
            raise HTTPException(status_code=403, detail={"code": "ROOM_PASSCODE_REQUIRED", "message": "Room passcode is required"})
        if not _verify_passcode(settings.get("passcode_hash"), req.passcode):
            _audit_event("chat_room_join", user_id=requester_id, room_id=room_id, status="failed", meta={"reason": "passcode_invalid"})
            raise HTTPException(status_code=403, detail={"code": "ROOM_PASSCODE_INVALID", "message": "Invalid room passcode"})

    # Save participant to database
    try:
        supabase.table("chat_participants").insert({
            "room_id": room_id,
            "user_id": requester_id,
            "session_id": session_id,
            "display_name": display_name,
            "joined_at": now_iso,
        }).execute()
    except Exception:
        pass

    room["participants"][session_id] = {
        "user_id": requester_id,
        "display_name": display_name,
        "joined_at": now_iso,
        "last_seen": now_iso,
        "last_seen_ts": now_ts,
    }
    room.setdefault("approval_requests", CHAT_ROOM_APPROVAL_REQUESTS.get(room_id) or [])

    is_creator = bool(room.get("creator_user_id") and room.get("creator_user_id") == requester_id)
    _audit_event("chat_room_join", user_id=requester_id, room_id=room_id, status="success", meta={"invite_policy": settings.get("invite_policy")})

    return {
        "room_id": room_id,
        "session_id": session_id,
        "display_name": display_name,
        "is_creator": is_creator,
        "creator_user_id": creator_id,
        "joined_at": now_iso,
        "room_settings": {
            "invite_policy": settings["invite_policy"],
            "has_passcode": bool(settings.get("passcode_hash")),
            "approved_count": len(settings.get("approved_user_ids") or []),
        },
    }


@app.get("/chat/rooms/{room_id}/state")
async def chat_room_state(room_id: str, session_id: str):
    room = _get_room_or_404(room_id)
    if session_id not in room["participants"]:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    now_ts = time.time()
    now_iso = _now_iso()
    room["participants"][session_id]["last_seen"] = now_iso
    room["participants"][session_id]["last_seen_ts"] = now_ts

    participants = [
        {
            "session_id": sid,
            "display_name": member.get("display_name", "Anonymous"),
            "joined_at": member.get("joined_at"),
        }
        for sid, member in room["participants"].items()
        if now_ts - float(member.get("last_seen_ts", 0)) <= CHAT_STALE_SECONDS
    ]

    participant_user_id = room["participants"][session_id].get("user_id")
    is_creator = bool(room.get("creator_user_id") and participant_user_id and room.get("creator_user_id") == participant_user_id)
    settings = _sanitize_room_settings(room.get("settings") or CHAT_ROOM_SETTINGS.get(room_id) or _default_room_settings())
    room["settings"] = settings
    pending_requests = _sanitize_approval_requests(room.get("approval_requests") or CHAT_ROOM_APPROVAL_REQUESTS.get(room_id) or [])
    CHAT_ROOM_APPROVAL_REQUESTS[room_id] = pending_requests
    room["approval_requests"] = pending_requests
    typing_users = _active_typing_users(room_id, exclude_session_id=session_id)

    return {
        "room_id": room_id,
        "is_creator": is_creator,
        "creator_user_id": room.get("creator_user_id"),
        "participants_count": _active_count(room),
        "participants": participants,
        "room_settings": {
            "invite_policy": settings["invite_policy"],
            "has_passcode": bool(settings.get("passcode_hash")),
            "approved_count": len(settings.get("approved_user_ids") or []),
            "viewer_is_approved": bool(participant_user_id and (participant_user_id in (settings.get("approved_user_ids") or []) or participant_user_id == room.get("creator_user_id"))),
        },
        "pending_approval_requests": pending_requests if is_creator else [],
        "typing_users": typing_users,
    }


@app.get("/chat/rooms/{room_id}/messages")
async def chat_room_messages(room_id: str, session_id: str):
    room = _get_room_or_404(room_id)
    if session_id not in room["participants"]:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    # Lazy-load user messages for rooms reconstructed from DB to make join/state fast.
    if not room.get("messages_loaded", True):
        db_messages = (
            supabase
            .table("chat_messages")
            .select("id, session_id, author, content, created_at")
            .eq("room_id", room_id)
            .neq("author", CHAT_SYSTEM_AUTHOR)
            .order("created_at", desc=False)
            .execute()
            .data
            or []
        )
        room["messages"] = [
            {
                "id": m.get("id"),
                "session_id": m.get("session_id"),
                "author": m.get("author", "Anonymous"),
                "content": m.get("content"),
                "created_at": m.get("created_at"),
            }
            for m in db_messages
        ]
        room["messages_loaded"] = True

    return {"messages": room["messages"]}


@app.post("/chat/rooms/{room_id}/messages")
async def chat_send_message(room_id: str, req: ChatSendReq):
    room = _get_room_or_404(room_id)
    member = room["participants"].get(req.session_id)
    if not member:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message content is required")

    now_ts = time.time()
    now_iso = _now_iso()
    member["last_seen"] = now_iso
    member["last_seen_ts"] = now_ts

    message_id = uuid.uuid4().hex
    author = member.get("display_name", "Anonymous")

    # Save message to database
    try:
        supabase.table("chat_messages").insert({
            "id": message_id,
            "room_id": room_id,
            "session_id": req.session_id,
            "author": author,
            "content": content,
            "created_at": now_iso,
        }).execute()
    except Exception:
        pass

    message = {
        "id": message_id,
        "session_id": req.session_id,
        "author": author,
        "content": content,
        "created_at": now_iso,
    }
    room["messages"].append(message)
    room_typing = CHAT_ROOM_TYPING.get(room_id) or {}
    room_typing.pop(req.session_id, None)
    if room_typing:
        CHAT_ROOM_TYPING[room_id] = room_typing
    elif room_id in CHAT_ROOM_TYPING:
        CHAT_ROOM_TYPING.pop(room_id, None)

    return {"status": "ok", "message": message}


@app.post("/chat/rooms/{room_id}/leave")
async def chat_leave_room(room_id: str, req: ChatLeaveReq):
    room = _get_room_or_404(room_id)
    room["participants"].pop(req.session_id, None)
    room_typing = CHAT_ROOM_TYPING.get(room_id) or {}
    room_typing.pop(req.session_id, None)
    if room_typing:
        CHAT_ROOM_TYPING[room_id] = room_typing
    elif room_id in CHAT_ROOM_TYPING:
        CHAT_ROOM_TYPING.pop(room_id, None)
    _audit_event("chat_room_leave", user_id=req.user_id, room_id=room_id, status="success")
    # Note: We keep the participant record in the database for history
    return {"status": "ok"}


@app.post("/chat/rooms/{room_id}/typing")
async def chat_room_typing(room_id: str, req: ChatTypingReq):
    room = _get_room_or_404(room_id)
    member = room["participants"].get(req.session_id)
    if not member:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    room_typing = CHAT_ROOM_TYPING.get(room_id) or {}
    if req.is_typing:
        room_typing[req.session_id] = {
            "user_id": req.user_id,
            "display_name": member.get("display_name", "Anonymous"),
            "last_seen_ts": time.time(),
        }
    else:
        room_typing.pop(req.session_id, None)

    if room_typing:
        CHAT_ROOM_TYPING[room_id] = room_typing
    elif room_id in CHAT_ROOM_TYPING:
        CHAT_ROOM_TYPING.pop(room_id, None)

    return {"status": "ok", "typing_users": _active_typing_users(room_id, exclude_session_id=req.session_id)}


@app.delete("/chat/rooms/{room_id}/messages/{message_id}")
async def chat_delete_message(room_id: str, message_id: str, user_id: str, session_id: str):
    room = _get_room_or_404(room_id)
    member = room["participants"].get(session_id)
    if not member:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    msg_idx = next((i for i, m in enumerate(room["messages"]) if m.get("id") == message_id), -1)
    if msg_idx == -1:
        raise HTTPException(status_code=404, detail="Message not found")

    msg = room["messages"][msg_idx]
    if msg.get("session_id") != session_id and room.get("creator_user_id") != user_id:
        raise HTTPException(status_code=403, detail={"code": "NOT_MESSAGE_OWNER", "message": "Can only delete your own messages"})

    room["messages"][msg_idx]["content"] = "[message unsent]"
    try:
        supabase.table("chat_messages").update({"content": "[message unsent]"}).eq("id", message_id).execute()
    except Exception:
        pass
    _audit_event("chat_message_delete", user_id=user_id, room_id=room_id, message_id=message_id, status="success")
    return {"status": "ok"}


@app.post("/chat/rooms/{room_id}/messages/{message_id}/read")
async def chat_read_message(room_id: str, message_id: str, user_id: str, session_id: str):
    room = _get_room_or_404(room_id)
    member = room["participants"].get(session_id)
    if not member:
        raise HTTPException(status_code=403, detail={"code": "NOT_ROOM_PARTICIPANT", "message": "Not a room participant"})

    msg_idx = next((i for i, m in enumerate(room["messages"]) if m.get("id") == message_id), -1)
    if msg_idx == -1:
        raise HTTPException(status_code=404, detail="Message not found")

    msg = room["messages"][msg_idx]
    if "read_by" not in msg:
        msg["read_by"] = []
    if user_id not in msg["read_by"]:
        msg["read_by"].append(user_id)

    return {"status": "ok"}


@app.patch("/chat/rooms/{room_id}/settings")
async def update_chat_room_settings(room_id: str, req: ChatUpdateSettingsReq):
    room = _get_room_or_404(room_id)
    if room.get("creator_user_id") != req.user_id:
        raise HTTPException(status_code=403, detail={"code": "ONLY_CREATOR", "message": "Only the room creator can update settings"})

    settings = _sanitize_room_settings(room.get("settings") or CHAT_ROOM_SETTINGS.get(room_id) or _default_room_settings())
    if req.invite_policy is not None:
        settings["invite_policy"] = _normalize_invite_policy(req.invite_policy)
    if req.clear_passcode:
        settings["passcode_hash"] = None
    elif (req.passcode or "").strip():
        settings["passcode_hash"] = _hash_passcode(req.passcode)
    if req.approved_user_ids is not None:
        settings["approved_user_ids"] = [str(x).strip() for x in req.approved_user_ids if str(x).strip()]
        if req.user_id not in settings["approved_user_ids"]:
            settings["approved_user_ids"].append(req.user_id)

    room["settings"] = settings
    CHAT_ROOM_SETTINGS[room_id] = settings
    _persist_system_event(room_id, "room_settings", settings)
    _audit_event("chat_room_settings_update", user_id=req.user_id, room_id=room_id, status="success", meta={"invite_policy": settings.get("invite_policy"), "has_passcode": bool(settings.get("passcode_hash"))})

    return {
        "status": "ok",
        "room_settings": {
            "invite_policy": settings["invite_policy"],
            "has_passcode": bool(settings.get("passcode_hash")),
            "approved_count": len(settings.get("approved_user_ids") or []),
        },
    }


@app.post("/chat/rooms/{room_id}/approvals")
async def update_chat_room_approvals(room_id: str, req: ChatApprovalReq):
    room = _get_room_or_404(room_id)
    if room.get("creator_user_id") != req.user_id:
        raise HTTPException(status_code=403, detail={"code": "ONLY_CREATOR", "message": "Only the room creator can update approvals"})

    target_user_id = (req.target_user_id or "").strip()
    if not target_user_id:
        raise HTTPException(status_code=400, detail={"code": "INVALID_TARGET", "message": "target_user_id is required"})

    settings = _sanitize_room_settings(room.get("settings") or CHAT_ROOM_SETTINGS.get(room_id) or _default_room_settings())
    approved = set(settings.get("approved_user_ids") or [])
    action = (req.action or "add").strip().lower()
    if action == "remove":
        approved.discard(target_user_id)
    else:
        approved.add(target_user_id)
    approved.add(req.user_id)
    settings["approved_user_ids"] = sorted(approved)
    _resolve_approval_request(room_id, target_user_id)

    room["settings"] = settings
    CHAT_ROOM_SETTINGS[room_id] = settings
    room["approval_requests"] = CHAT_ROOM_APPROVAL_REQUESTS.get(room_id) or []
    _persist_system_event(room_id, "room_settings", settings)
    _audit_event("chat_room_approval_update", user_id=req.user_id, room_id=room_id, status="success", meta={"action": action, "target_user_id": target_user_id})

    return {
        "status": "ok",
        "room_settings": {
            "invite_policy": settings["invite_policy"],
            "has_passcode": bool(settings.get("passcode_hash")),
            "approved_count": len(settings.get("approved_user_ids") or []),
        },
    }


@app.delete("/chat/rooms/{room_id}")
async def delete_chat_room(room_id: str, req: ChatDeleteReq):
    room = _get_room_or_404(room_id)
    if room.get("creator_user_id") != req.user_id:
        raise HTTPException(status_code=403, detail={"code": "ONLY_CREATOR", "message": "Only the room creator can delete this room"})

    reason = (req.deletion_reason or "").strip() or "Deleted by the room owner"
    tombstone = {
        "room_id": room_id,
        "deleted_at": _now_iso(),
        "deleted_by": req.user_id,
        "reason": reason,
    }

    # Keep a tombstone marker in the room stream so shared links return 410 with context.
    try:
        _persist_system_event(room_id, "room_deleted", tombstone)
        supabase.table("chat_participants").delete().eq("room_id", room_id).execute()
        # Keep system events for tombstone/settings reconstruction; remove only user messages.
        supabase.table("chat_messages").delete().eq("room_id", room_id).neq("author", CHAT_SYSTEM_AUTHOR).execute()
    except Exception:
        pass

    # Mark deleted in memory and remove active room caches.
    CHAT_DELETED_ROOMS[room_id] = tombstone
    CHAT_ROOMS.pop(room_id, None)
    CHAT_ROOM_SETTINGS.pop(room_id, None)
    CHAT_ROOM_APPROVAL_REQUESTS.pop(room_id, None)
    _audit_event("chat_room_delete", user_id=req.user_id, room_id=room_id, status="success", meta={"reason": reason})
    return {"status": "deleted", "tombstone": tombstone}

@app.delete("/vault/authenticator/{id}")
async def del_auth(id: str):
    supabase.table("authenticator_seeds").delete().eq("id", id).execute()
    return {"status": "ok"}

# --- RECOVERY KIT ---
@app.get("/vault/recovery-kit")
async def get_recovery_kit(user_id: str):
    res = supabase.table("users").select("id, email").eq("id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = res.data[0]
    digest = hashlib.sha256(f"{user['id']}:{user['email']}:{_recovery_secret()}".encode("utf-8")).hexdigest().upper()
    codes = ["-".join([digest[i:i+4], digest[i+4:i+8]]) for i in range(0, 80, 8)]
    master_key = _derive_user_recovery_key(user_id=user["id"], email=user["email"])
    return {
        "email": user["email"],
        "master_key": master_key,
        "backup_codes": codes
    }

# --- MISC ---
@app.get("/chat/rooms/user/{user_id}")
async def get_user_chat_rooms(user_id: str):
    """Get all chat rooms created by or joined by this user"""
    try:
        # Get rooms created by user
        created = supabase.table("chat_rooms").select("id, created_at, creator_user_id").eq("creator_user_id", user_id).execute().data or []
        
        # Get rooms user has participated in
        participated_data = supabase.table("chat_participants").select("room_id").eq("user_id", user_id).execute().data or []
        participated_room_ids = [p["room_id"] for p in participated_data]
        
        # Get full room data for participated rooms
        participated = []
        if participated_room_ids:
            participated = supabase.table("chat_rooms").select("id, created_at, creator_user_id").in_("id", participated_room_ids).execute().data or []
        
        # Get message counts and latest message
        all_rooms = {r["id"]: {**r, "message_count": 0, "latest_message_at": None} for r in created + participated}

        # Hide rooms that were deleted (tombstoned).
        active_room_ids = [rid for rid in all_rooms.keys() if not _build_tombstone(rid)]
        
        for room_id in active_room_ids:
            msg_count = supabase.table("chat_messages").select("id, author").eq("room_id", room_id).execute()
            all_rooms[room_id]["message_count"] = len([m for m in (msg_count.data or []) if m.get("author") != CHAT_SYSTEM_AUTHOR])
            
            latest = supabase.table("chat_messages").select("created_at").eq("room_id", room_id).neq("author", CHAT_SYSTEM_AUTHOR).order("created_at", desc=True).limit(1).execute()
            if latest.data:
                all_rooms[room_id]["latest_message_at"] = latest.data[0]["created_at"]
        
        return {
            "rooms": [all_rooms[rid] for rid in active_room_ids],
            "count": len(active_room_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat rooms: {str(e)}")


@app.get("/audit/user/{user_id}")
async def get_user_audit_events(user_id: str, limit: int = 100):
    cap = max(1, min(int(limit or 100), 500))
    rows = [entry for entry in reversed(CHAT_AUDIT_LOGS) if entry.get("user_id") == user_id]
    return {"events": rows[:cap], "count": min(len(rows), cap)}


@app.post("/secrets/one-time")
async def create_one_time_secret(req: OneTimeSecretCreateReq):
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Secret note cannot be empty")
    if len(content) > 8000:
        raise HTTPException(status_code=400, detail="Secret note is too long")

    expires_minutes = max(ONE_TIME_SECRET_MIN_MINUTES, min(int(req.expires_in_minutes or 30), ONE_TIME_SECRET_MAX_MINUTES))
    now_dt = datetime.now(timezone.utc)
    expires_dt = now_dt + timedelta(minutes=expires_minutes)
    token = secrets.token_urlsafe(20)
    ONE_TIME_SECRETS[token] = {
        "token": token,
        "created_by": req.user_id,
        "content": _encrypt_text(content),
        "created_at": now_dt.isoformat(),
        "expires_at": expires_dt.isoformat(),
        "expires_ts": expires_dt.timestamp(),
        "consumed": False,
    }
    _audit_event(
        "one_time_secret_create",
        user_id=req.user_id,
        status="success",
        meta={"expires_in_minutes": expires_minutes},
    )
    _cleanup_one_time_secrets()
    return {
        "token": token,
        "expires_at": expires_dt.isoformat(),
        "link_path": f"/dashboard/secrets?secret={token}",
    }


@app.get("/secrets/one-time/{token}")
async def consume_one_time_secret(token: str, user_id: str | None = None):
    payload = ONE_TIME_SECRETS.get(token)
    if not payload:
        _cleanup_one_time_secrets()
        raise HTTPException(status_code=404, detail="Secret note not found")

    if bool(payload.get("consumed")):
        ONE_TIME_SECRETS.pop(token, None)
        raise HTTPException(status_code=410, detail="Secret note already opened")

    if float(payload.get("expires_ts", 0)) <= time.time():
        ONE_TIME_SECRETS.pop(token, None)
        raise HTTPException(status_code=410, detail="Secret note expired")

    payload["consumed"] = True
    payload["consumed_at"] = _now_iso()
    ONE_TIME_SECRETS.pop(token, None)
    _audit_event(
        "one_time_secret_read",
        user_id=user_id or payload.get("created_by"),
        status="success",
        meta={"creator_user_id": payload.get("created_by")},
    )
    _cleanup_one_time_secrets()
    return {
        "content": _decrypt_text(payload.get("content")) or "",
        "created_at": payload.get("created_at"),
        "expires_at": payload.get("expires_at"),
        "consumed_at": payload.get("consumed_at"),
    }

@app.get("/misc/security-quotes")
async def security_quotes():
    quotes = [
        "A strong password is your first line of defense against cyber threats.",
        "Never reuse passwords across different platforms—attackers rely on it.",
        "Enable two-factor authentication wherever possible to protect your accounts.",
        "Your data is as secure as your weakest password.",
        "Regularly change passwords for accounts with sensitive information.",
        "Use a passphrase instead of a simple word for better security.",
    ]
    import random
    return {"quotes": random.sample(quotes, min(2, len(quotes)))}

# --- TOOLS ---
@app.post("/tools/password/generate")
async def generate_passwords(req: PasswordGeneratorReq):
    variants = dashboard_manager.generate_password_variants(
        base_string=req.base_string,
        length=req.length,
        include_uppercase=req.include_uppercase,
        include_numbers=req.include_numbers,
        include_special=req.include_special,
        exclude_ambiguous=req.exclude_ambiguous,
        count=req.count,
    )
    return {"variants": variants}

@app.post("/tools/ai-crack-estimate")
async def ai_crack_estimate(req: AICrackEstimateReq):
    # Prepare the prompt with the dynamic data
    prompt = f"""
    Analyze the following password metrics:
    Theoretical Entropy: {req.entropyBits} bits
    Expected Guesses to Crack (Zxcvbn heuristic): {req.expectedGuesses}

    Calculate and format the estimated crack times for:
    1. Online attack (rate limited to 100 guesses/sec)
    2. Offline PC (100 Million guesses/sec)
    3. High-end GPU Array (100 Billion guesses/sec)

    Also provide a brief 1-2 sentence 'aiNote' explaining the vulnerability or strength of these metrics in the real world.

    Return the results as a strict JSON object with the following keys: "online", "offline", "gpu", and "aiNote". The time estimates should be human-readable strings (e.g., "2 years", "3 months", "5000 years").

    IMPORTANT: Return ONLY a raw JSON object. Do not wrap it in markdown blockticks (```json). The keys must be exact.
    Format:
    {{
      "online": "time string",
      "offline": "time string",
      "gpu": "time string",
      "aiNote": "your brief explanation"
    }}
    """

    try:
        # Initialize the model and force it to return strict JSON
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2, 
                response_mime_type="application/json"
            )
        )

        
        raw_text = response.text
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

       
        parsed_data = json.loads(raw_text)
        return parsed_data

    except Exception as e:
        print(f"Gemini API Error: {e}")
        
        raise HTTPException(status_code=500, detail="Failed to fetch AI estimate")


@app.post("/tools/authenticator/scan-qr")
async def scan_authenticator_qr(req: QrDecodeReq):
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        raise HTTPException(status_code=500, detail="QR scanner dependency missing. Install opencv-python-headless and numpy")

    payload = (req.image_data or "").strip()
    if not payload:
        raise HTTPException(status_code=400, detail="No image data received")

    if payload.startswith("data:"):
        match = re.match(r"^data:[^;]+;base64,(.+)$", payload)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid data URL format")
        payload = match.group(1)

    try:
        raw = base64.b64decode(payload)
    except Exception:
        raise HTTPException(status_code=400, detail="Image data is not valid base64")

    np_data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    detector = cv2.QRCodeDetector()
    decoded_text, points, _ = detector.detectAndDecode(image)
    if not decoded_text:
        raise HTTPException(status_code=400, detail="No QR code detected in the image")

    parsed = _parse_otpauth_uri(decoded_text)
    return {
        "status": "ok",
        **parsed,
    }


@app.get("/tools/audit")
async def audit(user_id: str):
    res = supabase.table("saved_passwords").select("id, encrypted_password").eq("user_id", user_id).execute()
    breached = []
    async with httpx.AsyncClient() as client:
        for r in res.data:
            plain_password = _decrypt_text(r.get("encrypted_password")) or ""
            if not plain_password:
                continue
            sha1 = hashlib.sha1(plain_password.encode()).hexdigest().upper()
            pref, suff = sha1[:5], sha1[5:]
            try:
                resp = await client.get(f"https://api.pwnedpasswords.com/range/{pref}", timeout=5.0)
                if any(line.split(':')[0] == suff for line in resp.text.splitlines()):
                    breached.append(r['id'])
            except: pass
    return {"breached_ids": breached}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
