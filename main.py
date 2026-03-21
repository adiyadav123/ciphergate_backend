import os
import hashlib
import httpx
import secrets
import string
import re
import base64
import binascii
from urllib.parse import quote, unquote, urlparse, parse_qs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import auth_logic
import dashboard_manager

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

# --- MODELS ---
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
        raise HTTPException(status_code=404, detail="User not found")

    secret = res.data[0].get("otp_secret")
    if not _is_valid_totp_secret(secret):
        raise HTTPException(status_code=400, detail="MFA is not initialized for this account")

    if not auth_logic.verify_totp_code(secret, req.code):
        raise HTTPException(status_code=401, detail="Invalid authenticator code")

    supabase.table("users").update({"is_mfa_enabled": True}).eq("id", req.user_id).execute()
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