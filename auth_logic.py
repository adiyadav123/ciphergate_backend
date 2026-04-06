import bcrypt 
import pyotp

def hash_password(password: str) -> bytes:
    
    salt = bcrypt.gensalt()
    hashed= bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def generate_totp_secret() -> str:
    return pyotp.random_base32()

def get_totp_uri(email: str, secret: str) -> str:
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=email,
        issuer_name="CipherGate"
    )

def get_totp_uri_custom(account_name: str, secret: str, issuer: str) -> str:
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=account_name,
        issuer_name=issuer
    )

def verify_totp_code(secret: str, code: str) -> bool:
    normalized = (code or "").strip().replace(" ", "")
    if not normalized.isdigit() or len(normalized) != 6:
        return False
    totp = pyotp.totp.TOTP(secret)
    return totp.verify(normalized, valid_window=1)
