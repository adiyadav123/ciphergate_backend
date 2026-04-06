import secrets
import string

LEET_MAP = {
    "a": "@",
    "e": "3",
    "i": "1",
    "o": "0",
    "s": "$",
    "t": "7",
}


def _normalize_base(base_string: str) -> str:
    filtered = "".join(ch for ch in (base_string or "").strip() if ch.isalnum())
    return filtered or "cipher"


def _maybe_leet(ch: str) -> str:
    lower = ch.lower()
    if lower in LEET_MAP and secrets.randbelow(100) < 35:
        return LEET_MAP[lower]
    return ch


def _char_pool(include_uppercase: bool, include_numbers: bool, include_special: bool, exclude_ambiguous: bool) -> str:
    pool = string.ascii_lowercase
    if include_uppercase:
        pool += string.ascii_uppercase
    if include_numbers:
        pool += string.digits
    if include_special:
        pool += "!@#$%^&*()_+-={}[]|:;<>?,./"

    if exclude_ambiguous:
        ambiguous = set("il1Lo0O")
        pool = "".join(ch for ch in pool if ch not in ambiguous)

    return pool or string.ascii_lowercase


def _mutate_base(base_string: str, include_uppercase: bool, include_numbers: bool) -> str:
    core = _normalize_base(base_string)
    chars = []
    for ch in core:
        transformed = _maybe_leet(ch)
        if include_uppercase and transformed.isalpha() and secrets.randbelow(100) < 40:
            transformed = transformed.upper()
        chars.append(transformed)

    if include_numbers:
        chars.append(str(secrets.randbelow(10)))

    return "".join(chars)


def _ensure_required_chars(password: str, include_uppercase: bool, include_numbers: bool, include_special: bool) -> str:
    if not password:
        return password

    chars = list(password)
    required = []
    if include_uppercase:
        required.append(string.ascii_uppercase)
    if include_numbers:
        required.append(string.digits)
    if include_special:
        required.append("!@#$%^&*()_+-={}[]|:;<>?,./")

    for charset in required:
        if any(ch in charset for ch in chars):
            continue
        idx = secrets.randbelow(len(chars))
        chars[idx] = secrets.choice(charset)

    return "".join(chars)


def generate_password_variant(
    base_string: str,
    length: int = 16,
    include_uppercase: bool = True,
    include_numbers: bool = True,
    include_special: bool = True,
    exclude_ambiguous: bool = True,
) -> str:
    length = max(8, min(64, int(length)))
    pool = _char_pool(include_uppercase, include_numbers, include_special, exclude_ambiguous)
    seed = _mutate_base(base_string, include_uppercase, include_numbers)

    if len(seed) >= length:
        candidate = seed[:length]
    else:
        fill_len = length - len(seed)
        filler = "".join(secrets.choice(pool) for _ in range(fill_len))
        candidate = f"{seed}{filler}"

    mixed = list(candidate)
    for i in range(len(mixed) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        mixed[i], mixed[j] = mixed[j], mixed[i]

    return _ensure_required_chars("".join(mixed), include_uppercase, include_numbers, include_special)


def generate_password_variants(
    base_string: str,
    length: int = 16,
    include_uppercase: bool = True,
    include_numbers: bool = True,
    include_special: bool = True,
    exclude_ambiguous: bool = True,
    count: int = 5,
):
    qty = max(1, min(10, int(count)))
    variants = []
    for _ in range(qty):
        password = generate_password_variant(
            base_string=base_string,
            length=length,
            include_uppercase=include_uppercase,
            include_numbers=include_numbers,
            include_special=include_special,
            exclude_ambiguous=exclude_ambiguous,
        )
        variants.append({
            "password": password,
            "strength": check_password_strength(password),
        })
    return variants

def generate_strong_password(length=16):
    """Generates a high-entropy password for the user."""
    length = max(8, min(64, int(length)))
    characters = string.ascii_letters + string.digits + "!@#$%^&*()"
    return ''.join(secrets.choice(characters) for _ in range(length))


def check_password_strength(password: str):
    """Evaluates password based on length and variety."""
    score = 0
    if len(password) >= 12: score += 1
    if any(c.isupper() for c in password): score += 1
    if any(c.isdigit() for c in password): score += 1
    if any(c in "!@#$%^&*()_+-={}[]|:;<>?,./" for c in password): score += 1
    
    levels = {0: "Weak", 1: "Fair", 2: "Good", 3: "Strong", 4: "Excellent"}
    return levels.get(score, "Weak")
