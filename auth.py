"""
auth.py
-------
Handles user authentication for the Cognitive Guardrail AI app.

Features:
    - Separate username + password fields
    - Username uniqueness check
    - bcrypt password hashing (industry standard)
    - Local users.json to store credentials
    - Register vs Login flow

Exposes:
    - register_user()   → create new account
    - login_user()      → verify credentials
    - username_exists() → check if username is taken
"""

import json
import bcrypt
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# ── Storage ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
USERS_FILE = ROOT_DIR / "meta" / "users.json"
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_users() -> dict:
    """Loads all users from users.json."""
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    """Saves all users to users.json."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    bcrypt is deliberately slow — resistant to brute force attacks.
    work factor = 12 (industry standard, ~250ms per hash)
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12)
    ).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """Verifies a password against a bcrypt hash."""
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed.encode("utf-8")
    )


def _validate_username(username: str) -> Tuple[bool, str]:
    """
    Validates username format.
    Returns (is_valid, error_message).
    """
    username = username.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must be under 30 characters."
    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username can only contain letters, numbers, hyphens, and underscores."
    if username.startswith("-") or username.startswith("_"):
        return False, "Username cannot start with a hyphen or underscore."

    return True, ""


def _validate_password(password: str) -> Tuple[bool, str]:
    """
    Validates password strength.
    Returns (is_valid, error_message).
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter."

    return True, ""


# ── Public API ────────────────────────────────────────────────────────────────

def username_exists(username: str) -> bool:
    """
    Checks if a username is already taken.

    Args:
        username : username to check (case-insensitive)

    Returns:
        True if username exists
    """
    users = _load_users()
    return username.strip().lower() in users


def register_user(
    username: str,
    password: str,
) -> Tuple[bool, str]:
    """
    Registers a new user.

    Args:
        username : desired username
        password : plain text password (will be hashed)

    Returns:
        (success, message)
        success = True if registered successfully
        message = error description if failed
    """
    username = username.strip().lower()

    # Validate username format
    valid, error = _validate_username(username)
    if not valid:
        return False, error

    # Validate password strength
    valid, error = _validate_password(password)
    if not valid:
        return False, error

    # Check uniqueness
    if username_exists(username):
        return False, f"Username '{username}' is already taken. Please choose another."

    # Hash password and store
    users = _load_users()
    users[username] = {
        "password_hash": _hash_password(password),
        "created_at"   : datetime.utcnow().isoformat(),
        "last_login"   : None,
    }
    _save_users(users)

    print(f"[auth] New user registered: '{username}'")
    return True, f"Account created successfully! Welcome, {username}."


def login_user(
    username: str,
    password: str,
) -> Tuple[bool, str]:
    """
    Verifies login credentials.

    Args:
        username : entered username
        password : entered plain text password

    Returns:
        (success, message)
        success = True if credentials are correct
        message = error description if failed
    """
    username = username.strip().lower()

    if not username or not password:
        return False, "Please enter both username and password."

    users = _load_users()

    # Check if user exists
    if username not in users:
        return False, "Username not found. Please register first."

    # Verify password
    stored_hash = users[username]["password_hash"]
    if not _verify_password(password, stored_hash):
        return False, "Incorrect password. Please try again."

    # Update last login
    users[username]["last_login"] = datetime.utcnow().isoformat()
    _save_users(users)

    print(f"[auth] User logged in: '{username}'")
    return True, f"Welcome back, {username}!"


def get_user_info(username: str) -> Optional[dict]:
    """
    Returns user metadata (without password hash).

    Args:
        username : username to look up

    Returns:
        dict with created_at, last_login — or None if not found
    """
    users = _load_users()
    username = username.strip().lower()

    if username not in users:
        return None

    user = users[username].copy()
    user.pop("password_hash", None)  # never expose the hash
    return user


def delete_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Deletes a user account after verifying password.

    Args:
        username : username to delete
        password : password confirmation

    Returns:
        (success, message)
    """
    success, msg = login_user(username, password)
    if not success:
        return False, msg

    users = _load_users()
    username = username.strip().lower()
    del users[username]
    _save_users(users)

    print(f"[auth] User deleted: '{username}'")
    return True, "Account deleted successfully."


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Registration tests ──")

    # Valid registration
    ok, msg = register_user("test_user", "SecurePass123")
    print(f"Register test_user    : {ok} — {msg}")

    # Duplicate username
    ok, msg = register_user("test_user", "AnotherPass456")
    print(f"Duplicate username    : {ok} — {msg}")

    # Weak password
    ok, msg = register_user("new_user", "weak")
    print(f"Weak password         : {ok} — {msg}")

    # Invalid username
    ok, msg = register_user("a", "ValidPass123")
    print(f"Short username        : {ok} — {msg}")

    print("\n── Login tests ──")

    # Correct credentials
    ok, msg = login_user("test_user", "SecurePass123")
    print(f"Correct login         : {ok} — {msg}")

    # Wrong password
    ok, msg = login_user("test_user", "WrongPassword")
    print(f"Wrong password        : {ok} — {msg}")

    # Non-existent user
    ok, msg = login_user("ghost_user", "SomePass123")
    print(f"Non-existent user     : {ok} — {msg}")

    print("\n── User info ──")
    info = get_user_info("test_user")
    print(f"User info             : {info}")

    print("\n── Cleanup ──")
    ok, msg = delete_user("test_user", "SecurePass123")
    print(f"Delete test_user      : {ok} — {msg}")
