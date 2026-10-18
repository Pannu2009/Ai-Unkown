"""Arjan's secret vault. Encrypted storage for tokens, keys, credentials.
Arzh sets the vault key once (env ARJAN_VAULT_KEY or a key file only he
can read). Everything inside is Fernet-encrypted at rest. The agent can
USE a secret (pass to an API call) but the plaintext never hits logs,
diary, changelog, or git.

Brick wall: vault/ is outside ALLOWED write for free browsing — only
vault.py touches it, and only with the key.
"""
import os, json, getpass

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR = os.path.join(HERE, "vault")
VAULT_FILE = os.path.join(VAULT_DIR, "secrets.enc")
KEY_FILE = os.path.join(VAULT_DIR, ".key")


def _key():
    # Arzh's key: env first, then key file. Never in code, never in git.
    k = os.environ.get("ARJAN_VAULT_KEY", "")
    if k:
        return k.encode()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read().strip()
    return None


def init_vault():
    """Arzh runs this once. Generates a key, stores it with 0600."""
    from cryptography.fernet import Fernet
    os.makedirs(VAULT_DIR, exist_ok=True)
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    os.chmod(KEY_FILE, 0o600)
    with open(VAULT_FILE, "w") as f:
        f.write("{}")
    os.chmod(VAULT_FILE, 0o600)
    print("vault initialized. Key at agent/vault/.key (keep it safe, never git).")
    print("Or set ARJAN_VAULT_KEY env instead.")


def _fernet():
    from cryptography.fernet import Fernet
    key = _key()
    if not key:
        raise RuntimeError("vault not initialized: run vault.py init")
    return Fernet(key)


def _load():
    f = _fernet()
    with open(VAULT_FILE) as fh:
        blob = fh.read().strip()
    if not blob or blob == "{}":
        return {}
    return json.loads(f.decrypt(blob.encode()).decode())


def _save(secrets):
    f = _fernet()
    blob = f.encrypt(json.dumps(secrets).encode()).decode()
    with open(VAULT_FILE, "w") as fh:
        fh.write(blob)


def put(name, value):
    """Store a secret. Plaintext never logged."""
    s = _load()
    s[name] = value
    _save(s)
    return True


def get(name):
    """Retrieve a secret for USE (API call). Caller must not log it."""
    return _load().get(name)


def names():
    """List secret names only — never values."""
    return list(_load().keys())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_vault()
    elif len(sys.argv) > 1 and sys.argv[1] == "ls":
        print("secrets:", names())
    else:
        print("usage: vault.py init | vault.py ls")
