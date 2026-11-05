"""Create .env with random local-only values. Never overwrites an existing file or prints secrets."""
import base64
import secrets
from pathlib import Path

root = Path(__file__).resolve().parent.parent
target = root / ".env"
if target.exists():
    print(".env already exists; left unchanged")
    raise SystemExit(0)

text = (root / ".env.example").read_text()
text = text.replace("DB_PASSWORD=change-me", f"DB_PASSWORD={secrets.token_urlsafe(24)}")
text = text.replace("DEMO_PASSWORD=change-me", f"DEMO_PASSWORD={secrets.token_urlsafe(12)}")
text = text.replace(
    "JWT_SECRET=at-least-32-bytes-of-random-data",
    f"JWT_SECRET={base64.b64encode(secrets.token_bytes(48)).decode()}",
)
target.write_text(text)
print(".env created")
