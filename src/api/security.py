import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(
            status_code=500, detail="API key is not configured on the server"
        )
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key")
