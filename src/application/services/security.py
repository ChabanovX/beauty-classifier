from datetime import datetime, timedelta, timezone

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from hashids import Hashids

from src.config import config

password_hash = PasswordHash.recommended()
hashids = Hashids(config.auth.salt, min_length=6)


def _create_access_token(*_, **data) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=config.auth.access_token_expire)
    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})
    return jwt.encode(
        to_encode, config.auth.secret_key, algorithm=config.auth.algorithm
    )


class SecurityService:
    def hash_password(password: str) -> str:
        return password_hash.hash(password, salt=config.auth.salt.encode())

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return password_hash.verify(plain_password, hashed_password)

    def create_access_token(*_, **data) -> str:
        return _create_access_token(**data)

    def decode_access_token(token: str) -> dict | None:
        try:
            payload = jwt.decode(
                token, config.auth.secret_key, algorithms=[config.auth.algorithm]
            )
            return payload
        except InvalidTokenError:
            return None

    def hash_id(id: int) -> str:
        return hashids.encode(id)

    def decode_id(id: str) -> int:
        return hashids.decode(id)[0]

    def refresh_access_token(token: str) -> str:
        payload = jwt.decode(
            token, config.auth.secret_key, algorithms=[config.auth.algorithm]
        )
        return _create_access_token(payload)
