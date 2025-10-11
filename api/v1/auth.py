# back/routers/auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional
import os
import jwt
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from api.utils.security import get_password_hash, verify_password  # argon2/bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from api.models import users  # твой ORM
from api.db.db_connector import get_async_session  # Depends возвращает AsyncSession

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))

router = APIRouter( tags=["Auth"])

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_user_by_login(session: AsyncSession, login: str) -> Optional[users]:
    stmt = select(users).where(users.login == login)
    res = await session.execute(stmt)
    return res.scalar_one_or_none()

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(form: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_async_session)):
    # OAuth2PasswordRequestForm имеет поля username, password
    login = form.username
    password = form.password
    # проверка уникальности
    if await get_user_by_login(session, login):
        raise HTTPException(status_code=400, detail="Login already taken")
    # хэшируем пароль
    pwd_hash = get_password_hash(password)
    user = users(login=login, password_hash=pwd_hash, is_active=True, is_verified=False)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token = create_access_token({"sub": user.login, "uid": str(user.id)})
    return Token(access_token=token, user_id=str(user.id))  # bearer тип по умолчанию

@router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_async_session)):
    user = await get_user_by_login(session, form.username)
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")
    token = create_access_token({"sub": user.login, "uid": str(user.id)})
    return Token(access_token=token, user_id=str(user.id))
