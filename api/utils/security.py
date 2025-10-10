from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

import jwt
from starlette import status
from typing_extensions import Annotated
from api.models.token import TokenData
from jwt.exceptions import JWTException
from sqlalchemy import select

from api.models.token import TokenData
from api.models.users import UserBase
# from db.user import retrieve_user_by_login

from api.db.db_models.users import User

from passlib.context import CryptContext

import os

from itsdangerous import URLSafeTimedSerializer, SignatureExpired
import dotenv
# from utils.logger import logger

dotenv.load_dotenv(".env")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def verify_password(plain_password: str,
                    hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        login = payload.get("sub")
        if login is None:
            raise credentials_exception
        token_data = TokenData(login=login)
    except JWTException:
        raise credentials_exception
    user = select(User).where(User.login == token_data.login)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: Annotated[UserBase, Depends(get_current_user)],
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


serializer = URLSafeTimedSerializer(
    secret_key=SECRET_KEY,
    salt="email-configuration"
)


def create_url_safe_token(data: dict):
    token = serializer.dumps(data)

    return token


def decode_url_safe_token(token: str):
    try:
        token_data = serializer.loads(token)

        return token_data

    except Exception as e:
        # logger.error(str(e))
        print(f"Error loading security token: {e}")