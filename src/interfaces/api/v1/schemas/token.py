from pydantic import BaseModel


class TokenData(BaseModel):
    iat: int
    exp: int
    username: str
    user_id: int
    role: str


class Token(TokenData):
    token: str
    type: str
