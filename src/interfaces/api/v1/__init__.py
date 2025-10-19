from .ml import ml_router
from .user import user_router
from .auth import auth_router

v1_routers = [ml_router, user_router, auth_router]
