from .ml import ml_router
from .user import user_router
from .auth import auth_router
from .default import default_router

routers = [ml_router, user_router, auth_router, default_router]
