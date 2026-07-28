from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, chat, dashboard, health, prompts, resources, retrieve

from app.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(retrieve.router)
app.include_router(ask.router)
app.include_router(resources.router)
app.include_router(dashboard.router)
app.include_router(prompts.router)
