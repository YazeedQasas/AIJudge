from fastapi import APIRouter
from pydantic import BaseModel

from app.llm_client import get_completion

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await get_completion(request.message)
    return ChatResponse(answer=answer)
