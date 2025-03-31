from pydantic import BaseModel

class Request(BaseModel):
    prompt: str
    user_id: str  # New parameter to track user requests

# Updated Response Model
class Response(BaseModel):
    response: str
    timestamp: str  # New parameter to track response time