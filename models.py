from pydantic import BaseModel

class Request(BaseModel):
    prompt: str
    user_id: str  

# Updated Response Model
class Response(BaseModel):
    response: str
    timestamp: str