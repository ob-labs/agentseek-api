from pydantic import BaseModel


class User(BaseModel):
    identity: str
    is_authenticated: bool = True
