from pydantic import BaseModel

class Reward(BaseModel):
    hotkey: str
    reward: int
    epoch: int