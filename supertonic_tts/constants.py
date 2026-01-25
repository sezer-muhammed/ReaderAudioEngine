from pydantic import BaseModel
from typing import List

class VoiceStyle(BaseModel):
    id: str
    name: str
    gender: str
    description: str

MIN_SPEED = 0.9
MAX_SPEED = 1.4
MIN_STEPS = 3
MAX_STEPS = 14

VOICE_STYLES: List[VoiceStyle] = [
    VoiceStyle(id="F1", name="F1", gender="female", description="Correct and natural female voice with appropriate articulation"),
    VoiceStyle(id="F2", name="F2", gender="female", description="Energetic and lively female voice with a dynamic tone"),
    VoiceStyle(id="F3", name="F3", gender="female", description="Calm, mature, and steady female voice for a reassuring tone"),
    VoiceStyle(id="F4", name="F4", gender="female", description="Professional and neutral female voice suitable for formal delivery"),
    VoiceStyle(id="F5", name="F5", gender="female", description="Professional and attractive female voice with a polished and engaging tone"),
    VoiceStyle(id="M1", name="M1", gender="male", description="Bright, youthful, and vibrant male voice"),
    VoiceStyle(id="M2", name="M2", gender="male", description="Deep, resonant, and low-pitched male voice"),
    VoiceStyle(id="M3", name="M3", gender="male", description="Neutral and balanced male voice for general purpose use"),
    VoiceStyle(id="M4", name="M4", gender="male", description="Smooth, generic, and easy-to-listen-to male voice"),
    VoiceStyle(id="M5", name="M5", gender="male", description="Generic and calm male voice with a gentle delivery"),
]
