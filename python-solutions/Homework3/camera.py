from pydantic import BaseModel


class Camera(BaseModel):
    eye_pos: tuple[float, float, float]
    vup: tuple[float, float, float]
    look_at: tuple[float, float, float]
    vfov: float
    zNear: float
    zFar: float
