import numpy as np

class Robot:
    def __init__(self, arg: float) -> None: ...
    def get_robot_data(self) -> np.ndarray: ...
    def get_init_yaw(self) -> float: ...
    def motor_initialized(self) -> bool: ...
    def set_motor_cmd(self, cmd_array: np.ndarray) -> None: ...
    def set_mode(self, mode: bool) -> None: ...
    def spin(self) -> None: ...
    def stop(self) -> None: ...
