import zmq
import numpy as np
import threading
import time
from config import *


class BallDetector:
    def __init__(self, port="12345") -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://127.0.0.1:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

        self.ball_pos = np.array([0.0, 0.0, 0.0])
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.opponent_pos = np.array([0.0, 0.0, 0.0])
        self.mode = "base" # "base", "camera", "global"

        # Start a thread to run the refresh method asynchronously
        self.refresh_thread = threading.Thread(target=self.run_refresh)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()

    def run_refresh(self):
        while True:
            self.refresh()

    def refresh(self):
        resp = self.socket.recv_json()
        self.ball_pos[:2] = resp["ball_x"], resp["ball_y"]
        self.robot_pos[:2] = resp["robot_x"], resp["robot_y"]
        self.opponent_pos[:2] = resp["opponent_x"], resp["opponent_y"]
        self.mode = resp["mode"]

    def get_pos(self):
        return self.ball_pos.copy(), self.robot_pos.copy(), self.opponent_pos.copy(), self.mode


# when the ball is 0.5 ahead, 0.3 - 0.5 left, sharp left turn is good
# when the ball is 0.5 ahead, 0.5 right, a not sharp right turn

if __name__ == "__main__":
    import robot_interface
    import time

    robot = robot_interface.Robot(50, 1)

    bd = BallDetector(ball_detector_server_port)
    while True:
        print(bd.get_pos())
        time.sleep(1)
