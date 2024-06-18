import lcm
from robot_data_types import motor_ctrl_state_lcmt, spi_data_t, leg_control_data_lcmt, state_estimator_lcmt, robot_data_lcmt
import zmq
import pickle
import numpy as np
import time
from config import robot_server_port
from utils import Timer

def get_lcm_url_port(port, ttl):
    assert 0 <= ttl <= 255, "TTL must be between 0 and 255"
    return f"udpm://239.255.76.67:{port}?ttl={ttl}"


class LCMHandler:
    def __init__(self, robot_server_port, pub_freq=100):
        # self.q = np.zeros(12, np.float32)
        # self.rpy = np.zeros(3, np.float32)
        self.data = robot_data_lcmt()
        lcm_url_7667 = get_lcm_url_port(7667, 255)
        lcm_url_7669 = get_lcm_url_port(7669, 255)
        
        self.motor_ctrl_state_lcm = lcm.LCM(lcm_url_7667)
        self.motor_data_lcm = lcm.LCM(lcm_url_7667)
        self.leg_data_lcm = lcm.LCM(lcm_url_7667)
        self.robot_state_lcm = lcm.LCM(lcm_url_7669)

        self.motor_ctrl_state_lcm.subscribe("motor_ctrl_state", self.handle_motor_ctrl_state)
        self.motor_data_lcm.subscribe("spi_data", self.handle_spi_data)
        self.leg_data_lcm.subscribe("leg_control_data", self.handle_leg_control_data)
        self.robot_state_lcm.subscribe("state_estimator", self.handle_state_estimator)
        
        self.init_yaw = None
        self.iter = 0
        # setup publisher zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://0.0.0.0:{robot_server_port}")
        print(f"Robot data server started on port {robot_server_port}")
        
        self.pub_interval = 1 / pub_freq
        
        
    def handle_motor_ctrl_state(self, channel, data):
        # print("motor_ctrl_state")
        pass

    def handle_spi_data(self, channel, data):
        # print("spi_data")
        pass
    
    def handle_leg_control_data(self, channel, data):
        # print("leg_control_data")
        msg = leg_control_data_lcmt.decode(data)
        self.data.q = msg.q
    
    def handle_state_estimator(self, channel, data):
        # print("state_estimator")
        msg = state_estimator_lcmt.decode(data)
        if self.init_yaw is None:
            self.iter += 1
            if self.iter >= 200:
                self.init_yaw = msg.rpy[2]
            else:
                return
        rpy = list(msg.rpy)
        rpy[2] -= self.init_yaw
        print(self.init_yaw, rpy)
        self.data.rpy = rpy

    def run(self):
        try:
            while True:
                # 0.01s
                with Timer("robot_data_server"):
                    st = time.time()
                    # self.motor_ctrl_state_lcm.handle()
                    # self.motor_data_lcm.handle()
                    self.leg_data_lcm.handle()
                    self.robot_state_lcm.handle()
                    # pub mesg
                    serialized_data = pickle.dumps((self.data, time.time()))
                    self.socket.send(serialized_data)
                    end = time.time()
                    sleep_time = max(0, self.pub_interval - (end - st))
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Robot data server stopped!")
            pass

if __name__ == "__main__":
    handler = LCMHandler(robot_server_port)
    handler.run()
