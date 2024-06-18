#! /bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from threading import Thread, Lock

from collections import deque

import cv2
import time
import numpy as np
import math
import socket
import zmq
import pickle

from config import *
from utils import *

# from scipy.spatial.transform.rotation import Rotation as R


class CameraInfoNode(Node):
    def __init__(self):
        super().__init__("camera_info_node")
        self.info_subscriber = self.create_subscription(
            CameraInfo, "/camera/depth/camera_info", self.info_callback, 10
        )
        self.camera_info = None

    def info_callback(self, msg: CameraInfo):
        self.camera_info = msg


class ImgNode(Node):
    def __init__(self):
        super().__init__("img_node")
        self.rgb_subscription = self.create_subscription(
            Image, "/image_rgb", self.rgb_callback, 10
        )
        self.depth_subscription = self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self.depth_callback, 10
        )
        self.camera_info_node = CameraInfoNode()

        self.bridge = CvBridge()

        self.K = None
        self.rgb_img = None
        self.depth_img = None

    def rgb_callback(self, msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg)

    def depth_callback(self, msg: Image):
        if self.K is None:
            rclpy.spin_once(self.camera_info_node)
        if self.camera_info_node.camera_info is not None:
            # initialized camera info
            self.K = np.array(self.camera_info_node.camera_info.k).reshape(3, 3)
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def get_imgs(self):
        return self.rgb_img, self.depth_img
        try:
            return self.rgb_img.copy(), self.depth_img.copy()
        except:
            return None, None


def get_ball_center_from_rgb(img: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([40, 40, 0])
    high = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    # cv2.imwrite('rgb.jpg',image + 0.5 * mask)
    non_zero_points = cv2.findNonZero(mask)
    in_sight_flag = True
    if non_zero_points is None or non_zero_points.shape[0] <= 5000:
        in_sight_flag = False
        avgx, avgy = 0.0, 0.0
    else:
        avgx = np.mean(non_zero_points[:, 0, 0])
        avgy = np.mean(non_zero_points[:, 0, 1])
    return avgx, avgy, in_sight_flag


def center_coordinates_rgb_to_depth(u, v, K):
    # TODO
    # u,v are the ball central position in rgb image
    # rgb camera info
    fx = 470.0
    fy = 464.0
    cx = 300.0
    cy = 250.0
    # infra1 camera info
    fix = K[0, 0]
    fiy = K[1, 1]
    cix = K[0, 2]
    ciy = K[1, 2]
    # convert the rgb position to infra1 position
    depth_u = int((u - cx) * fix / fx + cix)
    depth_v = int((v - cy) * fiy / fy + ciy)
    alpha = math.atan((u - cx) / fx)
    beta = math.atan((v - cy) / fy)
    theta = math.atan(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
    phi = -math.atan((u - cx) / fx)
    return depth_u, depth_v, alpha, beta


def depth_coordinates_to_local_pos(u, v, alpha, beta, depth_img):
    # TODO
    # get the depth value, convert to mm to m
    depth = depth_img[v, u] * 0.001
    # get the real position of the ball
    # x axis is the forward direction of the dog head
    # y axis is the left direction of the dog head
    # z axis is the up direction of the dog head
    # compute the angle between the ray and the x axis
    x = depth * math.cos(alpha)
    y = -depth * math.sin(alpha)
    z = 0
    return x, y, z


class LocationServer:
    def __init__(
        self,
        upstream_ip=upstream_ip,
        upstream_port=upstream_port,
        server_ip="127.0.0.1",
        server_port=ball_detector_server_port,
        pub_freq=ball_detector_server_freq,
        refresh_freq=ball_detector_server_refresh_freq,
        color=color,
    ):
        self.in_sight_flag = False
        self.mode = "base"  # "base", "camera", "global"

        # ball location in rgb and depth img
        self.rgb_x = 0
        self.rgb_y = 0
        self.depth_x = 0
        self.depth_y = 0
        # 3d ball location in camera frame
        self.camera_ball_x = 0.0
        self.camera_ball_y = 0.0
        self.camera_ball_z = 0.0

        self.global_ball_x_raw = 0.0
        self.global_ball_y_raw = 0.0
        self.red_x_raw = 0.0
        self.red_y_raw = 0.0
        self.black_x_raw = 0.0
        self.black_y_raw = 0.0

        # ball location in global frame
        self.global_ball_x = 2.1
        self.global_ball_y = 3.5

        # dog positions in global frame
        self.red_x = 0.0
        self.red_y = 0.0
        self.black_x = 0.0
        self.black_y = 0.0
        self.color = color
        self.opponent_color = "black" if color == "red" else "red"

        # ball location in base frame
        self.ball_x = 0.0
        self.ball_y = 0.0

        # set up connection with upstream server
        self.upstream_client_socket = socket.socket()
        self.upstream_client_socket.settimeout(1)
        try:
            self.upstream_client_socket.connect((upstream_ip, upstream_port))
            print(f"Connected to upstream machine at {upstream_ip}:{upstream_port}!")
        except socket.timeout:
            print("Connection to upstream server timed out.")
            self.upstream_client_socket = None
        # except OSError:
        #    print("OSError during upstream connection.")
        #    self.upstream_client_socket = None

        # set up connection with robot data server
        self.robot_data_context = zmq.Context()
        self.robot_data_socket = self.robot_data_context.socket(zmq.SUB)
        self.robot_data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.robot_data_socket.setsockopt(zmq.CONFLATE, 1)
        self.robot_data_socket.connect(f"tcp://127.0.0.1:{robot_server_port}")
        print(f"Connected to robot data server at {robot_server_port}!")

        # set up threads to refresh upstream data and robot roll pitch yaw
        self.rpy = np.zeros(3, dtype=np.float32)
        self.upstream_data = None
        self.refresh_data_thread = Thread(target=self.refresh_data)

        # self.yaw_init = get_yaw_init()
        # self.yaw_init = None
        serialized_data = self.robot_data_socket.recv()
        msg, _ = pickle.loads(serialized_data)
        self.yaw_init = msg.rpy[2]

        # set up threads to retrieve images
        self.img_node = ImgNode()
        self.nodes_executor = MultiThreadedExecutor()
        self.nodes_executor.add_node(self.img_node)
        self.spin_thread = Thread(target=self.nodes_executor.spin)
        self.spin_thread.start()

        # set up threads to process retrieved data
        self.refresh_thread = Thread(target=self.refresh)

        self.lock = Lock()
        self.refresh_data_thread.start()
        self.refresh_interval = 1 / refresh_freq
        self.refresh_thread.start()

        self.pub_interval = 1 / pub_freq
        # ZMQ context and socket for client connections
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{server_ip}:{server_port}")
        print(f"Location Server started at {server_ip}:{server_port}")

    def refresh_data(self):
        # upstream: 0.01 - 0.03
        # robot: 0.0002 - 0.0005

        # Initialize deques to store the last 20 values
        global_ball_x_buffer = deque(maxlen=20)
        global_ball_y_buffer = deque(maxlen=20)
        red_x_buffer = deque(maxlen=20)
        red_y_buffer = deque(maxlen=20)
        black_x_buffer = deque(maxlen=20)
        black_y_buffer = deque(maxlen=20)

        global_ball_x_buffer.append(0.0)
        global_ball_y_buffer.append(0.0)
        red_x_buffer.append(0.0)
        red_y_buffer.append(0.0)
        black_x_buffer.append(0.0)
        black_y_buffer.append(0.0)

        while True:
            # with Timer("refresh_data, upstream"):
            if self.upstream_client_socket is not None:
                try:
                    msg = "start"
                    self.upstream_client_socket.send(msg.encode())
                    self.upstream_data = self.upstream_client_socket.recv(256).decode()
                    (
                        global_ball_x,
                        global_ball_y,
                        red_x,
                        red_y,
                        black_x,
                        black_y,
                    ) = [
                        float(data) if data != "None" else None
                        for data in self.upstream_data.split(" ")
                    ]

                    # Append new values to the buffers
                    if global_ball_x is not None:
                        global_ball_x_buffer.append(global_ball_x)
                    if global_ball_y is not None:
                        global_ball_y_buffer.append(global_ball_y)
                    if red_x is not None:
                        red_x_buffer.append(red_x)
                    if red_y is not None:
                        red_y_buffer.append(red_y)
                    if black_x is not None:
                        black_x_buffer.append(black_x)
                    if black_y is not None:
                        black_y_buffer.append(black_y)

                    # Calculate the mean of the buffers
                    self.global_ball_x_raw = sum(global_ball_x_buffer) / len(
                        global_ball_x_buffer
                    )
                    self.global_ball_y_raw = sum(global_ball_y_buffer) / len(
                        global_ball_y_buffer
                    )
                    self.red_x_raw = sum(red_x_buffer) / len(red_x_buffer)
                    self.red_y_raw = sum(red_y_buffer) / len(red_y_buffer)
                    self.black_x_raw = sum(black_x_buffer) / len(black_x_buffer)
                    self.black_y_raw = sum(black_y_buffer) / len(black_y_buffer)

                    # print("upstream data: ", self.upstream_data)
                    self.global_ball_x = self.global_ball_x_raw
                    self.global_ball_y = self.global_ball_y_raw
                    self.red_x = self.red_x_raw
                    self.red_y = self.red_y_raw
                    self.black_x = self.black_x_raw
                    self.black_y = self.black_y_raw

                    self.apply_homo()
                except socket.timeout as e:
                    print("Upstream server timeout.")

            # get robot data
            serialized_data = self.robot_data_socket.recv()
            msg, _ = pickle.loads(serialized_data)
            self.rpy[:] = msg.rpy

    def apply_homo(self):
        xys = np.array(
            [
                [
                    [self.global_ball_x_raw, self.global_ball_y_raw],
                    [self.red_x_raw, self.red_y_raw],
                    [self.black_x_raw, self.black_y_raw],
                ]
            ],
            dtype="float32",
        )
        xys_transformed = cv2.perspectiveTransform(xys, H)[0]
        xys_transformed = xys_transformed.astype(float)
        self.global_ball_x, self.global_ball_y = xys_transformed[0]
        self.red_x, self.red_y = xys_transformed[1]
        self.black_x, self.black_y = xys_transformed[2]

    @property
    def robot_x(self):
        return getattr(self, f"{self.color}_x")

    @property
    def robot_y(self):
        return getattr(self, f"{self.color}_y")

    @property
    def opponent_x(self):
        return getattr(self, f"{self.opponent_color}_x")

    @property
    def opponent_y(self):
        return getattr(self, f"{self.opponent_color}_y")

    def refresh(self):
        # 0.0001 seconds per iter
        while True:
            st = time.perf_counter()
            with self.lock:
                rgb_img, depth_img = self.img_node.get_imgs()

                in_sight_flag = False
                if rgb_img is not None:
                    print("img not none")
                    # cv2.imwrite("rgb.png", rgb_img)
                    # cv2.imwrite("depth.png", depth_img)
                    rgb_x, rgb_y, in_sight_flag = get_ball_center_from_rgb(rgb_img)
                    self.rgb_x, self.rgb_y, self.in_sight_flag = (
                        rgb_x,
                        rgb_y,
                        in_sight_flag,
                    )

                # if in_sight_flag == False:
                if True:
                    # use the coordinates from upstream machine
                    if (
                        self.robot_x is None
                        or self.robot_y is None
                        or self.global_ball_x is None
                        or self.global_ball_y is None
                    ):
                        self.ball_x, self.ball_y = 0.5, 0.2
                        self.mode = "base"
                    else:
                        self.mode = "global"
                else:
                    # use RealSense, return the accurate relative position
                    self.depth_x, self.depth_y, alpha, beta = (
                        center_coordinates_rgb_to_depth(rgb_x, rgb_y, self.img_node.K)
                    )
                    self.camera_ball_x, self.camera_ball_y, self.camera_ball_z = (
                        depth_coordinates_to_local_pos(
                            self.depth_x, self.depth_y, alpha, beta, depth_img
                        )
                    )
                    self.mode = "camera"

                if self.mode == "global":
                    # 1: rpy rotate
                    # ball_xyz = np.array(
                    #     [self.global_ball_x - self.robot_x, self.global_ball_y - self.robot_y, 0.0]
                    # )
                    # rpy = self.rpy.copy()
                    # rpy[2] -= self.yaw_init
                    # # ball_xyz = R.from_euler(rpy, "xyz").inv().apply(ball_xyz)
                    # ball_xyz = inverse_rotate_rpy(ball_xyz, rpy)
                    # self.ball_x, self.ball_y = ball_xyz[:2]
                    # 2: yaw rotate
                    ball_xy = np.array(
                        [
                            self.global_ball_x - self.robot_x,
                            self.global_ball_y - self.robot_y,
                        ]
                    )
                    yaw = self.rpy[2] - self.yaw_init
                    ball_xy = rotate2d(ball_xy, -yaw)
                    self.ball_x, self.ball_y = ball_xy
                    self.mode = "base"
                elif self.mode == "camera":
                    self.ball_x = self.camera_ball_x + 0.272
                    self.ball_y = self.camera_ball_y
                    self.mode = "base"
            end = time.perf_counter()
            sleep_time = max(0, self.refresh_interval - (end - st))
            time.sleep(sleep_time)

    def run(self):
        try:
            while True:
                st = time.perf_counter()
                with self.lock:
                    data = {
                        "global_ball_x": self.global_ball_x,
                        "global_ball_y": self.global_ball_y,
                        "ball_x": self.ball_x,
                        "ball_y": self.ball_y,
                        "robot_x": self.robot_x,
                        "robot_y": self.robot_y,
                        "opponent_x": self.opponent_x,
                        "opponent_y": self.opponent_y,
                        "mode": self.mode,
                        "insight_flag": self.in_sight_flag,
                    }
                    self.socket.send_json(data)
                end = time.perf_counter()
                sleep_time = max(0, self.pub_interval - (end - st))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Server is shutting down...")
        finally:
            self.socket.close()
            self.context.term()
            print("Server closed")


if __name__ == "__main__":
    rclpy.init()
    server = LocationServer(server_ip="0.0.0.0", server_port=ball_detector_server_port)
    server.run()
