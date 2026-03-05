#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion

from tf_transformations import quaternion_from_euler

import numpy as np  

class EKF(Node):

    # Inicializando nó
    def __init__(self):
        super().__init__('EKF')
        self.get_logger().info('Inicializando o nó!')

        self.create_subscription(Odometry, '/odom', self.subscriber_callback, 10)
        self.publisher_pose = self.create_publisher(Pose, '/pose', 10)

        ########## inicializando variáveis ##########
        self.dT = 1.0                                       # variacao do tempo
        self.x = np.array([[0.0],[0.0],[0.0]])              # vetor do estado
        self.P = np.diag([0.1**2,0.1**2,np.radians(10)**2]) # matriz de covariância da estimativa inicial

        self.z = None                                       # medição em cada passo
        self.u = None                                       # vetor de controle
        self.Q = np.diag([0.1**2,0.1**2,np.radians(10)**2]) # covariância do ruído de processo
        self.R = None                                       # covariância do ruído de medição

        #############################################

        self.timer = self.create_timer(self.dT,self.timer_callback)
        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    # Odom callback
    def subscriber_callback(self, msg):
        self.z = np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y]])
        self.R = np.diag([msg.pose.covariance[0],msg.pose.covariance[7]])
        self.u = np.array([[msg.twist.twist.linear.x],[msg.twist.twist.angular.z]])

    # Executando
    def timer_callback(self):     
        self.x, self.P = self.ekf_algorithm(self.x,self.P,self.u,self.z,self.Q,self.R)

        msg = Pose()
        msg.position = Point(x=self.x[0].item(), y=self.x[1].item(), z=0) 
        qx, qy, qz, qw = quaternion_from_euler(0, 0, self.x[2].item())
        msg.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self.publisher_pose.publish(msg)

    # Atualização
    def ekf_algorithm(self, x, P, u, z, Q, R):
        v = u[0,0]
        w = u[1,0]
        theta = x[2,0]
        dt = self.dT

        x_pred = np.array([[x[0,0] + v*np.cos(theta)*dt],[x[1,0] + v*np.sin(theta)*dt],[x[2,0] + w*dt]])
        F = np.array([[1, 0, -v*np.sin(theta)*dt], [0, 1, v*np.cos(theta)*dt], [0, 0, 1]])
        H = np.array([[1,0,0],[0,1,0]])

        P_pred = F @ P @ F.T + Q

        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @  np.linalg.inv(S)
        x = x_pred + K @ y
        I = np.eye(3)
        P = (I - K @ H) @ P_pred 

        print(x_pred)
        print(x)

        return x.copy(), P.copy()

def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = EKF()          # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
