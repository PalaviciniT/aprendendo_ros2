#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion

import tf_transformations

import numpy as np
import math

class MCL(Node):
    def __init__(self):
        super().__init__('mcl')
        self.get_logger().info('Inicializando o nó!')

        qos_profile_map = QoSProfile(depth=10)
        qos_profile_map.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile_map.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile_scan_odom = QoSProfile(depth=10)
        qos_profile_scan_odom.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile_map)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_scan_odom)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile_scan_odom)
    
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/mcl_pose', 10)
        self.pub_particles = self.create_publisher(PoseArray, '/particlecloud', 10)

        ########## inicializando variáveis ##########
        self.dT = 0.2                                               # período do timer
        self.M = 1000                                               # nº partículas
        self.p = np.zeros((self.M, 3), dtype=float)                 # [x, y, th]
        self.w = np.ones(self.M, dtype=float) / self.M              # pesos
        self.odom = None
        self.scan = None
        self.map = None  
        self.mcl_pose = PoseWithCovarianceStamped()
        #############################################

        self.timer = self.create_timer(self.dT,self.timer_callback)
        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    def odom_callback(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        self.odom = (pos_x, pos_y, yaw)

    def laser_callback(self, msg):
        self.laser = msg

    def map_callback(self, msg):
        self.map = msg

        self.inicializacao()

    # Executando
    def timer_callback(self):
        if self.odom is None or self.laser == None or self.map == None:
            return       
        
        self.mcl_algorithm()
        self.publicar_pose()
        self.publicar_particulas()

    def publicar_pose(self):
        msg = self.mcl_pose 
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'     
        self.pub_pose.publish(msg)
        
    def publicar_particulas(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        for i in range(self.M):
            x, y, th = self.p[i]
            q = tf_transformations.quaternion_from_euler(0.0, 0.0, th)
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            pa.poses.append(pose)
        self.pub_particles.publish(pa)

    # algorithm - inicialização
    def inicializacao(self):
        width = self.map.info.width
        height = self.map.info.height
        resolution = self.map.info.resolution
        origin = self.map.info.origin

        free_cells = []

        # encontra células livres (valor 0)
        for i in range(width * height):
            if self.map.data[i] == 0:
                free_cells.append(i)

        # amostra partículas
        for i in range(self.M):
            idx = np.random.choice(free_cells)

            y = idx // width
            x = idx % width

            world_x = origin.position.x + x * resolution
            world_y = origin.position.y + y * resolution
            theta = np.random.uniform(-math.pi, math.pi)

            self.p[i] = [world_x, world_y, theta]

        self.w.fill(1.0 / self.M)

    # algorithm - atualização
    def mcl_algorithm(self):
        # Previsão
        if self.odom is None:
            return

        x_odom, y_odom, th_odom = self.odom

        noise = [0.02, 0.02, 0.01]

        for i in range(self.M):
            dx = np.random.normal(0, noise[0])
            dy = np.random.normal(0, noise[1])
            dth = np.random.normal(0, noise[2])

            self.p[i][0] += dx
            self.p[i][1] += dy
            self.p[i][2] += dth
        
        # Correção
        if self.laser is None:
            return

        ranges = np.array(self.laser.ranges)
        valid = np.isfinite(ranges)

        z = ranges[valid]

        for i in range(self.M):
            # modelo extremamente simples (placeholder)
            expected = np.mean(z)  # simplificação

            error = np.abs(z - expected)
            self.w[i] = np.exp(-np.sum(error) * 0.1)

        # normalizar pesos
        self.w += 1e-300
        self.w /= np.sum(self.w)
        
        # Reamostragem
        indices = np.random.choice(self.M, self.M, p=self.w)

        self.p = self.p[indices]
        self.w.fill(1.0 / self.M)

        # Estimativa da posição
        mean = np.average(self.p, weights=self.w, axis=0)

        self.mcl_pose.pose.pose.position.x = float(mean[0])
        self.mcl_pose.pose.pose.position.y = float(mean[1])

        q = tf_transformations.quaternion_from_euler(0, 0, mean[2])

        self.mcl_pose.pose.pose.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
        )
        


def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = MCL()          # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
