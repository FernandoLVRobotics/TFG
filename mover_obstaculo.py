import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class PatrullaObstaculo(Node):
    def __init__(self):
        super().__init__('nodo_patrulla_obstaculo')
        self.publisher_ = self.create_publisher(Twist, '/obstaculo/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = time.time()
        self.velocidad = 0.5  # 0.5 m/s

    def timer_callback(self):
        msg = Twist()
        tiempo_actual = time.time()
        
        # Trayectoria: 8 metros de distancia a 0.5 m/s = 16 segundos exactos.
        if (tiempo_actual - self.start_time) > 16.0:
            self.velocidad *= -1.0
            self.start_time = tiempo_actual
            self.get_logger().info('🔄 Límite alcanzado. Cambiando de sentido...')

        # Forzamos movimiento puro en el eje Y
        msg.linear.x = 0.0
        msg.linear.y = self.velocidad 
        msg.linear.z = 0.0
        
        # Bloqueamos rotaciones
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    nodo = PatrullaObstaculo()
    print("🚶‍♂️ Obstáculo físico activado. Patrullando de Y=-4 a Y=4...")
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()