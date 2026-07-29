
# FERNANDO LÓPEZ VERDEJO - TFG

import rclpy
from rclpy.node import Node
# QoS (Quality of Service) define las reglas de comunicación en ROS 2. Vital para los sensores.
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# Mensajes estándar de ROS 2 para mover el robot (Twist) y leer posiciones (Pose, Point, Quaternion)
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan # Para recibir los datos del LiDAR
from std_srvs.srv import Empty # Para llamar al servicio de resetear el simulador
from gazebo_msgs.srv import SetEntityState # Para teletransportar al robot en el simulador
import gymnasium as gym # Librería estándar para crear entornos de Inteligencia Artificial
from gymnasium import spaces
import numpy as np
# PPO (Proximal Policy Optimization) es el algoritmo matemático de la IA que vamos a usar
from stable_baselines3 import PPO 
# Monitor permite guardar los datos del entrenamiento en un CSV para luego hacer gráficas
from stable_baselines3.common.monitor import Monitor 
import math
import time
import random
import os

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convierte ángulos de Euler (rotación típica en X, Y, Z) a Cuaterniones (4D).
    Explicación para la memoria: ROS 2 y Gazebo utilizan cuaterniones internamente
    para calcular la orientación del robot y evitar el problema matemático del 
    "Gimbal Lock" (pérdida de grados de libertad).
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

# ==============================================================================
# ENTORNO DE APRENDIZAJE (Custom Gym Environment)
# ==============================================================================
class TurtleBotEnv(gym.Env):
    """
    Esta clase es el "traductor" entre la Inteligencia Artificial y el robot simulado.
    Hereda de 'gym.Env', lo que obliga a tener funciones estandarizadas (step y reset)
    que la librería Stable-Baselines3 (PPO) pueda entender.
    """
    def __init__(self):
        super(TurtleBotEnv, self).__init__()
        
        # 1. INICIALIZACIÓN DE ROS 2
        # Levantamos el nodo que se comunicará con el resto del ecosistema ROS
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('ppo_tfg_env')
        
        # 2. CONFIGURACIÓN DE COMUNICACIONES
        # Configuramos el QoS (Quality of Service) en "BEST_EFFORT".
        # Explicación: Los sensores como el LiDAR envían muchísimos datos por segundo. 
        # Si la red está saturada, es mejor perder un paquete viejo (Best Effort) 
        # que acumular retraso intentando garantizar su entrega.
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publicador: Envía comandos de velocidad al robot.
        self.pub_cmd_vel = self.node.create_publisher(Twist, '/cmd_vel', 10)
        # Suscriptor: Escucha lo que "ve" el robot a través del láser.
        self.sub_scan = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos_policy)
        # Clientes de servicios: Permiten reiniciar Gazebo y mover al robot al instante.
        self.client_reset = self.node.create_client(Empty, '/reset_simulation')
        self.client_set_state = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')

        # 3. DEFINICIÓN DE LA INTERFAZ DE LA IA (ESPACIOS)
        # Action Space (Salidas de la Red Neuronal):
        # La IA decide 2 valores: Velocidad Lineal (avance) y Velocidad Angular (giro).
        # Los limitamos a la capacidad física del TurtleBot3 Burger (0.22 m/s máximo).
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([0.22, 1.0]), dtype=np.float32)
        
        # Observation Space (Entradas a la Red Neuronal):
        # El LiDAR devuelve 360 distancias (una por grado). Es demasiada información pura
        # para que la red neuronal aprenda rápido. Lo agrupamos en 24 sectores.
        self.n_sectores = 24
        # Cada sector mide entre 0.0 metros (choque) y 3.5 metros (alcance máximo del láser).
        self.observation_space = spaces.Box(low=0.0, high=3.5, shape=(self.n_sectores,), dtype=np.float32)
        
        self.latest_scan = np.zeros(self.n_sectores)
        self.scan_ready = False

        # Zonas de spawn para el entrenamiento avanzado 
        # Actualmente desactivadas en el código base para evitar inestabilidades de Gazebo.
        self.spawn_points = [
            {'x': 8.70,  'y': -8.60},
            {'x': -8.95, 'y': 9.11}, 
            {'x': -3.02, 'y': -8.97},
            {'x': 8.53,  'y': 8.97},  
        ]

    def scan_callback(self, msg):
        """
        Función que se ejecuta automáticamente cada vez que llega un mensaje del LiDAR.
        Filtra datos erróneos de Gazebo (Infinitos o NaN) y reduce los 360 grados a 24 sectores.
        """
        ranges = msg.ranges
        # Filtro de limpieza: Si el rayo láser no choca con nada o da error, le asignamos 3.5m (distancia segura)
        ranges = [r if not math.isinf(r) and not math.isnan(r) else 3.5 for r in ranges]
        
        count = len(ranges)
        step = int(count / self.n_sectores)
        new_scan = []
        
        # Agrupación: De cada bloque de rayos (ej. 15 grados), nos quedamos con el valor MÍNIMO,
        # ya que es el obstáculo más cercano y el que representa el mayor peligro.
        for i in range(0, count, step):
            sector = ranges[i:i+step]
            new_scan.append(min(sector) if len(sector) > 0 else 3.5)
            
        self.latest_scan = np.array(new_scan[:self.n_sectores], dtype=np.float32)
        self.scan_ready = True # Bandera para avisar de que hay datos frescos listos

    def step(self, action):
        """
        El núcleo del Aprendizaje por Refuerzo. 
        Este método ejecuta una acción elegida por la IA, avanza el simulador, y devuelve 
        los resultados: (Nuevo estado, Recompensa obtenida, Si el episodio terminó, Info extra).
        """
        # 1. Aplicar la acción (mandar velocidad al robot)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.pub_cmd_vel.publish(twist)
        
        # 2. Sincronización empírica: Damos un breve tiempo para que Gazebo actualice 
        # las físicas y el LiDAR antes de leer el nuevo estado.
        time.sleep(0.05)
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        distancia_minima = np.min(self.latest_scan)
        
        # 3. SISTEMA DE RECOMPENSAS (REWARD SHAPING)
        # Así es como "educamos" a la IA: mediante puntos positivos o negativos.
        
        if distancia_minima < 0.15: 
            # CONDICIÓN DE CHOQUE: Si está a menos de 15 cm de una pared, se considera colisión.
            # Castigo muy fuerte y terminamos el intento.
            reward = -100.0
            done = True
        else:
            # COMPORTAMIENTO DESEADO:
            # - (twist.linear.x * 2.0): Premiamos ir hacia adelante lo más rápido posible.
            # - (abs(twist.angular.z) * 0.5): Restamos puntos si gira sin necesidad. 
            #   Esto obliga a la IA a buscar trayectorias rectas y suaves.
            reward = (twist.linear.x * 2.0) - (abs(twist.angular.z) * 0.5)
            
            # RECOMPENSA DE AJUSTE (BONUS):
            # Si el robot va rápido en un pasillo (cerca de paredes pero sin chocar), le damos un extra.
            # Sin este bonus, la IA suele coger "miedo", frenando a 0 m/s cuando ve pasillos estrechos.
            if twist.linear.x > 0.15 and distancia_minima > 0.3:
                reward += 0.2
            done = False

        return self.latest_scan, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """
        Devuelve el robot y el simulador al estado inicial tras un choque o al empezar.
        Contiene medidas de seguridad porque los simuladores como Gazebo a veces 
        se desincronizan o tardan en responder a la petición de reset.
        """
        super().reset(seed=seed)
        
        # Frenar al robot enviando velocidades a cero.
        # Si no hacemos esto, el robot reaparece manteniendo la inercia del choque anterior.
        self.pub_cmd_vel.publish(Twist())
        
        # Petición de reinicio de físicas a Gazebo, con control de tiempo máximo (timeout)
        if not self.client_reset.wait_for_service(timeout_sec=2.0):
            print("WARNING: El servicio de reset del simulador no responde.")

        req = Empty.Request()
        future = self.client_reset.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        
        # Espera para que Gazebo procese el reposicionamiento y el láser no lea "fantasmas"
        time.sleep(0.5) 
        
        # Bucle de seguridad para garantizar que la primera lectura del nuevo episodio sea limpia
        self.scan_ready = False
        intentos = 0
        while not self.scan_ready:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            intentos += 1
            if intentos > 20: # Evita que el programa se quede congelado infinitamente
                break
            
        # Parche extra: Si Gazebo "suelta" al robot demasiado cerca de una pared por error,
        # esperamos un instante más para confirmar la lectura.
        if np.min(self.latest_scan) < 0.20:
             time.sleep(0.2)
             rclpy.spin_once(self.node, timeout_sec=0.1)

        return self.latest_scan, {}

# ==============================================================================
# BUCLE PRINCIPAL (Ejecución del Entrenamiento)
# ==============================================================================
if __name__ == '__main__':
    env = None
    try:
        print("Inicializando entorno y loggers del TFG...")
        
        # Directorio donde se guardarán los datos para TensorBoard (Gráficas)
        log_dir = "./logs_tfg/"
        os.makedirs(log_dir, exist_ok=True)
        
        env = TurtleBotEnv()
        # Envolvemos el entorno en un Monitor. Esto es estrictamente necesario para que 
        # Stable-Baselines3 guarde los registros de las recompensas ganadas por episodio.
        env = Monitor(env, log_dir) 

        # Rutas de guardado de los modelos (el "cerebro" de la IA)
        modelo_base = "modelos/ppo_fase1"          
        modelo_destino = "modelos/ppo_finetuned"   
        modelo_rescate = "modelos/ppo_backup"      
        
        model = None
        
        # LÓGICA DE CARGA DE PESOS DE LA IA:
        # En lugar de empezar a aprender desde cero cada vez que se ejecuta el código,
        # intentamos cargar el conocimiento previo almacenado en archivos .zip
        if os.path.exists(modelo_rescate + ".zip"):
            print("Cargando modelo de backup (se detectó un corte de ejecución anterior)...")
            model = PPO.load(modelo_rescate, env=env, tensorboard_log=log_dir)
            
        elif os.path.exists(modelo_base + ".zip"):
            print(f"Cargando {modelo_base} para continuar refinando el entrenamiento...")
            model = PPO.load(modelo_base, env=env, tensorboard_log=log_dir)
            
        else:
            # Si no hay archivos, creamos una red neuronal vacía ("MlpPolicy") y empezamos desde cero.
            print("No se encontró ningún modelo previo. Se inicia entrenamiento desde cero absoluto.")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

        # Configuramos los pasos de entrenamiento. 50.000 pasos es un ciclo de prueba estándar.
        pasos_totales = 50000
        print(f"Lanzando {pasos_totales} timesteps de interacción con el entorno...")
        
        # .learn() es el método donde la IA juega miles de veces y ajusta su red neuronal matemáticamente
        model.learn(total_timesteps=pasos_totales, tb_log_name="tfg_run_final")
        
        # Guardado final tras completar los pasos
        model.save(modelo_destino)
        print("Entrenamiento completado satisfactoriamente y modelo exportado.")
        
    except KeyboardInterrupt:
        # Si el estudiante pulsa Ctrl+C en la terminal para parar el proceso, 
        # se intercepta el comando para no perder el progreso y guardar un archivo de rescate.
        print("\nEntrenamiento interrumpido manualmente. Guardando estado de seguridad...")
        if 'model' in locals() and model is not None:
            model.save(modelo_rescate)
            
    except Exception as e:
        print(f"\nExcepción en tiempo de ejecución: {e}")
        
    finally:
        # Bloque de limpieza: Se ejecuta siempre al salir, haya error o no.
        # Asegura que el robot reciba una orden de parada y que los hilos de ROS se cierren limpiamente.
        if rclpy.ok():
            node = rclpy.create_node('shutdown_hook')
            pub = node.create_publisher(Twist, '/cmd_vel', 10)
            pub.publish(Twist()) # Parada total del hardware/simulador
            rclpy.shutdown()