
# FERNANDO LÓPEZ VERDEJO - TFG

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import math
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox, font

# ==============================================================================
# MATEMÁTICAS DE NAVEGACIÓN
# ==============================================================================
def euler_from_quaternion(x, y, z, w):
    """
    Transforma la orientación del robot de Cuaterniones (formato 3D de ROS) a 
    ángulos de Euler (Yaw, pitch, roll). 
    Para la navegación 2D solo nos interesa el Yaw (hacia dónde apunta el morro en radianes).
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

# ==============================================================================
# ENTORNO DE EJECUCIÓN E INFERENCIA (ROS 2 + LÓGICA)
# ==============================================================================
class TurtleBotEnv(gym.Env):
    def __init__(self, update_callback=None):
        super(TurtleBotEnv, self).__init__()
        
        if not rclpy.ok():
            rclpy.init()
        # Nodo principal que orquesta todo el sistema durante la demostración
        self.node = rclpy.create_node('ppo_navegador_master')
        
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # --- COMUNICACIONES DEL ROBOT ---
        self.pub_cmd_vel = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_scan = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos_policy)
        # Novedad respecto al entreno: Ahora leemos la Odometría para saber dónde estamos en el mapa
        self.sub_odom = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.client_reset = self.node.create_client(Empty, '/reset_simulation')

        # --- SIMULACIÓN DE ENTORNO DINÁMICO ---
        # Los "actors" nativos de Gazebo para simular humanos
        # a veces fallan al colisionar con los sensores láser. La solución de ingeniería
        # ha sido mover un obstáculo dinámico controlando su velocidad directamente.
        self.pub_obstaculo = self.node.create_publisher(Twist, '/obstaculo/cmd_vel', 10)
        self.obstaculo_start_time = time.time()
        
        # Inicia bajando en el eje Y del simulador
        self.obstaculo_velocidad = -0.4  
        self.timer_obstaculo = self.node.create_timer(0.1, self.mover_obstaculo_callback)

        # Límites físicos ampliados para la fase de producción/demo
        self.action_space = spaces.Box(low=np.array([0.0, -1.5]), high=np.array([0.50, 1.5]), dtype=np.float32)
        self.n_sectores = 24
        self.observation_space = spaces.Box(low=0.0, high=3.5, shape=(self.n_sectores,), dtype=np.float32)
        
        self.latest_scan = np.zeros(self.n_sectores)
        self.scan_ready = False
        
        # Variables de estado del robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.contador_atasco = 0  
        self.bateria_simulada = 100.0 
        
        self.update_callback = update_callback
        
        # --- WAYPOINTS DE LA RUTAS ---
        # Diseño de ingeniería: En lugar de mandar al robot en línea recta (lo que causaba
        # choques con las paredes si el destino estaba tras una esquina), se han definido
        # nodos de aproximación ("pasillos") para guiar al robot de forma segura.
        self.INICIO = {'x': 2.0, 'y': 2.0}
        self.LUGARES = {
            "DESTINO_1": {'destino': {'x': -8.95, 'y': 9.11}, 'pasillo': {'x': -5.86, 'y': 8.75}},
            "DESTINO_2": {'destino': {'x': 8.70, 'y': -8.60}, 'pasillo': {'x': 8.75, 'y': -6.24}}
        }

        self.pedidos_pendientes = []
        self.objetivos = []
        self.indice_obj = 0
        self.mision_completada = False

    def mover_obstaculo_callback(self):
        """ Bucle que hace que el obstáculo patrulle de lado a lado indefinidamente. """
        msg = Twist()
        tiempo_actual = time.time()
        
        # Matemáticas simples: 8 metros a 0.5 m/s son 16 segundos. Invertimos sentido.
        if (tiempo_actual - self.obstaculo_start_time) > 16.0:
            self.obstaculo_velocidad *= -1.0
            self.obstaculo_start_time = tiempo_actual

        msg.linear.y = self.obstaculo_velocidad 
        # Bloqueamos giros para que el modelo 3D en Gazebo no vuelque por inercias
        msg.angular.x = msg.angular.y = msg.angular.z = 0.0

        self.pub_obstaculo.publish(msg)

    def set_pedidos(self, lista_pedidos):
        self.pedidos_pendientes = lista_pedidos.copy()
        self.objetivos = self.generar_ruta_dinamica(self.pedidos_pendientes)
        self.indice_obj = 0
        self.mision_completada = False
        self.bateria_simulada = 100.0

    def generar_ruta_dinamica(self, lista_pedidos):
        # Generador de máquina de estados de navegación: Va al pasillo -> Entrega -> Vuelve al pasillo -> Base
        ruta = []
        for pedido in lista_pedidos:
            datos = self.LUGARES[pedido]
            ruta.append({'coords': datos['pasillo'], 'msg': f'Aproximacion a {pedido}', 'tipo': 'TRANSITO'})
            ruta.append({'coords': datos['destino'], 'msg': f'Entregando en {pedido}', 'tipo': 'ENTREGA'})
            ruta.append({'coords': datos['pasillo'], 'msg': f'Saliendo de {pedido}', 'tipo': 'TRANSITO'})
        ruta.append({'coords': self.INICIO, 'msg': 'Retornando a Base', 'tipo': 'FINAL'})
        return ruta

    def enviar_datos_gui(self, estado, color_hex, mensaje, distancia):
        if self.update_callback:
            self.update_callback(estado, color_hex, mensaje, distancia, self.bateria_simulada)

    def odom_callback(self, msg):
        # Actualiza la posición global usando la odometría (encoders de las ruedas)
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        rot = msg.pose.pose.orientation
        self.robot_yaw = euler_from_quaternion(rot.x, rot.y, rot.z, rot.w)

    def scan_callback(self, msg):
        ranges = msg.ranges
        ranges = [r if not math.isinf(r) and not math.isnan(r) else 3.5 for r in ranges]
        step = int(len(ranges) / self.n_sectores)
        new_scan = []
        for i in range(0, len(ranges), step):
            sector = ranges[i:i+step]
            new_scan.append(min(sector) if len(sector) > 0 else 3.5)
        self.latest_scan = np.array(new_scan[:self.n_sectores], dtype=np.float32)
        self.scan_ready = True

    def get_nav_logic(self):
        """
        Calcula el vector de dirección entre la posición actual y la meta elegida.
        """
        if self.indice_obj >= len(self.objetivos):
            self.mision_completada = True
            return 0.0, 0.0, ""

        obj_actual = self.objetivos[self.indice_obj]
        target = obj_actual['coords']
        mensaje = obj_actual['msg']
        
        # Distancia euclídea
        dx = target['x'] - self.robot_x
        dy = target['y'] - self.robot_y
        distancia = math.sqrt(dx**2 + dy**2)
        
        # Corrección de ángulo (normalizado entre -PI y +PI para no dar giros absurdos de 360 grados)
        angulo_deseado = math.atan2(dy, dx)
        error_angulo = angulo_deseado - self.robot_yaw
        while error_angulo > math.pi: error_angulo -= 2 * math.pi
        while error_angulo < -math.pi: error_angulo += 2 * math.pi
        
        # Tolerancia de llegada (0.65m para no buscar la perfección milimétrica y quedarse atascado oscilando)
        if distancia < 0.65: 
            self.pub_cmd_vel.publish(Twist()) # Parada en seco
            print(f"OBJETIVO ALCANZADO: {mensaje}")
            self.indice_obj += 1
            # Simula el tiempo que tardaría un cliente en recoger el paquete
            time.sleep(1.0 if obj_actual['tipo'] == 'TRANSITO' else 2.5)
            return 0.0, 0.0, ""
            
        return distancia, error_angulo, mensaje

    def step(self, action):
        """
        Control Jerárquico Híbrido (Basado en la arquitectura de Subsunción).
        La IA propone un movimiento, pero un sistema de reglas clásicas tiene la autoridad final
        dependiendo de lo crítico que sea el entorno (ej: un muro a 20 cm).
        """
        if self.mision_completada:
            self.enviar_datos_gui("SISTEMA EN REPOSO", "#A6ADC8", "Esperando órdenes", 0.0)
            return self.latest_scan, 0.0, True, False, {}

        self.bateria_simulada -= 0.006 
        distancia_minima = np.min(self.latest_scan)
        twist = Twist()
        
        # Lo que la Red Neuronal (IA) dice que deberíamos hacer
        vel_ia = float(action[0])
        giro_ia = float(action[1])
        
        dist_meta, error_angulo, msg_actual = self.get_nav_logic()
        if dist_meta == 0.0: return self.latest_scan, 0.0, False, False, {}

        # --- 1. WATCHDOG (Perro Guardián Antibloqueos) ---
        # Las IA a veces se quedan "congeladas" dudando frente a un obstáculo simétrico.
        # Si el robot pasa mucho tiempo (65 iteraciones) muy cerca de una pared, forzamos un rescate manual.
        if distancia_minima < 0.25: self.contador_atasco += 1
        else: self.contador_atasco = 0

        if self.contador_atasco > 65: 
            self.enviar_datos_gui("MANIOBRA RESCATE", "#F38BA8", "Saliendo de bloqueo", dist_meta)
            twist.linear.x = -0.12 # Marcha atrás
            twist.angular.z = 1.4  # Giro agresivo
            self.pub_cmd_vel.publish(twist)
            time.sleep(0.6)
            self.contador_atasco = 0
            return self.latest_scan, 0.0, False, False, {}

        # --- 2. CAPAS DE COMPORTAMIENTO ---
        
        # CAPA A: EMERGENCIA VITAL (< 0.25m). Evitar choque a toda costa. La IA se ignora casi por completo.
        if distancia_minima < 0.25:
            twist.linear.x = 0.20
            twist.angular.z = giro_ia * 2.0 
            self.enviar_datos_gui("FRENADO EMERGENCIA", "#F38BA8", msg_actual, dist_meta)

        # CAPA B: EVASIÓN FINA (0.25m - 0.45m). Confiamos en la IA para esquivar, pero limitamos la velocidad.
        elif distancia_minima < 0.45:
            twist.linear.x = 0.10  
            twist.angular.z = giro_ia * 1.5
            self.enviar_datos_gui("EVADIENDO OBSTACULO", "#F9E2AF", msg_actual, dist_meta)

        # CAPA C: NAVEGACIÓN PURA (> 0.75m). Camino despejado. Usamos control matemático puro hacia la meta.
        elif distancia_minima > 0.75:
            if abs(error_angulo) > 0.6: 
                # Rotar sobre el propio eje antes de avanzar si está muy desviado
                twist.linear.x = 0.0
                twist.angular.z = 0.9 if error_angulo > 0 else -0.9
                self.enviar_datos_gui("REORIENTANDO", "#89B4FA", "Alineando con meta", dist_meta)
            else: 
                # Acelerador a fondo corrigiendo el rumbo con un control Proporcional
                twist.linear.x = 0.45 
                twist.angular.z = error_angulo * 2.0
                twist.angular.z = max(min(twist.angular.z, 1.2), -1.2) # Saturación de control
                self.enviar_datos_gui("VEL. DE CRUCERO", "#A6E3A1", msg_actual, dist_meta)

        # CAPA D: TRANSICIÓN. Mezcla entre seguir a la IA y apuntar hacia la meta.
        else:
            twist.linear.x = 0.20
            twist.angular.z = (giro_ia * 0.5) + (error_angulo * 0.5)
            self.enviar_datos_gui("APROXIMACION FINA", "#89B4FA", msg_actual, dist_meta)

        self.pub_cmd_vel.publish(twist)
        time.sleep(0.05)
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # Detector de colisión letal (Termina episodio temporalmente)
        if distancia_minima < 0.12:
            self.enviar_datos_gui("COLISIÓN", "#F38BA8", "Reiniciando...", dist_meta)
            return self.latest_scan, 0.0, True, False, {}
            
        return self.latest_scan, 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        self.pub_cmd_vel.publish(Twist())
        req = Empty.Request()
        self.client_reset.call_async(req)
        self.scan_ready = False
        while not self.scan_ready: rclpy.spin_once(self.node, timeout_sec=0.1)
        self.indice_obj = 0 
        self.mision_completada = False
        # IMPORTANTE: No reseteamos el obstáculo aquí, para que no se "teletransporte" y la simulación sea realista
        return self.latest_scan, {}

# ==============================================================================
# INTERFAZ GRÁFICA (Tkinter)
# ==============================================================================
class FleetManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interfaz TFG Fernando López Verdejo")
        self.root.geometry("820x650") 
        self.root.configure(bg="#1E1E2E") 
        self.root.resizable(False, False)
        
        self.pedidos = []
        self.mision_activa = False
        
        # Fuentes y estética
        self.font_title = font.Font(family="Segoe UI", size=18, weight="bold")
        self.font_subtitle = font.Font(family="Segoe UI", size=11, weight="bold")
        self.font_btn = font.Font(family="Segoe UI", size=10, weight="bold")
        self.font_list = font.Font(family="Consolas", size=10)
        self.font_telemetry = font.Font(family="Consolas", size=11, weight="bold")
        
        # Enlazamos el entorno ROS con la interfaz gráfica pasando el callback de actualización
        self.env = TurtleBotEnv(update_callback=self.update_telemetry_labels)
        self.model = self.cargar_modelo_ia()
        self.setup_ui()

    def cargar_modelo_ia(self):
        # Intentamos cargar el modelo pulido. Si no está, caemos al modelo base.
        path = "mi_modelo_ppo_finetuned"
        if not os.path.exists(path + ".zip"): path = "mi_modelo_ppo_ajustado"
        return PPO.load(path) if os.path.exists(path + ".zip") else None
        
    def setup_ui(self):
        # [Código estándar de Tkinter para crear la ventana visual, frames y botones]
        header = tk.Frame(self.root, bg="#181825", height=70)
        header.pack(fill=tk.X)
        tk.Label(header, text="GESTIÓN DE REPARTO DE PAQUETES", font=self.font_title, bg="#181825", fg="#89B4FA").pack(pady=15)

        body = tk.Frame(self.root, bg="#1E1E2E")
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left = tk.Frame(body, bg="#313244", padx=15, pady=15)
        left.pack(side=tk.LEFT, fill=tk.Y, ipadx=10)
        
        tk.Label(left, text="ASIGNACIÓN DE DESTINOS", font=self.font_subtitle, bg="#313244", fg="#CDD6F4").pack(fill=tk.X, pady=(0, 15))
        self.btn_p1 = tk.Button(left, text=" Añadir destino 1", font=self.font_btn, bg="#89B4FA", command=lambda: self.add_pedido("DESTINO_1"))
        self.btn_p1.pack(fill=tk.X, pady=8, ipady=5)
        self.btn_p2 = tk.Button(left, text=" Añadir destino 2", font=self.font_btn, bg="#CBA6F7", command=lambda: self.add_pedido("DESTINO_2"))
        self.btn_p2.pack(fill=tk.X, pady=8, ipady=5)
        tk.Frame(left, bg="#45475A", height=2).pack(fill=tk.X, pady=15)
        tk.Button(left, text=" Limpiar Ruta", font=self.font_btn, bg="#F38BA8", command=self.limpiar_pedidos).pack(fill=tk.X, pady=8, ipady=5)
        self.btn_start = tk.Button(left, text=" INICIAR MISIÓN", font=self.font_btn, bg="#A6E3A1", command=self.iniciar_mision)
        self.btn_start.pack(fill=tk.X, side=tk.BOTTOM, pady=10, ipady=15)

        right = tk.Frame(body, bg="#1E1E2E")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        tk.Label(right, text="ELECCIÓN DE LA RUTA :", font=self.font_subtitle, bg="#1E1E2E", fg="#CDD6F4").pack(fill=tk.X, pady=(0, 5))
        list_f = tk.Frame(right, bg="#11111B", bd=2)
        list_f.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.lista_box = tk.Listbox(list_f, font=self.font_list, bg="#11111B", fg="#A6E3A1", bd=0, height=10)
        self.lista_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.actualizar_lista()

        tk.Label(right, text="TELEMETRÍA EN VIVO:", font=self.font_subtitle, bg="#1E1E2E", fg="#CDD6F4").pack(fill=tk.X, pady=(0, 5))
        tele_f = tk.Frame(right, bg="#181825", padx=20, pady=20)
        tele_f.pack(fill=tk.BOTH, expand=True)

        self.lbl_estado = tk.Label(tele_f, text="[ STANDBY ]", font=self.font_telemetry, bg="#181825", fg="#A6ADC8", anchor="w")
        self.lbl_estado.pack(fill=tk.X, pady=4)
        self.lbl_mision = tk.Label(tele_f, text="Misión: Esperando órdenes", font=self.font_telemetry, bg="#181825", fg="#CDD6F4", anchor="w")
        self.lbl_mision.pack(fill=tk.X, pady=4)
        self.lbl_distancia = tk.Label(tele_f, text="Distancia meta: 0.0m", font=self.font_telemetry, bg="#181825", fg="#CDD6F4", anchor="w")
        self.lbl_distancia.pack(fill=tk.X, pady=4)
        self.lbl_bateria = tk.Label(tele_f, text=" Batería: 100.0%", font=self.font_telemetry, bg="#181825", fg="#F9E2AF", anchor="w")
        self.lbl_bateria.pack(fill=tk.X, pady=4)

    # --- LÓGICA DE LA INTERFAZ ---
    def add_pedido(self, h):
        if not self.mision_activa: self.pedidos.append(h); self.actualizar_lista()

    def limpiar_pedidos(self):
        if not self.mision_activa: self.pedidos.clear(); self.actualizar_lista()

    def actualizar_lista(self):
        self.lista_box.delete(0, tk.END)
        if not self.pedidos: self.lista_box.insert(tk.END, "> Cola vacía...")
        else:
            for i, p in enumerate(self.pedidos): self.lista_box.insert(tk.END, f"  [+] {i+1}: {p}")

    def update_telemetry_labels(self, e, c, m, d, b):
        # Las actualizaciones que vienen del hilo de ROS deben pasarse a la UI de forma segura
        self.root.after(0, lambda: self._ui_upd(e, c, m, d, b))

    def _ui_upd(self, e, c, m, d, b):
        self.lbl_estado.config(text=f"[ {e} ]", fg=c)
        self.lbl_mision.config(text=f"Misión: {m}")
        self.lbl_distancia.config(text=f"Distancia meta: {d:.1f}m")
        self.lbl_bateria.config(text=f" Batería: {max(0, b):.1f}%")

    def iniciar_mision(self):
        if not self.pedidos: return
        self.mision_activa = True
        self.btn_start.config(state=tk.DISABLED, text="🚀 EN RUTA...")
        self.env.set_pedidos(self.pedidos)
        # CRUCIAL: ROS es un bucle continuo. Si lo ejecutamos en el mismo hilo que Tkinter,
        # la ventana se queda pillada (congelada). Levantamos un Thread para separar cálculos de la UI.
        threading.Thread(target=self.ejecutar_ia, daemon=True).start()

    def ejecutar_ia(self):
        obs, _ = self.env.reset()
        while self.mision_activa:
            # Pedimos predicción al modelo. Deterministic=True asegura el mejor camino aprendido (no explora).
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, _ = self.env.step(action)
            
            if done:
                if self.env.mision_completada:
                    self.root.after(0, self.rehabilitar_ui)
                    break
                else: 
                    # Si chocó antes de tiempo, respawnea y espera 1 seg antes de reintentar
                    obs, _ = self.env.reset(); time.sleep(1.0)

    def rehabilitar_ui(self):
        self.mision_activa = False; self.pedidos.clear(); self.actualizar_lista()
        self.btn_start.config(state=tk.NORMAL, text="INICIAR MISIÓN")
        self._ui_upd("STANDBY", "#A6ADC8", "Misión finalizada", 0.0, self.env.bateria_simulada)

    def on_closing(self):
        self.mision_activa = False
        if rclpy.ok(): rclpy.shutdown()
        self.root.destroy(); sys.exit(0)

# Punto de entrada principal
if __name__ == '__main__':
    root = tk.Tk()
    app = FleetManagerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()