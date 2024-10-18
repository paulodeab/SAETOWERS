import threading
from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

from src.service.lightsignal import LightSignal
from src.service.soundsignal import SoundSignal
from src.config.config import AI

class HandRecognition:

    _model: YOLO

    def __init__(self):
        # Inicializar o modelo YOLO
        self._model = YOLO(AI.get("model"))

        # Inicializar MediaPipe para detecção de mãos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Inicializar a câmera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Erro ao abrir a câmera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

        # Definir as coordenadas do retângulo de segurança
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rect_width, self.rect_height = 100, 200
        self.rect_x1 = (self.frame_width - self.rect_width) // 2
        self.rect_y1 = (self.frame_height - self.rect_height) // 2
        self.rect_x2 = self.rect_x1 + self.rect_width
        self.rect_y2 = self.rect_y1 + self.rect_height

        # Inicializar o sinal de luz
        self.light_signal = LightSignal()

    def _draw_safety_rectangle(self, frame, area_violated):
        color = (0, 255, 0) if not area_violated else (0, 0, 255)
        cv2.rectangle(frame, (self.rect_x1, self.rect_y1),
                      (self.rect_x2, self.rect_y2), color, 3)

    def _process_hand(self, frame, box):
        confidence = box.conf[0].item()
        if confidence < 0.7:
            return False  # Ignorar detecções de baixa confiança

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        hand_region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_violated = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                contour[:, 0, 0] += x1
                contour[:, 0, 1] += y1
                cv2.drawContours(frame, [contour], -1, (255, 0, 255), 3)

                x, y, w, h = cv2.boundingRect(contour)
                if (x < self.rect_x2 and x + w > self.rect_x1 and y < self.rect_y2 and y + h > self.rect_y1):
                    area_violated = True

                    # Executar a lógica de sinalização em uma thread separada
                    threading.Thread(target=self._trigger_signals).start()
                threading.Thread(target=self.light_signal.stopSignal).start()

        cv2.putText(frame, f'Conf: {confidence:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        result = self.hands.process(hand_region_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame[y1:y2, x1:x2],
                                            hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)
        return area_violated

    def _trigger_signals(self):
        # Chamar o sinal de luz e som de forma independente
        s = SoundSignal()
        s.startSignal()
        self.light_signal.startSignal()

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = self._model(frame)
            area_violated = False

            for r in results:
                for box in r.boxes:
                    if self._process_hand(frame, box):
                        area_violated = True
                        break

            self._draw_safety_rectangle(frame, area_violated)
            cv2.imshow('Detecção de Mãos em Tempo Real', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()









