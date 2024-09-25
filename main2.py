import cv2
from ultralytics import YOLO
import mediapipe as mp

# Carregar o modelo YOLOv8 treinado
model = YOLO('runs/detect/train11/weights/best.pt')  # Substitua pelo caminho correto para o modelo treinado

# Inicializar MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Abrir a webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Loop para processar vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar a inferência com YOLOv8
    results = model(frame)

    # Iterar sobre cada detecção
    for r in results:
        for box in r.boxes:  # Cada box é uma detecção de mão do YOLO
            # Extrair as coordenadas da bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box (x1, y1) (x2, y2)
            confidence = box.conf[0]  # Confiança da detecção

            # Desenhar a bounding box no frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Desenha o retângulo (bounding box)

            # Exibir o percentual de confiança na detecção
            label = f'Conf: {confidence:.2f}'  # Formata a confiança como um texto
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Desenha o texto

            # Cortar a região da mão para passar para o MediaPipe
            hand_roi = frame[y1:y2, x1:x2]

            # Converter a imagem para o formato RGB exigido pelo MediaPipe
            hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)

            # Processar a ROI com o MediaPipe para detectar landmarks
            results_hands = hands.process(hand_roi_rgb)

            # Se landmarks forem detectados, desenhar
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Desenhar os landmarks diretamente no frame original, ajustando as coordenadas
                    mp_drawing.draw_landmarks(
                        frame[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Exibir o frame processado com as detecções do YOLOv8 (bounding boxes) e os landmarks do MediaPipe
    cv2.imshow('Detecção de Mãos: Bounding Boxes, Confiança e Landmarks', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
