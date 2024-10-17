from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

from src.service.soundsignal import SoundSignal

# Carregar o modelo treinado do YOLO
model = YOLO('runs/detect/train28/weights/best.pt')  # Certifique-se de que o caminho está correto

# Inicializar MediaPipe para mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Abrir a webcam
cap = cv2.VideoCapture(0)

# Verifique se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Definir a resolução da câmera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Definir as coordenadas do retângulo de segurança no centro da tela
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definindo as dimensões do retângulo de segurança
rect_width, rect_height = 100, 600
rect_x1 = (frame_width - rect_width) // 2
rect_y1 = (frame_height - rect_height) // 2
rect_x2 = rect_x1 + rect_width
rect_y2 = rect_y1 + rect_height

# Loop para processar o vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Inverter a imagem para obter o efeito de selfie
    frame = cv2.flip(frame, 1)

    # Realizar a inferência com YOLO para detectar mãos
    results = model(frame)

    # Inicializar a variável que indica se a área de segurança foi violada
    area_violated = False

    # Definir o retângulo de segurança
    safety_rect = (rect_x1, rect_y1, rect_x2 - rect_x1, rect_y2 - rect_y1)

    # Filtrar as detecções para aquelas que têm confiança maior que 60%
    for r in results:
        for box in r.boxes:  # Cada box é uma detecção do YOLO

            # Extrair a confiança da detecção
            confidence = box.conf[0].item()

            # Apenas considere detecções com confiança maior que 60%
            if confidence >= 0.7:
                # Extrair as coordenadas da bounding box e garantir que são inteiros
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Recortar a região da mão da imagem original
                hand_region = frame[y1:y2, x1:x2]

                # Converter para escala de cinza e aplicar um filtro gaussiano para suavizar
                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)

                # Aplicar um limiar binário para obter uma imagem binária da mão
                _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Encontrar contornos na região da mão
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Desenhar o contorno da mão detectada na cor rosa sobre a imagem original
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Ignorar pequenos ruídos
                        # Deslocar o contorno da mão para a posição correta no frame completo
                        contour[:, 0, 0] += x1
                        contour[:, 0, 1] += y1

                        # Desenhar o contorno
                        cv2.drawContours(frame, [contour], -1, (255, 0, 255), 3)

                        # Verificar se o contorno rosa cruza o retângulo de segurança usando boundingRect
                        x, y, w, h = cv2.boundingRect(contour)
                        if (
                            x < rect_x2 and x + w > rect_x1 and
                            y < rect_y2 and y + h > rect_y1
                        ):
                            area_violated = True
                            sound = SoundSignal()
                            sound.startSignal()
                            break

                # Exibir o valor da confiança acima do contorno
                label = f'Conf: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Converter a região da mão para RGB, pois o MediaPipe usa RGB
                hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)

                # Processar a região da mão com o MediaPipe para detectar landmarks
                result = hands.process(hand_region_rgb)

                # Se landmarks forem detectados, desenhá-los na imagem original
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Desenhar os landmarks da mão na imagem original com o offset
                        mp_draw.draw_landmarks(frame[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Desenhar o retângulo de segurança no centro da tela
    rect_color = (0, 255, 0) if not area_violated else (0, 0, 255)  # Verde se seguro, vermelho se violado
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, 3)

    # Exibir o frame processado com as detecções
    cv2.imshow('Detecção de Mãos em Tempo Real', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
