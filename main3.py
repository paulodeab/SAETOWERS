import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv8 treinado
model = YOLO('runs/detect/train11/weights/best.pt')  # Substitua pelo caminho correto para o modelo treinado

# Abrir a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Função para encontrar e desenhar os contornos dentro da bounding box
def draw_hand_contours(frame, x1, y1, x2, y2):
    # Recortar a região da mão (ROI) usando as coordenadas da bounding box
    hand_roi = frame[y1:y2, x1:x2]

    # Converter a ROI para escala de cinza
    gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

    # Aplicar um desfoque para reduzir o ruído
    blurred_hand = cv2.GaussianBlur(gray_hand, (5, 5), 0)

    # Detectar os bordas usando Canny (para melhor detecção de contornos)
    edges = cv2.Canny(blurred_hand, 50, 150)

    # Encontrar os contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos no frame original
    cv2.drawContours(hand_roi, contours, -1, (0, 255, 0), 2)

# Loop para processar o vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar a inferência com YOLOv8
    results = model(frame)

    # Iterar sobre cada detecção
    for r in results:
        for box in r.boxes:  # Cada box é uma detecção de mão do YOLO
            confidence = box.conf[0]  # Confiança da detecção
             # Apenas processar se a confiança for maior ou igual a 70%
            if confidence >= 0.5: 
                # Extrair as coordenadas da bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
            
                label = f'Conf: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Desenhar a bounding box no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
                # Desenhar os contornos da mão dentro da bounding box
                draw_hand_contours(frame, x1, y1, x2, y2)
               

    # Exibir o frame processado com as detecções e os contornos das mãos
    cv2.imshow('Detecção de Mãos e Contornos em Tempo Real', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
