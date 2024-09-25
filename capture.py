import cv2
import os

# Função para criar diretórios
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Diretório onde as imagens serão salvas
dataset_directory = 'dataset/train/images'
create_directory(dataset_directory)

# Definir as categorias para as imagens (como "com luvas" e "sem luvas")
categories = ['hands_gloves_333']
for category in categories:
    create_directory(os.path.join(dataset_directory, category))

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

current_category = 0  # Controla a categoria atual (0 -> com luvas, 1 -> sem luvas)
image_count = 0  # Contador para nomear as imagens

print(f"Categoria atual: {categories[current_category]}")
print("Pressione 's' para capturar uma imagem, 'n' para mudar de categoria, ou 'q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Mostra o feed da câmera
    cv2.imshow('Dataset Capture', frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):  # Pressione 's' para salvar uma imagem
        # Define o nome da imagem e o diretório correspondente
        image_name = f"{categories[current_category]}_{image_count}.jpg"
        image_path = os.path.join(dataset_directory, categories[current_category], image_name)

        # Salva a imagem
        cv2.imwrite(image_path, frame)
        print(f"Imagem salva: {image_path}")
        image_count += 1

    elif key & 0xFF == ord('n'):  # Pressione 'n' para mudar de categoria
        current_category += 1
        image_count = 0  # Reiniciar o contador de imagens
        if current_category >= len(categories):
            current_category = 0  # Voltar para a primeira categoria
        print(f"Categoria atual: {categories[current_category]}")

    elif key & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

cap.release()
cv2.destroyAllWindows()


    




