import os

# Caminho para as pastas de labels de treino e validação
train_labels_path = 'D:/GitHub/SAETOWERS/dataset/train/labels'
val_labels_path = 'D:/GitHub/SAETOWERS/dataset/val/labels'

def fix_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Corrigir a classe "15" para "0"
            new_lines = []
            for line in lines:
                if line.startswith('16'):
                    new_lines.append(line.replace('15', '0', 1))  # Substituir apenas o class_id
                else:
                    new_lines.append(line)

            # Sobrescrever o arquivo com os novos labels corrigidos
            with open(file_path, 'w') as file:
                file.writelines(new_lines)

# Corrigir labels em treino e validação
fix_labels(train_labels_path)
fix_labels(val_labels_path)

print("Labels corrigidos!")
