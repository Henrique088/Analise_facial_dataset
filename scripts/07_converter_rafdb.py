# scripts/07_converter_rafdb.py

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

# Caminhos
BASE_PATH = Path("../data")
RAW_PATH = BASE_PATH / "raw"
UNIFIED_PATH = BASE_PATH / "unified_yolo"

# Mapeamento do RAF-DB para nossas 7 classes
# RAF-DB: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
RAFDB_MAP = {
    "1": 5,  # Surprise
    "2": 2,  # Fear
    "3": 1,  # Disgust
    "4": 3,  # Happiness
    "5": 4,  # Sadness
    "6": 0,  # Anger
    "7": 6,  # Neutral
}

def criar_label_yolo_para_face(imagem_path, classe_id, dest_labels):
    """
    Cria um arquivo .txt no formato YOLO para a imagem.
    RAF-DB tem rostos centralizados, usamos a imagem inteira como bounding box.
    """
    # Ler imagem para obter dimensões
    img = cv2.imread(str(imagem_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # YOLO formato: <class> <x_center> <y_center> <width> <height>
    x_center = 0.5
    y_center = 0.5
    width = 0.9
    height = 0.9
    
    # Nome do arquivo label
    label_path = dest_labels / f"{imagem_path.stem}.txt"
    
    # Escrever label
    with open(label_path, 'w') as f:
        f.write(f"{classe_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True

def converter_rafdb():
    """Converte RAF-DB para formato YOLO usando os CSVs de label"""
    
    rafdb_path = RAW_PATH / "rafdb"
    dataset_path = rafdb_path / "DATASET"
    
    print("📋 Convertendo RAF-DB (15.339 imagens)...")
    
    # Carregar CSVs
    train_csv = pd.read_csv(rafdb_path / "train_labels.csv")
    test_csv = pd.read_csv(rafdb_path / "test_labels.csv")
    
    # Assumindo que os CSVs têm colunas: 'image' e 'label'
    # Pode ser necessário ajustar os nomes das colunas
    print(f"\n   📊 train_labels.csv: {len(train_csv)} linhas")
    print(f"   📊 test_labels.csv: {len(test_csv)} linhas")
    
    # Estatísticas
    stats = {'train': 0, 'test': 0}
    distribuicao = {classe: 0 for classe in range(7)}
    
    # Processar TRAIN
    print("\n   Processando TRAIN...")
    dest_images_train = UNIFIED_PATH / 'train' / 'images'
    dest_labels_train = UNIFIED_PATH / 'train' / 'labels'
    
    for idx, row in tqdm(train_csv.iterrows(), total=len(train_csv), desc="   Convertendo train"):
        # Pode ser que a coluna de imagem seja 'image' e label seja 'label'
        # Ajuste conforme necessário
        nome_imagem = row.iloc[0]  # Primeira coluna
        label_original = str(int(row.iloc[1]))  # Segunda coluna como string
        
        # Mapear label
        classe_id = RAFDB_MAP.get(label_original)
        if classe_id is None:
            continue
        
        # Procurar imagem na pasta DATASET/train
        imagem_encontrada = None
        pasta_label = dataset_path / 'train' / label_original
        if pasta_label.exists():
            possivel_imagem = pasta_label / nome_imagem
            if possivel_imagem.exists():
                imagem_encontrada = possivel_imagem
            else:
                # Procurar qualquer imagem com nome parecido
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
                    matches = list(pasta_label.glob(f"*{nome_imagem}*"))
                    if matches:
                        imagem_encontrada = matches[0]
                        break
        
        if imagem_encontrada and imagem_encontrada.exists():
            # Gerar nome único
            novo_nome = f"rafdb_train_{idx}_{imagem_encontrada.name}"
            dest_img = dest_images_train / novo_nome
            
            # Copiar imagem
            if not dest_img.exists():
                shutil.copy2(imagem_encontrada, dest_img)
            
            # Criar label
            if criar_label_yolo_para_face(dest_img, classe_id, dest_labels_train):
                stats['train'] += 1
                distribuicao[classe_id] += 1
    
    # Processar TEST
    print("\n   Processando TEST...")
    dest_images_test = UNIFIED_PATH / 'test' / 'images'
    dest_labels_test = UNIFIED_PATH / 'test' / 'labels'
    
    for idx, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="   Convertendo test"):
        nome_imagem = row.iloc[0]
        label_original = str(int(row.iloc[1]))
        
        classe_id = RAFDB_MAP.get(label_original)
        if classe_id is None:
            continue
        
        # Procurar imagem na pasta DATASET/test
        imagem_encontrada = None
        pasta_label = dataset_path / 'test' / label_original
        if pasta_label.exists():
            possivel_imagem = pasta_label / nome_imagem
            if possivel_imagem.exists():
                imagem_encontrada = possivel_imagem
            else:
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
                    matches = list(pasta_label.glob(f"*{nome_imagem}*"))
                    if matches:
                        imagem_encontrada = matches[0]
                        break
        
        if imagem_encontrada and imagem_encontrada.exists():
            novo_nome = f"rafdb_test_{idx}_{imagem_encontrada.name}"
            dest_img = dest_images_test / novo_nome
            
            if not dest_img.exists():
                shutil.copy2(imagem_encontrada, dest_img)
            
            if criar_label_yolo_para_face(dest_img, classe_id, dest_labels_test):
                stats['test'] += 1
                distribuicao[classe_id] += 1
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("📊 RAF-DB - Conversão Concluída:")
    print("="*50)
    print(f"   Train: {stats['train']} imagens")
    print(f"   Test: {stats['test']} imagens")
    print(f"   TOTAL: {stats['train'] + stats['test']} imagens")
    
    print("\n📊 Distribuição por classe:")
    classes = ["Raiva", "Nojo", "Medo", "Felicidade", "Tristeza", "Surpresa", "Neutro"]
    for i, nome in enumerate(classes):
        print(f"   {nome}: {distribuicao[i]} imagens")
    
    return stats

def verificar_duplicatas():
    """Verifica se há imagens duplicadas no dataset"""
    print("\n🔍 Verificando imagens duplicadas...")
    
    from collections import defaultdict
    import hashlib
    
    hash_map = defaultdict(list)
    
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        if images_path.exists():
            for img_path in tqdm(list(images_path.glob("*.*")), desc=f"   Calculando hash {split}"):
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_map[file_hash].append(img_path)
    
    duplicatas = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    
    if duplicatas:
        print(f"   ⚠️ Encontrados {len(duplicatas)} grupos de imagens duplicadas")
        total_duplicadas = sum(len(paths) - 1 for paths in duplicatas.values())
        print(f"   Total de imagens duplicadas: {total_duplicadas}")
        
        # Opção: remover duplicatas (manter apenas uma)
        resposta = input("\n   Remover imagens duplicadas? (s/n): ")
        if resposta.lower() == 's':
            removidas = 0
            for hash_val, paths in duplicatas.items():
                # Manter a primeira, remover as outras
                for path in paths[1:]:
                    path.unlink()
                    # Remover label correspondente
                    label = path.parent.parent / 'labels' / f"{path.stem}.txt"
                    if label.exists():
                        label.unlink()
                    removidas += 1
            print(f"   ✅ {removidas} imagens duplicadas removidas")
    else:
        print("   ✅ Nenhuma imagem duplicada encontrada")
    
    return len(duplicatas)

if __name__ == "__main__":
    # Converter RAF-DB
    stats = converter_rafdb()
    
    # Verificar duplicatas
    duplicatas = verificar_duplicatas()
    
    # Contar total atual do dataset
    print("\n" + "="*50)
    print("📊 STATUS ATUAL DO DATASET UNIFICADO:")
    print("="*50)
    
    total_geral = 0
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        if images_path.exists():
            qtd = len(list(images_path.glob("*.*")))
            print(f"   {split}: {qtd} imagens")
            total_geral += qtd
    
    print(f"\n🔥 TOTAL GERAL: {total_geral} imagens")
    print("\n✅ PASSO 7 CONCLUÍDO!")