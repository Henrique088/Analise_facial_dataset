# scripts/converter_ferplus.py

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# Caminhos
BASE_PATH = Path("../data")
RAW_PATH = BASE_PATH / "raw"
UNIFIED_PATH = BASE_PATH / "unified_yolo"

# Mapeamento do FERPlus para nossas 7 classes
# FERPlus: angry, contempt, disgust, fear, happy, neutral, sad, suprise
FERPLUS_MAP = {
    "angry": 0,      # Raiva
    "contempt": 1,   # Nojo (mapeado)
    "disgust": 1,    # Nojo
    "fear": 2,       # Medo
    "happy": 3,      # Felicidade
    "neutral": 6,    # Neutro
    "sad": 4,        # Tristeza
    "suprise": 5,    # Surpresa
}

def criar_label_yolo_para_face(imagem_path, classe_id, dest_labels):
    """
    Cria um arquivo .txt no formato YOLO para a imagem.
    Como FERPlus tem apenas rostos centralizados, usamos a imagem inteira como bounding box.
    """
    # Ler imagem para obter dimensões
    img = cv2.imread(str(imagem_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # YOLO formato: <class> <x_center> <y_center> <width> <height>
    # Como o rosto ocupa quase toda a imagem, usamos 0.9 da imagem
    x_center = 0.5
    y_center = 0.5
    width = 0.9
    height = 0.9
    
    # Nome do arquivo label (mesmo nome da imagem, mas .txt)
    label_path = dest_labels / f"{imagem_path.stem}.txt"
    
    # Escrever label
    with open(label_path, 'w') as f:
        f.write(f"{classe_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True

def converter_ferplus():
    """Converte FERPlus para formato YOLO e distribui nos splits train/valid/test"""
    
    ferplus_path = RAW_PATH / "ferplus"
    
    print("📋 Convertendo FERPlus (78.293 imagens)...")
    
    # Estatísticas
    stats = {split: 0 for split in ['train', 'validation', 'test']}
    
    # Processar cada split do FERPlus
    for split_orig in ['train', 'validation', 'test']:
        split_path = ferplus_path / split_orig
        
        if not split_path.exists():
            print(f"   ⚠️ Split {split_orig} não encontrado")
            continue
        
        # Mapear para nosso split
        if split_orig == 'validation':
            split_dest = 'valid'
        else:
            split_dest = split_orig  # train → train, test → test
        
        print(f"\n   Processando {split_orig} → {split_dest}:")
        
        dest_images = UNIFIED_PATH / split_dest / 'images'
        dest_labels = UNIFIED_PATH / split_dest / 'labels'
        
        # Processar cada emoção
        for emocao_orig, classe_id in FERPLUS_MAP.items():
            emocao_path = split_path / emocao_orig
            
            if not emocao_path.exists():
                continue
            
            # Listar imagens
            imagens = list(emocao_path.glob("*.[jJ][pP][gG]")) + list(emocao_path.glob("*.[pP][nN][gG]"))
            
            print(f"      📁 {emocao_orig}: {len(imagens)} imagens → classe {classe_id}")
            
            for img_path in tqdm(imagens, desc=f"         Convertendo", leave=False):
                # Gerar nome único para evitar conflitos
                novo_nome = f"ferplus_{split_orig}_{emocao_orig}_{img_path.name}"
                dest_img = dest_images / novo_nome
                
                # Copiar imagem
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)
                
                # Criar label
                if criar_label_yolo_para_face(dest_img, classe_id, dest_labels):
                    stats[split_orig] += 1
    
    # Mostrar resumo
    print("\n" + "="*50)
    print("📊 FERPlus - Conversão Concluída:")
    print("="*50)
    print(f"   Train: {stats['train']} imagens")
    print(f"   Validation: {stats['validation']} imagens → movidas para valid")
    print(f"   Test: {stats['test']} imagens")
    print(f"   TOTAL: {sum(stats.values())} imagens")
    
    return stats

def verificar_imagens_corrompidas():
    """Verifica se há imagens corrompidas no dataset unificado"""
    print("\n🔍 Verificando imagens corrompidas...")
    
    corrompidas = []
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        if images_path.exists():
            for img_path in tqdm(list(images_path.glob("*.*")), desc=f"   Verificando {split}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        corrompidas.append(img_path)
                except:
                    corrompidas.append(img_path)
    
    if corrompidas:
        print(f"   ⚠️ Encontradas {len(corrompidas)} imagens corrompidas:")
        for img in corrompidas[:5]:  # Mostrar apenas as primeiras 5
            print(f"      - {img.name}")
        
        # Opção: remover imagens corrompidas
        resposta = input("\n   Remover imagens corrompidas? (s/n): ")
        if resposta.lower() == 's':
            for img in corrompidas:
                img.unlink()
                # Remover label correspondente
                label = img.parent.parent / 'labels' / f"{img.stem}.txt"
                if label.exists():
                    label.unlink()
            print(f"   ✅ {len(corrompidas)} imagens removidas")
    else:
        print("   ✅ Nenhuma imagem corrompida encontrada")
    
    return len(corrompidas)

if __name__ == "__main__":
    # Converter FERPlus
    stats = converter_ferplus()
    
    # Verificar integridade
    corrompidas = verificar_imagens_corrompidas()
    
    print("\n" + "="*50)
    print("✅ PASSO 6 CONCLUÍDO!")
    print("="*50)
    print(f"FERPlus adicionado ao dataset unificado")
    print(f"Total de imagens processadas: {sum(stats.values())}")