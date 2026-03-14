# scripts/04_copiar_affectnet.py

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# Caminhos
BASE_PATH = Path("../data")
RAW_PATH = BASE_PATH / "raw"
UNIFIED_PATH = BASE_PATH / "unified_yolo"

# Mapeamento de classes do AffectNet para nosso padrão (7 classes)
# AffectNet: 0=Anger,1=Contempt,2=Disgust,3=Fear,4=Happy,5=Neutral,6=Sad,7=Surprise
AFFECTNET_MAP = {
    "0": 0,  # Anger → Raiva
    "1": 1,  # Contempt → Nojo
    "2": 1,  # Disgust → Nojo
    "3": 2,  # Fear → Medo
    "4": 3,  # Happy → Felicidade
    "5": 6,  # Neutral → Neutro
    "6": 4,  # Sad → Tristeza
    "7": 5,  # Surprise → Surpresa
}

def converter_label_yolo(linha_label, mapping):
    """Converte uma linha de label YOLO para nova classe mapeada"""
    partes = linha_label.strip().split()
    if len(partes) >= 5:
        classe_original = partes[0]
        if classe_original in mapping:
            nova_classe = str(mapping[classe_original])
            return f"{nova_classe} {' '.join(partes[1:])}\n"
    return None

def copiar_affectnet():
    """Copia AffectNet para o dataset unificado, convertendo labels"""
    
    affectnet_path = RAW_PATH / "affectnet"
    
    # Mapeamento de splits: affectnet → unified
    split_map = {
        'train': 'train',
        'valid': 'valid',  # AffectNet usa 'valid', nós usamos 'valid'
        'test': 'test'
    }
    
    print("📋 Copiando AffectNet...")
    
    total_imagens = 0
    
    for split_orig, split_dest in split_map.items():
        orig_images = affectnet_path / split_orig / 'images'
        orig_labels = affectnet_path / split_orig / 'labels'
        
        dest_images = UNIFIED_PATH / split_dest / 'images'
        dest_labels = UNIFIED_PATH / split_dest / 'labels'
        
        if not orig_images.exists():
            print(f"   ⚠️ Pasta {split_orig} não encontrada, pulando...")
            continue
        
        # Listar imagens
        imagens = list(orig_images.glob("*.[jJ][pP][gG]")) + list(orig_images.glob("*.[pP][nN][gG]"))
        
        print(f"   Processando {split_orig} → {split_dest}: {len(imagens)} imagens")
        
        for img_path in tqdm(imagens, desc=f"   Copiando {split_orig}"):
            # Copiar imagem
            dest_img = dest_images / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            
            # Processar label correspondente
            label_path = orig_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                # Ler label original
                with open(label_path, 'r') as f:
                    linhas = f.readlines()
                
                # Converter linhas
                novas_linhas = []
                for linha in linhas:
                    nova_linha = converter_label_yolo(linha, AFFECTNET_MAP)
                    if nova_linha:
                        novas_linhas.append(nova_linha)
                
                # Salvar label convertido
                if novas_linhas:
                    dest_label = dest_labels / f"{img_path.stem}.txt"
                    with open(dest_label, 'w') as f:
                        f.writelines(novas_linhas)
            
            total_imagens += 1
    
    print(f"✅ AffectNet copiado: {total_imagens} imagens")
    return total_imagens

if __name__ == "__main__":
    # Primeiro, garantir que estrutura existe
    from criar_estrutura_unificada import criar_estrutura
    criar_estrutura()
    
    # Copiar AffectNet
    total = copiar_affectnet()
    print(f"\n📊 Total de imagens do AffectNet: {total}")