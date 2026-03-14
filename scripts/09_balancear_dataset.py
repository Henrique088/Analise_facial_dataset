# scripts/09_balancear_dataset_corrigido.py

import os
import shutil
import random
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Caminhos
BASE_PATH = Path("../data")
UNIFIED_PATH = BASE_PATH / "unified_yolo"

# Configurações
TARGET_IMAGES_PER_CLASS = 30000  # Meta: 30k imagens por classe
AUGMENTACAO_FATOR_MAX = 3  # Máximo de 3x augmentation

def contar_imagens_por_classe():
    """Conta quantas imagens temos por classe"""
    
    contagem = defaultdict(int)
    classes = ["Raiva", "Nojo", "Medo", "Felicidade", "Tristeza", "Surpresa", "Neutro"]
    
    print("🔍 Contando imagens por classe...")
    
    for split in ['train', 'valid', 'test']:
        labels_path = UNIFIED_PATH / split / 'labels'
        if labels_path.exists():
            for label_path in tqdm(list(labels_path.glob("*.txt")), desc=f"   {split}"):
                with open(label_path, 'r', encoding='utf-8') as f:
                    primeira_linha = f.readline().strip()
                    if primeira_linha:
                        classe = int(primeira_linha.split()[0])
                        contagem[classe] += 1
    
    print("\n📊 DISTRIBUICAO ATUAL:")  # Sem emoji
    for i, nome in enumerate(classes):
        print(f"   {nome}: {contagem[i]:,} imagens")
    
    return contagem

def identificar_classes_para_augmentar(contagem):
    """Identifica quais classes precisam de augmentation"""
    
    maior_classe = max(contagem.values())
    alvo = min(maior_classe, TARGET_IMAGES_PER_CLASS)
    
    necessidades = {}
    for classe, qtd in contagem.items():
        if qtd < alvo * 0.7:  # Se tiver menos que 70% da maior classe
            deficit = alvo - qtd
            fator = min(AUGMENTACAO_FATOR_MAX, int(deficit / qtd) + 1)
            if fator > 1:  # Só incluir se precisar aumentar
                necessidades[classe] = {
                    'atual': qtd,
                    'deficit': deficit,
                    'fator': fator,
                    'alvo': min(alvo, qtd * fator)
                }
    
    return necessidades

def criar_transformacoes_augmentation():
    """Cria pipeline de data augmentation para faces (CORRIGIDO)"""
    
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # CORRIGIDO: GaussNoise usa std dev, não var_limit
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3, mean=0),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        # CORRIGIDO: Usar Affine em vez de ShiftScaleRotate (warning)
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            p=0.3
        ),
    ])

def aplicar_augmentation(imagem_path, label_path, classe_id, dest_images, dest_labels, transform, n_variacoes):
    """Aplica augmentation a uma imagem e cria variações"""
    
    # Ler imagem original
    img = cv2.imread(str(imagem_path))
    if img is None:
        return 0
    
    # Ler label original (bounding box)
    with open(label_path, 'r', encoding='utf-8') as f:
        linha = f.readline().strip()
        partes = linha.split()
        if len(partes) < 5:
            return 0
        bbox = [float(x) for x in partes[1:5]]
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    criadas = 0
    for i in range(n_variacoes):
        try:
            # Aplicar augmentation
            augmented = transform(image=img_rgb)
            img_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            
            # Gerar nome único
            nome_base = f"aug_{imagem_path.stem}_v{i}{imagem_path.suffix}"
            dest_img = dest_images / nome_base
            
            # Salvar imagem aumentada
            cv2.imwrite(str(dest_img), img_aug)
            
            # Salvar label (mesmo bbox da original)
            dest_label = dest_labels / f"{dest_img.stem}.txt"
            with open(dest_label, 'w', encoding='utf-8') as f:
                f.write(f"{classe_id} {' '.join([str(x) for x in bbox])}\n")
            
            criadas += 1
        except Exception as e:
            print(f"      ⚠️ Erro ao augmentar {imagem_path.name}: {e}")
            continue
    
    return criadas

def balancear_dataset(contagem, necessidades):
    """Executa o balanceamento com augmentation"""
    
    print("\n🔧 APLICANDO DATA AUGMENTATION...")
    
    transform = criar_transformacoes_augmentation()
    
    stats = defaultdict(int)
    
    # Processar apenas imagens de TREINO
    images_path = UNIFIED_PATH / 'train' / 'images'
    labels_path = UNIFIED_PATH / 'train' / 'labels'
    
    dest_images = images_path
    dest_labels = labels_path
    
    # Agrupar imagens por classe
    imagens_por_classe = defaultdict(list)
    
    print("   Agrupando imagens por classe...")
    for label_path in tqdm(list(labels_path.glob("*.txt")), desc="   Varrendo labels"):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                primeira_linha = f.readline().strip()
                if primeira_linha:
                    classe = int(primeira_linha.split()[0])
                    # Procurar imagem correspondente
                    imagem_correspondente = images_path / f"{label_path.stem}.jpg"
                    if not imagem_correspondente.exists():
                        imagem_correspondente = images_path / f"{label_path.stem}.png"
                    if not imagem_correspondente.exists():
                        imagem_correspondente = images_path / f"{label_path.stem}.jpeg"
                    
                    if imagem_correspondente.exists():
                        imagens_por_classe[classe].append((imagem_correspondente, label_path))
        except Exception as e:
            continue
    
    # Aplicar augmentation nas classes necessárias
    for classe, info in necessidades.items():
        if classe not in imagens_por_classe:
            continue
        
        imagens_classe = imagens_por_classe[classe]
        n_imagens = len(imagens_classe)
        fator = info['fator']
        
        print(f"\n   📁 Classe {classe}: {n_imagens} originais, fator {fator}x")
        
        n_variacoes = fator - 1
        
        if n_variacoes <= 0:
            continue
        
        # Selecionar imagens para aumentar (usar 80% das imagens)
        imagens_selecionadas = random.sample(
            imagens_classe,
            min(len(imagens_classe), int(len(imagens_classe) * 0.8))
        )
        
        for img_path, label_path in tqdm(imagens_selecionadas, desc=f"      Augmentando classe {classe}"):
            criadas = aplicar_augmentation(
                img_path, label_path, classe,
                dest_images, dest_labels,
                transform, n_variacoes
            )
            stats[classe] += criadas
    
    return stats

def salvar_relatorio(contagem_original, contagem_final, stats_aug):
    """Salva relatório do balanceamento (SEM EMOJIS)"""
    
    relatorio_path = UNIFIED_PATH / 'balanceamento_relatorio.txt'
    
    # Usar encoding utf-8 e errors='ignore' para evitar problemas com caracteres especiais
    with open(relatorio_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write("="*60 + "\n")
        f.write("RELATORIO DE BALANCEAMENTO DO DATASET\n")  # Sem emoji
        f.write("="*60 + "\n\n")
        
        f.write("CLASSES:\n")
        classes = ["Raiva", "Nojo", "Medo", "Felicidade", "Tristeza", "Surpresa", "Neutro"]
        
        f.write("\nANTES DO AUGMENTATION:\n")
        for i, nome in enumerate(classes):
            f.write(f"   {nome}: {contagem_original[i]:,} imagens\n")
        
        f.write("\nIMAGENS CRIADAS POR AUGMENTATION:\n")
        for i, nome in enumerate(classes):
            if i in stats_aug and stats_aug[i] > 0:
                f.write(f"   {nome}: +{stats_aug[i]} imagens\n")
        
        f.write("\nDEPOIS DO AUGMENTATION:\n")
        for i, nome in enumerate(classes):
            f.write(f"   {nome}: {contagem_final[i]:,} imagens\n")
        
        f.write(f"\nTOTAL ORIGINAL: {sum(contagem_original.values()):,} imagens\n")
        f.write(f"TOTAL CRIADO: {sum(stats_aug.values()):,} imagens\n")
        f.write(f"TOTAL FINAL: {sum(contagem_final.values()):,} imagens\n")
    
    print(f"\n✅ Relatorio salvo em: {relatorio_path}")
    
    # Também mostrar na tela sem emojis
    print("\n" + "="*60)
    print("RESUMO DO BALANCEAMENTO:")
    print("="*60)
    print(f"Imagens originais: {sum(contagem_original.values()):,}")
    print(f"Imagens criadas: {sum(stats_aug.values()):,}")
    print(f"Total final: {sum(contagem_final.values()):,}")

def atualizar_data_yaml():
    """Atualiza o data.yaml com as novas contagens"""
    
    import yaml
    
    yaml_path = UNIFIED_PATH / 'data.yaml'
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Adicionar info de balanceamento
    config['balanceado'] = True
    config['total_imagens'] = sum(contar_imagens_por_classe().values())
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ data.yaml atualizado")

if __name__ == "__main__":
    print("="*60)
    print("PASSO 9: BALANCEAMENTO COM DATA AUGMENTATION")
    print("="*60)
    
    # Contar distribuição atual
    contagem_original = contar_imagens_por_classe()
    
    # Identificar classes que precisam de augmentation
    necessidades = identificar_classes_para_augmentar(contagem_original)
    
    if necessidades:
        print("\n📌 Classes que precisam de augmentation:")
        for classe, info in necessidades.items():
            print(f"   Classe {classe}: {info['atual']} → alvo {info['alvo']:.0f} (fator {info['fator']}x)")
        
        # Aplicar augmentation
        stats_aug = balancear_dataset(contagem_original, necessidades)
        
        # Contar nova distribuição
        contagem_final = contar_imagens_por_classe()
        
        # Salvar relatório (SEM EMOJIS)
        salvar_relatorio(contagem_original, contagem_final, stats_aug)
        
        # Atualizar data.yaml
        atualizar_data_yaml()
        
    else:
        print("\n✅ Dataset já está bem balanceado! Não precisa de augmentation.")
    
    print("\n" + "="*60)
    print("PASSO 9 CONCLUIDO! Dataset pronto para treinamento!")
    print("="*60)