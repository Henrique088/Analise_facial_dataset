# scripts/16_unificar_datasets.py

import shutil
from pathlib import Path
from tqdm import tqdm

# Caminhos
UNIFIED_PATH = Path("../data/unified_yolo")
AUGMENTED_PATH = Path("../data/augmented_yolo")
BACKUP_PATH = Path("../data/unified_yolo_backup")

print("="*60)
print("🔄 UNIFICANDO DATASET ORIGINAL + IMAGENS GERADAS")
print("="*60)

# 1. FAZER BACKUP (por segurança)
print("\n📦 Criando backup...")
if not BACKUP_PATH.exists():
    shutil.copytree(UNIFIED_PATH, BACKUP_PATH)
    print(f"   ✅ Backup criado em: {BACKUP_PATH}")
else:
    print(f"   ⚠️ Backup já existe")

# 2. COPIAR IMAGENS GERADAS PARA O DATASET ORIGINAL
print("\n📋 Copiando imagens geradas...")

# Imagens
aug_images = list((AUGMENTED_PATH / 'train' / 'images').glob("*.*"))
for img_path in tqdm(aug_images, desc="Copiando imagens"):
    dest = UNIFIED_PATH / 'train' / 'images' / img_path.name
    if not dest.exists():
        shutil.copy2(img_path, dest)

# Labels
aug_labels = list((AUGMENTED_PATH / 'train' / 'labels').glob("*.txt"))
for label_path in tqdm(aug_labels, desc="Copiando labels"):
    dest = UNIFIED_PATH / 'train' / 'labels' / label_path.name
    if not dest.exists():
        shutil.copy2(label_path, dest)

# 3. VERIFICAR NOVAS CONTAGENS
print("\n📊 NOVA DISTRIBUIÇÃO DO DATASET:")

from collections import defaultdict
nova_contagem = defaultdict(int)

labels_path = UNIFIED_PATH / 'train' / 'labels'
for label_file in tqdm(list(labels_path.glob("*.txt")), desc="Contando classes"):
    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            classe = int(first_line.split()[0])
            nova_contagem[classe] += 1

print("\n📈 Comparação Antes vs Depois:")
print("Classe  | Antes    | Depois   | Aumento")
print("-"*45)

# Classes alvo
target_classes = [2, 4, 5]  # Medo, Tristeza, Surpresa
classes_nomes = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutro']

for classe in range(7):
    antes = 19182 if classe == 2 else 12155 if classe == 4 else 12069 if classe == 5 else "?"
    depois = nova_contagem[classe]
    
    if classe in target_classes:
        aumento = depois - (19182 if classe == 2 else 12155 if classe == 4 else 12069)
        print(f"{classes_nomes[classe]:8} | {antes:<7} | {depois:<7} | +{aumento}")
    else:
        print(f"{classes_nomes[classe]:8} | {antes:<7} | {depois:<7} | -")

print("\n" + "="*60)
print(f"✅ DATASET UNIFICADO!")
print(f"📁 Local: {UNIFIED_PATH}")
print(f"📊 Total de imagens: {sum(nova_contagem.values()):,}")
print("="*60)