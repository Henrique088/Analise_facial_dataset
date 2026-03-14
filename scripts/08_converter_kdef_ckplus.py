# scripts/08_converter_kdef_ckplus.py

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

# Caminhos
BASE_PATH = Path("../data")
RAW_PATH = BASE_PATH / "raw"
UNIFIED_PATH = BASE_PATH / "unified_yolo"

# Mapeamento KDEF para nossas 7 classes
KDEF_MAP = {
    "angry": 0,      # Raiva
    "disgust": 1,    # Nojo
    "fear": 2,       # Medo
    "happy": 3,      # Felicidade
    "neutral": 6,    # Neutro
    "sad": 4,        # Tristeza
    "surprise": 5,   # Surpresa
}

# Mapeamento CKPlus para nossas 7 classes
CKPLUS_MAP = {
    "anger": 0,      # Raiva
    "contempt": 1,   # Nojo (mapeado)
    "disgust": 1,    # Nojo
    "fear": 2,       # Medo
    "happiness": 3,  # Felicidade
    "neutral": 6,    # Neutro
    "sadness": 4,    # Tristeza
    "surprise": 5,   # Surpresa
}

def criar_label_yolo_para_face(imagem_path, classe_id, dest_labels):
    """
    Cria um arquivo .txt no formato YOLO para a imagem.
    KDEF e CKPlus têm rostos centralizados, usamos a imagem inteira.
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

def converter_kdef():
    """Converte KDEF para formato YOLO"""
    
    kdef_path = RAW_PATH / "kdef"
    
    print("\n📋 Convertendo KDEF (2.938 imagens)...")
    
    stats = {emocao: 0 for emocao in KDEF_MAP.keys()}
    total = 0
    
    # KDEF vai todo para TREINO (já temos test/valid do AffectNet)
    dest_images = UNIFIED_PATH / 'train' / 'images'
    dest_labels = UNIFIED_PATH / 'train' / 'labels'
    
    for emocao_orig, classe_id in KDEF_MAP.items():
        emocao_path = kdef_path / emocao_orig
        
        if not emocao_path.exists():
            print(f"   ⚠️ Pasta {emocao_orig} não encontrada")
            continue
        
        # Listar imagens
        imagens = list(emocao_path.glob("*.[jJ][pP][gG]")) + list(emocao_path.glob("*.[pP][nN][gG]"))
        
        print(f"   📁 {emocao_orig}: {len(imagens)} imagens → classe {classe_id}")
        
        for img_path in tqdm(imagens, desc=f"      Convertendo", leave=False):
            # Gerar nome único
            novo_nome = f"kdef_{emocao_orig}_{img_path.name}"
            dest_img = dest_images / novo_nome
            
            # Copiar imagem (se já não existir)
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            
            # Criar label
            if criar_label_yolo_para_face(dest_img, classe_id, dest_labels):
                stats[emocao_orig] += 1
                total += 1
    
    print(f"\n   ✅ KDEF convertido: {total} imagens")
    return total

def converter_ckplus():
    """Converte CKPlus para formato YOLO"""
    
    ckplus_path = RAW_PATH / "ckplus"
    
    print("\n📋 Convertendo CKPlus (920 imagens)...")
    
    stats = {emocao: 0 for emocao in CKPLUS_MAP.keys()}
    total = 0
    
    # CKPlus vai todo para TREINO (já temos test/valid)
    dest_images = UNIFIED_PATH / 'train' / 'images'
    dest_labels = UNIFIED_PATH / 'train' / 'labels'
    
    for emocao_orig, classe_id in CKPLUS_MAP.items():
        emocao_path = ckplus_path / emocao_orig
        
        if not emocao_path.exists():
            continue
        
        # Listar imagens
        imagens = list(emocao_path.glob("*.[jJ][pP][gG]")) + list(emocao_path.glob("*.[pP][nN][gG]"))
        
        if imagens:
            print(f"   📁 {emocao_orig}: {len(imagens)} imagens → classe {classe_id}")
        
        for img_path in tqdm(imagens, desc=f"      Convertendo", leave=False):
            # Gerar nome único
            novo_nome = f"ckplus_{emocao_orig}_{img_path.name}"
            dest_img = dest_images / novo_nome
            
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            
            if criar_label_yolo_para_face(dest_img, classe_id, dest_labels):
                stats[emocao_orig] += 1
                total += 1
    
    print(f"\n   ✅ CKPlus convertido: {total} imagens")
    return total

def verificar_balanceamento():
    """Verifica o balanceamento das classes no dataset unificado"""
    
    print("\n" + "="*50)
    print("📊 VERIFICANDO BALANCEAMENTO DAS CLASSES:")
    print("="*50)
    
    # Contar labels por classe
    classes = ["Raiva", "Nojo", "Medo", "Felicidade", "Tristeza", "Surpresa", "Neutro"]
    contagem = {i: 0 for i in range(7)}
    
    for split in ['train', 'valid', 'test']:
        labels_path = UNIFIED_PATH / split / 'labels'
        if labels_path.exists():
            for label_path in tqdm(list(labels_path.glob("*.txt")), desc=f"   Analisando {split}"):
                with open(label_path, 'r') as f:
                    for linha in f:
                        classe = int(linha.strip().split()[0])
                        if classe in contagem:
                            contagem[classe] += 1
    
    # Mostrar distribuição
    print("\n📊 Distribuição atual:")
    for i, nome in enumerate(classes):
        print(f"   {nome}: {contagem[i]:,} imagens")
    
    # Calcular estatísticas
    valores = list(contagem.values())
    total = sum(valores)
    media = total / 7
    maximo = max(valores)
    minimo = min(valores)
    razao = maximo / minimo
    
    print(f"\n📈 Estatísticas:")
    print(f"   Total: {total:,} imagens")
    print(f"   Média por classe: {media:.0f} imagens")
    print(f"   Classe majoritária: {maximo:,} imagens")
    print(f"   Classe minoritária: {minimo:,} imagens")
    print(f"   Razão majoritária/minoritária: {razao:.1f}x")
    
    if razao > 3:
        print("\n⚠️  Dataset desbalanceado! Precisa de balanceamento.")
        print("   Classe minoritária precisa de augmentation.")
    else:
        print("\n✅ Dataset bem balanceado!")
    
    return contagem

def criar_arquivos_lista():
    """Cria arquivos .txt com listas de imagens para treino/validação/teste"""
    
    print("\n📋 Criando arquivos de lista para treinamento...")
    
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        if images_path.exists():
            lista_path = UNIFIED_PATH / f"{split}.txt"
            
            with open(lista_path, 'w') as f:
                for img_path in sorted(images_path.glob("*.*")):
                    f.write(f"{img_path}\n")
            
            qtd = len(list(images_path.glob("*.*")))
            print(f"   ✅ {split}.txt: {qtd} imagens")

if __name__ == "__main__":
    print("="*60)
    print("🚀 PASSO 8: CONVERTENDO KDEF E CKPLUS")
    print("="*60)
    
    # Converter KDEF
    kdef_total = converter_kdef()
    
    # Converter CKPlus
    ckplus_total = converter_ckplus()
    
    # Verificar balanceamento
    contagem = verificar_balanceamento()
    
    # Criar arquivos de lista
    criar_arquivos_lista()
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO FINAL - PASSO 8 CONCLUÍDO!")
    print("="*60)
    print(f"KDEF adicionado: {kdef_total} imagens")
    print(f"CKPlus adicionado: {ckplus_total} imagens")
    print(f"TOTAL adicionado nesta etapa: {kdef_total + ckplus_total} imagens")
    
    # Contar total final
    total_final = 0
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        if images_path.exists():
            qtd = len(list(images_path.glob("*.*")))
            total_final += qtd
    
    print(f"\n🔥 TOTAL FINAL DO DATASET UNIFICADO: {total_final:,} imagens!")