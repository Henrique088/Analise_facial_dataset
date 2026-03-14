# scripts/01_verificar_datasets_refinado.py

import os
from pathlib import Path
from collections import defaultdict
import yaml

BASE_PATH = Path("../data/raw")

def verificar_estrutura_refinada():
    print("🔍 VERIFICANDO ESTRUTURA DETALHADA DOS DATASETS...\n")
    
    estatisticas = {}
    
    # 1. AFFECTNET (Já em formato YOLO!)
    affectnet_path = BASE_PATH / "affectnet"
    if affectnet_path.exists():
        print("📁 AFFECTNET (FORMATO YOLO):")
        estatisticas['affectnet'] = {'train': 0, 'val': 0, 'test': 0}
        
        # Carregar data.yaml para ver classes
        yaml_path = affectnet_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                print(f"   📄 data.yaml: {data_yaml.get('nc', '?')} classes")
                print(f"      names: {data_yaml.get('names', [])}")
        
        # Verificar cada split
        for split in ['train', 'valid', 'test']:
            split_path = affectnet_path / split
            if split_path.exists():
                images_path = split_path / 'images'
                labels_path = split_path / 'labels'
                
                if images_path.exists():
                    imagens = list(images_path.glob("*.[jJ][pP][gG]")) + list(images_path.glob("*.[pP][nN][gG]"))
                    print(f"   📂 {split}/images: {len(imagens)} imagens")
                    estatisticas['affectnet'][split] = len(imagens)
                
                if labels_path.exists():
                    labels = list(labels_path.glob("*.txt"))
                    print(f"   📂 {split}/labels: {len(labels)} labels")
    
    # 2. CKPLUS
    ckplus_path = BASE_PATH / "ckplus"
    if ckplus_path.exists():
        print("\n📁 CKPLUS (8 EMOÇÕES):")
        estatisticas['ckplus'] = {}
        emocoes_ck = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
        for emocao in emocoes_ck:
            pasta = ckplus_path / emocao
            if pasta.exists():
                imagens = list(pasta.glob("*.[jJ][pP][gG]")) + list(pasta.glob("*.[pP][nN][gG]"))
                print(f"   📂 {emocao}: {len(imagens)} imagens")
                estatisticas['ckplus'][emocao] = len(imagens)
    
    # 3. FERPLUS (já organizado por split e emoção!)
    ferplus_path = BASE_PATH / "ferplus"
    if ferplus_path.exists():
        print("\n📁 FERPLUS (ORGANIZADO POR SPLIT):")
        estatisticas['ferplus'] = {'train': {}, 'validation': {}, 'test': {}}
        
        for split in ['train', 'validation', 'test']:
            split_path = ferplus_path / split
            if split_path.exists():
                print(f"   📂 {split}:")
                emocoes_fer = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
                
                for emocao in emocoes_fer:
                    pasta_emocao = split_path / emocao
                    if pasta_emocao.exists():
                        imagens = list(pasta_emocao.glob("*.[jJ][pP][gG]")) + list(pasta_emocao.glob("*.[pP][nN][gG]"))
                        print(f"      📁 {emocao}: {len(imagens)} imagens")
                        estatisticas['ferplus'][split][emocao] = len(imagens)
    
    # 4. KDEF
    kdef_path = BASE_PATH / "kdef"
    if kdef_path.exists():
        print("\n📁 KDEF (7 EMOÇÕES):")
        estatisticas['kdef'] = {}
        emocoes_kdef = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        for emocao in emocoes_kdef:
            pasta = kdef_path / emocao
            if pasta.exists():
                imagens = list(pasta.glob("*.[jJ][pP][gG]")) + list(pasta.glob("*.[pP][nN][gG]"))
                print(f"   📂 {emocao}: {len(imagens)} imagens")
                estatisticas['kdef'][emocao] = len(imagens)
    
    # 5. RAF-DB (com CSVs)
    rafd_path = BASE_PATH / "rafdb"
    if rafd_path.exists():
        print("\n📁 RAF-DB (COM LABELS EM CSV):")
        estatisticas['rafd'] = {'train': 0, 'test': 0}
        
        # Ler CSVs
        train_csv = rafd_path / "train_labels.csv"
        test_csv = rafd_path / "test_labels.csv"
        
        if train_csv.exists():
            import pandas as pd
            df_train = pd.read_csv(train_csv)
            print(f"   📊 train_labels.csv: {len(df_train)} linhas")
            print(f"      Colunas: {list(df_train.columns)}")
            print(f"      Distribuição de labels:")
            print(df_train.iloc[:, 1].value_counts().sort_index())
            estatisticas['rafd']['train'] = len(df_train)
        
        if test_csv.exists():
            df_test = pd.read_csv(test_csv)
            print(f"   📊 test_labels.csv: {len(df_test)} linhas")
            print(f"      Colunas: {list(df_test.columns)}")
            print(f"      Distribuição de labels:")
            print(df_test.iloc[:, 1].value_counts().sort_index())
            estatisticas['rafd']['test'] = len(df_test)
        
        # Verificar estrutura de pastas
        dataset_path = rafd_path / "DATASET"
        if dataset_path.exists():
            for split in ['train', 'test']:
                split_path = dataset_path / split
                if split_path.exists():
                    print(f"   📂 DATASET/{split}:")
                    pastas = sorted([p for p in split_path.glob("*") if p.is_dir()])
                    for pasta in pastas[:3]:  # Mostrar apenas algumas
                        imagens = list(pasta.glob("*.[jJ][pP][gG]")) + list(pasta.glob("*.[pP][nN][gG]"))
                        print(f"      📁 {pasta.name}: {len(imagens)} imagens")
    
    # RESUMO
    print("\n" + "="*60)
    print("📊 RESUMO GERAL - TOTAL DE IMAGENS POR DATASET:")
    print("="*60)
    
    total_geral = 0
    
    # AffectNet
    total_aff = sum(estatisticas.get('affectnet', {}).values())
    print(f"AFFECTNET: {total_aff:,} imagens (train/valid/test)")
    total_geral += total_aff
    
    # CKPlus
    total_ck = sum(estatisticas.get('ckplus', {}).values())
    print(f"CKPLUS: {total_ck:,} imagens (8 classes)")
    total_geral += total_ck
    
    # FERPlus
    total_fer = 0
    if 'ferplus' in estatisticas:
        for split in estatisticas['ferplus'].values():
            total_fer += sum(split.values())
    print(f"FERPLUS: {total_fer:,} imagens (train/validation/test)")
    total_geral += total_fer
    
    # KDEF
    total_kdef = sum(estatisticas.get('kdef', {}).values())
    print(f"KDEF: {total_kdef:,} imagens (7 classes)")
    total_geral += total_kdef
    
    # RAF-DB
    total_rafd = sum(estatisticas.get('rafd', {}).values())
    print(f"RAF-DB: {total_rafd:,} imagens (train/test)")
    total_geral += total_rafd
    
    print("-"*60)
    print(f"🔥 TOTAL COMBINADO: {total_geral:,} imagens!")
    
    return estatisticas

if __name__ == "__main__":
    estatisticas = verificar_estrutura_refinada()