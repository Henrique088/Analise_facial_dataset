# scripts/03_criar_estrutura_unificada.py

import os
import shutil
from pathlib import Path

# Caminhos
BASE_PATH = Path("../data")
RAW_PATH = BASE_PATH / "raw"
UNIFIED_PATH = BASE_PATH / "unified_yolo"

def criar_estrutura():
    """Cria a estrutura de pastas para o dataset unificado"""
    
    print("🔨 Criando estrutura unificada...")
    
    # Criar pastas principais
    for split in ['train', 'valid', 'test']:
        images_path = UNIFIED_PATH / split / 'images'
        labels_path = UNIFIED_PATH / split / 'labels'
        
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   ✅ Criado: {split}/images e {split}/labels")
    
    # Copiar data.yaml base do AffectNet
    affectnet_yaml = RAW_PATH / "affectnet" / "data.yaml"
    if affectnet_yaml.exists():
        # Modificar o data.yaml para refletir nosso dataset unificado
        import yaml
        
        with open(affectnet_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Atualizar paths para nosso novo diretório
        config['train'] = str(UNIFIED_PATH / 'train' / 'images')
        config['val'] = str(UNIFIED_PATH / 'valid' / 'images')
        config['test'] = str(UNIFIED_PATH / 'test' / 'images')
        config['nc'] = 7  # Vamos usar 7 classes
        
        # Ajustar nomes das classes (removendo Contempt)
        config['names'] = [
            "Raiva",
            "Nojo",      # Contempt será mapeado para Nojo
            "Medo",
            "Felicidade",
            "Tristeza",
            "Surpresa",
            "Neutro"
        ]
        
        # Salvar novo data.yaml
        with open(UNIFIED_PATH / 'data.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"   ✅ data.yaml criado com {config['nc']} classes")
        print(f"      Classes: {config['names']}")
    
    print(f"\n📁 Estrutura criada em: {UNIFIED_PATH}")
    return UNIFIED_PATH

if __name__ == "__main__":
    criar_estrutura()