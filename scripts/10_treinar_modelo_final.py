# scripts/10_treinar_modelo_final_corrigido.py

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def corrigir_data_yaml():
    """Corrige os paths no data.yaml para absolutos"""
    
    yaml_path = Path("../data/unified_yolo/data.yaml")
    base_path = Path("C:/Users/henri/Documents/IA_03FINAL/data/unified_yolo")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Converter para caminhos absolutos
    config['train'] = str(base_path / 'train' / 'images')
    config['val'] = str(base_path / 'valid' / 'images')
    config['test'] = str(base_path / 'test' / 'images')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ data.yaml corrigido com caminhos absolutos")
    return config

def verificar_gpu():
    """Verifica se GPU está disponível"""
    if torch.cuda.is_available():
        print(f"✅ GPU disponível: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️  GPU não disponível. Usando CPU (será MUITO lento!)")
        print("   Recomendo usar Google Colab ou Kaggle para treinar")
        return 'cpu'

def treinar_modelo():
    """Treina o modelo com paths corrigidos"""
    
    print("="*60)
    print("🚀 TREINAMENTO DO MODELO FINAL (CORRIGIDO)")
    print("="*60)
    
    # Corrigir data.yaml
    config = corrigir_data_yaml()
    
    # Verificar GPU
    device = verificar_gpu()
    
    if device == 'cpu':
        print("\n⚠️  ATENÇÃO: Treinamento em CPU levará dias!")
        resposta = input("Continuar mesmo assim? (s/n): ")
        if resposta.lower() != 's':
            print("Treinamento cancelado.")
            return
    
    # Carregar modelo
    print("\n📦 Carregando YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # Treinar
    print("\n⚙️ Iniciando treinamento...")
    
    results = model.train(
        data=str(Path("../data/unified_yolo/data.yaml").absolute()),
        epochs=200,
        batch=16 if device == 'cpu' else 64,  # Batch menor para CPU
        imgsz=224,
        device=device,
        project="psico_ai_final",
        name="yolov8n_affectnet",
        exist_ok=True,
        plots=True,
        save=True,
        workers=4 if device == 'cpu' else 8,
    )
    
    print("\n✅ Treinamento concluído!")
    return model

if __name__ == "__main__":
    treinar_modelo()