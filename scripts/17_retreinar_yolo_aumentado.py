# scripts/17_retreinar_yolo_aumentado_corrigido.py

from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    print("="*60)
    print("🚀 RETREINANDO YOLO COM DATASET AUMENTADO")
    print("="*60)
    
    # Verificar GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"📦 Dispositivo: {device}")
    if device == 0:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Configurações
    BATCH_SIZE = 64
    EPOCHS = 100
    IMG_SIZE = 224
    
    print(f"\n📊 Configurações:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Épocas: {EPOCHS}")
    print(f"   Imagem size: {IMG_SIZE}")
    
    # 1. CARREGAR MODELO
    print("\n📦 Carregando modelo...")
    best_previous = Path("psico_ai_final/yolov8n_affectnet/weights/best.pt")
    
    if best_previous.exists():
        model = YOLO(str(best_previous))
        print(f"   ✅ Carregado checkpoint anterior: {best_previous}")
    else:
        model = YOLO("yolov8n.pt")
        print(f"   ⚠️ Checkpoint não encontrado, usando modelo base")
    
    # 2. TREINAR COM DADOS AUMENTADOS
    print("\n⚙️ Iniciando treinamento com dados aumentados...")
    
    results = model.train(
        data="../data/unified_yolo/data.yaml",
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=0.0005,
        lrf=0.00005,
        warmup_epochs=2,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=2.0,
        flipud=0.05,
        fliplr=0.3,
        mosaic=0.8,
        mixup=0.1,
        device=device,
        workers=8,  # MUDANÇA IMPORTANTE: workers=0 para Windows!
        project="psico_ai_augmented",
        name="yolov8n_augmented",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("="*60)
    
    # 3. AVALIAR NO CONJUNTO DE TESTE
    print("\n📊 Avaliando no conjunto de teste...")
    metrics = model.val(data="../data/unified_yolo/data.yaml", split='test')
    
    print(f"\n📈 RESULTADOS FINAIS:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precisão: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # 4. COMPARAR COM RESULTADO ANTERIOR
    print("\n📊 COMPARAÇÃO COM MODELO ANTERIOR:")
    print(f"   Anterior (85.1%): 0.851 mAP50")
    print(f"   Novo: {metrics.box.map50:.4f} mAP50")
    
    if metrics.box.map50 > 0.851:
        print(f"   ✅ GANHO DE {((metrics.box.map50 - 0.851) * 100):.1f}%!")
    else:
        print(f"   📊 Similar ao modelo anterior")

if __name__ == '__main__':
    # Necessário para Windows evitar o erro de multiprocessing
    main()