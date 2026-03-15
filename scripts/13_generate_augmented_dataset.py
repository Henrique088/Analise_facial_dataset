# scripts/13_generate_augmented_dataset.py

def augment_minority_classes(stargan_model, class_images, target_classes, samples_per_class=5000):
    """
    Gera imagens sintéticas para classes minoritárias
    """
    augmented_path = Path("../data/augmented_yolo")
    
    for split in ['train']:
        (augmented_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (augmented_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_generated = 0
    
    for classe in target_classes:
        print(f"\n📸 Gerando {samples_per_class} amostras para classe {classe}...")
        
        # Selecionar imagens base
        base_images = class_images[classe][:100]  # Usar 100 imagens reais como base
        
        generated = []
        for _ in range(samples_per_class // len(base_images) + 1):
            # Gerar variações
            batch = load_images(base_images)
            fake = stargan_model.generate_samples(batch, classe)
            generated.append(fake)
        
        # Salvar imagens geradas
        for i, img_tensor in enumerate(torch.cat(generated)[:samples_per_class]):
            img_np = ((img_tensor.numpy().transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)
            
            # Salvar imagem
            img_path = augmented_path / 'train' / 'images' / f"aug_classe{classe}_{i}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # Criar label (bounding box padrão)
            label_path = augmented_path / 'train' / 'labels' / f"aug_classe{classe}_{i}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{classe} 0.5 0.5 0.9 0.9\n")
            
            total_generated += 1
        
        print(f"   ✅ Geradas {samples_per_class} imagens para classe {classe}")
    
    print(f"\n🔥 Total de imagens geradas: {total_generated}")
    return augmented_path