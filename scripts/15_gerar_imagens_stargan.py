# scripts/15_gerar_imagens_stargan.py

import torch
import torch.nn as nn
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

# Criar diretórios
output_dir = Path("../data/augmented_yolo/train")
(output_dir / 'images').mkdir(parents=True, exist_ok=True)
(output_dir / 'labels').mkdir(parents=True, exist_ok=True)

# Classes alvo
TARGET_CLASSES = [2, 4, 5]  # Medo, Tristeza, Surpresa
EMOCOES = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutro']

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📦 Usando: {device}")

# Recriar arquitetura do Generator (igual ao treinamento)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class Generator(nn.Module):
    def __init__(self, num_domains=7, img_size=128):
        super().__init__()
        self.img_size = img_size
        
        self.enc = nn.Sequential(
            nn.Conv2d(3 + num_domains, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(6)])
        
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        labels_map = labels.view(-1, 7, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels_map], dim=1)
        x = self.enc(x)
        x = self.res_blocks(x)
        x = self.dec(x)
        return x

# Carregar modelo treinado
G = Generator().to(device)
G.load_state_dict(torch.load("../models/stargan_generator_final.pth"))
G.eval()
print("✅ Modelo carregado!")

# Carregar imagens base para gerar variações
base_images = []
base_path = Path("../data/unified_yolo/train/images")
all_images = list(base_path.glob("*.jpg")) + list(base_path.glob("*.png"))
base_images = all_images[:2000]  # Usar 2000 imagens base
print(f"📸 {len(base_images)} imagens base carregadas")

# Gerar imagens para cada classe alvo
target_per_class = 5000
generated_count = {c: 0 for c in TARGET_CLASSES}

print("\n🎨 Gerando imagens sintéticas...")
for target_class in TARGET_CLASSES:
    print(f"\n📁 Classe {target_class} ({EMOCOES[target_class]})")
    
    # Quantas gerar por imagem base
    samples_per_base = target_per_class // len(base_images) + 1
    
    pbar = tqdm(base_images, desc=f"Gerando {EMOCOES[target_class]}")
    for img_path in pbar:
        if generated_count[target_class] >= target_per_class:
            break
            
        # Carregar imagem base
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        
        # Converter para tensor
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float() / 127.5 - 1
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Criar label target
        target_label = torch.tensor([target_class]).to(device)
        target_onehot = torch.zeros(1, 7, device=device)
        target_onehot.scatter_(1, target_label.unsqueeze(1), 1)
        
        # Gerar imagem
        with torch.no_grad():
            for _ in range(samples_per_base):
                if generated_count[target_class] >= target_per_class:
                    break
                    
                # Adicionar ruído para variar
                noise = torch.randn(1, 3, 128, 128).to(device) * 0.05
                fake = G(img_tensor + noise, target_onehot)
                
                # Converter para numpy
                fake_img = fake[0].cpu().numpy().transpose(1,2,0)
                fake_img = (fake_img * 0.5 + 0.5) * 255
                fake_img = fake_img.astype(np.uint8)
                fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
                
                # Salvar imagem
                img_filename = f"stargan_{target_class}_{generated_count[target_class]:06d}.jpg"
                img_save_path = output_dir / 'images' / img_filename
                cv2.imwrite(str(img_save_path), fake_img)
                
                # Salvar label (bounding box padrão)
                label_filename = f"stargan_{target_class}_{generated_count[target_class]:06d}.txt"
                label_save_path = output_dir / 'labels' / label_filename
                with open(label_save_path, 'w') as f:
                    f.write(f"{target_class} 0.5 0.5 0.9 0.9\n")
                
                generated_count[target_class] += 1
                pbar.set_postfix({'geradas': generated_count[target_class]})

print("\n" + "="*60)
print("📊 RESUMO DA GERAÇÃO:")
print("="*60)
for c in TARGET_CLASSES:
    print(f"   {EMOCOES[c]}: {generated_count[c]} imagens geradas")
print(f"\n✅ Total: {sum(generated_count.values())} imagens")
print(f"📁 Salvas em: {output_dir}")