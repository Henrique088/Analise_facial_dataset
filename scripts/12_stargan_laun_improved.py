# scripts/12_stargan_laun_improved.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
from collections import defaultdict
import os
import random

# Criar diretório models se não existir
os.makedirs("../models", exist_ok=True)

class FaceDataset(Dataset):
    """Dataset para treinar StarGAN com imagens reais"""
    def __init__(self, class_images, img_size=128):  # Reduzido para 128
        self.img_paths = []
        self.labels = []
        self.img_size = img_size
        
        # Coletar paths
        for classe, paths in class_images.items():
            for path in paths[:5000]:  # Limitar para 5000 por classe
                self.img_paths.append(path)
                self.labels.append(classe)
        
        # Transformações
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformações
        img_tensor = self.transform(img)
        
        return img_tensor, self.labels[idx]

class ResidualBlock(nn.Module):
    """Bloco residual básico"""
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
    """Gerador simplificado (mais estável)"""
    def __init__(self, num_domains=7, img_size=128):
        super().__init__()
        
        # Encoder
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
        
        # Residual blocks
        res_blocks = []
        for _ in range(6):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Decoder
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
        # Concatenar imagem e labels
        labels_map = labels.view(-1, labels.size(1), 1, 1)
        labels_map = labels_map.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels_map], dim=1)
        
        # Passar pela rede
        x = self.enc(x)
        x = self.res_blocks(x)
        x = self.dec(x)
        
        return x

class Discriminator(nn.Module):
    """Discriminador simplificado"""
    def __init__(self, num_domains=7, img_size=128):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.01),
        )
        
        # Saída dual
        self.fc_adv = nn.Conv2d(512, 1, 3, 1, 1)
        self.fc_cls = nn.Conv2d(512, num_domains, 3, 1, 1)
    
    def forward(self, x):
        features = self.conv(x)
        adv = self.fc_adv(features)
        cls = self.fc_cls(features)
        return adv, cls

class StarGAN:
    """StarGAN simplificado e estável"""
    def __init__(self, num_domains=7, img_size=128):
        self.num_domains = num_domains
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"📦 Dispositivo: {self.device}")
        
        # Modelos
        self.G = Generator(num_domains, img_size).to(self.device)
        self.D = Discriminator(num_domains, img_size).to(self.device)
        
        # Losses
        self.adv_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.L1Loss()
        
        # Otimizadores com learning rate menor
        self.opt_G = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Pesos
        self.lambda_cls = 1
        self.lambda_rec = 10
    
    def create_labels(self, labels):
        """Cria one-hot encoding para labels"""
        batch_size = labels.size(0)
        labels_onehot = torch.zeros(batch_size, self.num_domains, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return labels_onehot
    
    def train_step(self, real, labels):
        """Um passo de treinamento com validação de estabilidade"""
        batch_size = real.size(0)
        
        # Mover para device
        real = real.to(self.device)
        labels = labels.to(self.device)
        
        # Gerar labels aleatórios para target
        target_labels = torch.randint(0, self.num_domains, (batch_size,)).to(self.device)
        target_onehot = self.create_labels(target_labels)
        
        # 1. Treinar Discriminador
        self.opt_D.zero_grad()
        
        # Real images
        d_adv_real, d_cls_real = self.D(real)
        d_adv_real = d_adv_real.mean()
        
        loss_d_real_adv = self.adv_loss(d_adv_real, torch.ones_like(d_adv_real) * 0.9)  # Label smoothing
        loss_d_real_cls = self.cls_loss(d_cls_real.mean(dim=[2,3]), labels)
        
        # Fake images
        with torch.no_grad():
            fake = self.G(real, target_onehot)
        
        d_adv_fake, _ = self.D(fake)
        d_adv_fake = d_adv_fake.mean()
        loss_d_fake_adv = self.adv_loss(d_adv_fake, torch.zeros_like(d_adv_fake) * 0.1)  # Label smoothing
        
        loss_d = (loss_d_real_adv + loss_d_fake_adv) * 0.5 + loss_d_real_cls * self.lambda_cls
        
        # Verificar NaN
        if torch.isnan(loss_d):
            print("⚠️ NaN detected in D loss, skipping...")
            return None
        
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)  # Gradient clipping
        self.opt_D.step()
        
        # 2. Treinar Generator
        self.opt_G.zero_grad()
        
        # Gerar imagens falsas
        fake = self.G(real, target_onehot)
        
        # Adversarial loss
        d_adv_fake, d_cls_fake = self.D(fake)
        d_adv_fake = d_adv_fake.mean()
        loss_g_adv = self.adv_loss(d_adv_fake, torch.ones_like(d_adv_fake) * 0.9)
        
        # Classification loss
        loss_g_cls = self.cls_loss(d_cls_fake.mean(dim=[2,3]), target_labels)
        
        # Reconstruction loss
        recon = self.G(fake, self.create_labels(labels))
        loss_rec = self.rec_loss(recon, real)
        
        loss_g = loss_g_adv + loss_g_cls * self.lambda_cls + loss_rec * self.lambda_rec
        
        # Verificar NaN
        if torch.isnan(loss_g):
            print("⚠️ NaN detected in G loss, skipping...")
            return None
        
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)  # Gradient clipping
        self.opt_G.step()
        
        return {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'loss_rec': loss_rec.item(),
            'd_adv_real': d_adv_real.item(),
            'd_adv_fake': d_adv_fake.item()
        }

def prepare_data_for_gan():
    """Prepara dados para treinar StarGAN"""
    unified_path = Path("../data/unified_yolo")
    
    # Coletar imagens por classe
    class_images = defaultdict(list)
    
    for split in ['train']:
        labels_path = unified_path / split / 'labels'
        images_path = unified_path / split / 'images'
        
        if not labels_path.exists():
            continue
            
        for label_file in tqdm(list(labels_path.glob("*.txt")), desc=f"Carregando {split}"):
            try:
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        classe = int(first_line.split()[0])
                        
                        # Procurar imagem
                        for ext in ['.jpg', '.png', '.jpeg']:
                            img_file = images_path / f"{label_file.stem}{ext}"
                            if img_file.exists():
                                class_images[classe].append(str(img_file))
                                break
            except:
                continue
    
    # Filtrar classes minoritárias
    target_classes = [2, 4, 5]  # Medo, Tristeza, Surpresa
    
    print("\n📊 Imagens disponíveis por classe:")
    for classe in target_classes:
        print(f"   Classe {classe}: {len(class_images[classe])} imagens")
    
    return class_images

def train_stargan():
    """Treina o StarGAN com validação de estabilidade"""
    print("="*60)
    print("🚀 TREINANDO STARGAN PARA AUGMENTATION")
    print("="*60)
    
    # Preparar dados
    class_images = prepare_data_for_gan()
    
    # Filtrar apenas classes alvo
    target_classes = [2, 4, 5]
    filtered_images = {k: v for k, v in class_images.items() if k in target_classes}
    
    # Criar dataset
    dataset = FaceDataset(filtered_images, img_size=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    print(f"\n📦 Dataset size: {len(dataset)} imagens")
    print(f"   Batches por época: {len(dataloader)}")
    
    # Inicializar modelo
    stargan = StarGAN(num_domains=7, img_size=128)
    
    # Treinar
    print("\n⚙️ Iniciando treinamento...")
    
    for epoch in range(50):  # 50 épocas
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Época {epoch}")
        for batch_idx, (real, labels) in enumerate(pbar):
            losses = stargan.train_step(real, labels)
            
            if losses is not None:
                epoch_losses.append(losses)
                
                # Atualizar barra de progresso
                pbar.set_postfix({
                    'D': f"{losses['loss_d']:.4f}",
                    'G': f"{losses['loss_g']:.4f}",
                    'rec': f"{losses['loss_rec']:.4f}"
                })
        
        # Mostrar média da época
        if epoch_losses:
            avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
            print(f"\n📊 Época {epoch} - D: {avg_losses['loss_d']:.4f}, G: {avg_losses['loss_g']:.4f}, Rec: {avg_losses['loss_rec']:.4f}")
            
            # Salvar checkpoint
            if epoch % 10 == 0:
                torch.save(stargan.G.state_dict(), f"../models/stargan_generator_epoch{epoch}.pth")
                print(f"   ✅ Modelo salvo: stargan_generator_epoch{epoch}.pth")
    
    # Salvar modelo final
    torch.save(stargan.G.state_dict(), "../models/stargan_generator_final.pth")
    print("\n✅ Modelo final salvo em: ../models/stargan_generator_final.pth")
    
    return stargan

if __name__ == "__main__":
    stargan = train_stargan()