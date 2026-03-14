# Criar um arquivo diagnostic.py e executar
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"CUDA versão: {torch.version.cuda}")
print(f"PyTorch build: {torch._C._cuda_getCompiledVersion() if hasattr(torch._C, '_cuda_getCompiledVersion') else 'N/A'}")