import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detetada: {torch.cuda.get_device_name(0)}")
else:
    print("O PyTorch não vê a tua GPU. Verifica se tens os drivers da NVIDIA atualizados.")