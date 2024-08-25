import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from timm import create_model

def finetune(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.Grayscale(3),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = MNIST(root=args.data_location, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = create_model('deit_tiny_patch16_224', pretrained=True, num_classes=10)
    model = model.to(args.device)
    
    os.makedirs(args.save, exist_ok=True)
    pretrained_model_path = os.path.join(args.save, 'deit_tiny_pretrained.pth')
    torch.save(model.state_dict(), pretrained_model_path)
    print(f"Initial pretrained model saved to {pretrained_model_path}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    print(f"Starting fine-tuning on MNIST for {args.epochs} epochs")
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % args.print_every == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}")
    
    finetuned_model_path = os.path.join(args.save, 'deit_tiny_mnist_finetuned.pth')
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Fine-tuned model saved to {finetuned_model_path}")

if __name__ == '_main_':
    class Args:
        data_location = './data'
        batch_size = 128
        lr = 1e-4
        epochs = 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        save = './checkpoints'
        print_every = 100
    
    args = Args()
    finetune(args)