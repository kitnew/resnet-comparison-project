import torch
import torch.nn as nn
from models.resnet import resnet101, resnet152
from models.model import ResNet
from utils.data_loader import create_data_loaders
from utils.logs import init_log, log_train, log_val, end_log

import argparse
import os
import time
import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_dir, "results")
start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

train_loader, val_loader, test_loader = create_data_loaders()
model: ResNet = None

def train_model(model, train_loader, val_loader, device, visualize=False, save_model=False):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(90):
        model.train()
        current_loss = 0
        current_acc = 0
        for images, labels in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            current_acc += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = current_loss / len(train_loader)
        train_acc = current_acc / len(train_loader.dataset)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, LR: {lr:.6f}")
        log_train(train_loss, train_acc, lr)


        lr_scheduler.step()

        model.eval()
        current_val_loss = 0
        current_val_acc = 0
        for images, labels in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            current_val_loss += loss.item()
            current_val_acc += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss = current_val_loss / len(val_loader)
        val_acc = current_val_acc / len(val_loader.dataset)

        logger.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        log_val(val_loss, val_acc)

def test_model(model, test_loader, device, visualize=False, save_model=False):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    if save_model:
        torch.save(model.state_dict(), f"{model.name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet training script")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode to run")
    parser.add_argument("--model", choices=["resnet101", "resnet152"], default="resnet101", help="Model to train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--save-model", action="store_true", help="Save model")

    args = parser.parse_args()

    init_log(args.device)

    if args.mode == "train":
        if args.model == "resnet101":
            model = resnet101()
        elif args.model == "resnet152":
            model = resnet152()
        train_model(model, train_loader, val_loader, args.device, args.visualize, args.save_model)
    elif args.mode == "test":
        if args.model == "resnet101":
            model = resnet101(pretrained=True)
        elif args.model == "resnet152":
            model = resnet152(pretrained=True)
        test_model(model, test_loader, args.device, args.visualize, args.save_model)
        
    end_log()
