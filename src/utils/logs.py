import time
import json
import os

log = {
    "device": str,
    "start_time": str,
    "end_time": str,
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "lr": []
}

def init_log(device):
    log["device"] = device
    log["start_time"] = time.strftime("%Y-%m-%d_%H-%M-%S")

def log_train(loss, accuracy, lr):
    log["train_loss"].append(loss)
    log["train_accuracy"].append(accuracy)
    log["lr"].append(lr)

def log_val(loss, accuracy):
    log["val_loss"].append(loss)
    log["val_accuracy"].append(accuracy)

def end_log(log_dir):
    log["end_time"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    filename = f"training_log_{log['start_time']}.json"
    log_path = os.path.join(log_dir, filename)
    
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
