import time
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

def end_log():
    log["end_time"] = time.strftime("%Y-%m-%d_%H-%M-%S")