import os
import time

import torch
import torchvision
#from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from dataset import BridgeDataset

from torchstat import stat





model = torchvision.models.resnet101(pretrained = True) #False  True
# print(model)
#model.fc = torch.nn.Linear(model.fc.in_features, 2)  # shufflenet_v2_x0_5，shufflenet_v2_x1_0，shufflenet_v2_x1_5，shufflenet_v2_x2_0
#model.classifier._modules['1']=torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # squeezenet1_0/squeezenet1_1
#model.classifier._modules['6'] = torch.nn.Linear(4096, 2)  # vgg
#model.classifier._modules['1'] = torch.nn.Linear(1280, 2)  # mobilenet_v2
#model.classifier._modules['3'] = torch.nn.Linear(1280, 2)  # mobilenet_v3_large
#model.classifier._modules['3'] = torch.nn.Linear(1024, 2)  # mobilenet_v3_small
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # resnet
#model.fc = torch.nn.Linear(model.fc.in_features, 2)  # googlenet
#model.fc = torch.nn.Linear(model.fc.in_features, 2)  # inception_v3
# print(model)


#stat(model, (3, 256, 128))#View model complexity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


train_set = BridgeDataset('./datas/images', './datas/labels/train.txt')
valid_set = BridgeDataset('./datas/images', './datas/labels/valid.txt')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)

lr = []
epochs = 80
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

#scheduler = ExponentialLR(optimizer, gamma=0.9)          # Adjust the learning rate exponentially


month, day, hour, minute = time.strftime("%m,%d,%H,%M").split(",")
save_root = f"./output_24/{month}_{day}_{hour}_{minute}"
if not os.path.exists(save_root):
    os.mkdir(save_root)


train_info = f"{save_root}/train.txt"
valid_info = f"{save_root}/valid.txt"
epoch_info = f"{save_root}/epoch.txt"
with open(train_info, "w") as f:
    f.write("epoch,step,loss,acc\n")
with open(valid_info, "w") as f:
    f.write("epoch,step,loss,acc\n")
with open(epoch_info, "w") as f:
    f.write("epoch,train_loss,train_acc,valid_loss,valid_acc\n")


best_state = {
    "epoch": 0,
    "weight": model.state_dict(),
    "train_loss": 0,
    "train_acc": 0,
    "valid_loss": 0,
    "valid_acc": 0
}


for epoch in range(epochs):

    # training
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # lr.append(scheduler.get_last_lr()[0])
        # print(epoch, scheduler.get_last_lr()[0])
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #_, predicted = torch.max(outputs.logits.data, 1) #Googlenet without transfer learning
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = correct / total
        loss = criterion(outputs, labels)
        #loss = criterion(outputs.logits, labels) #Googlenet without transfer learning
        loss.backward()
        optimizer.step()
        # scheduler.step()            # update  learning rate
        with open(train_info, "a") as f:
            f.write(f"{epoch:d},{i:d},{loss.item():.4f},{acc:.4f}\n")
    print(
        f"Epoch {epoch:d} train loss: {loss.item():.4f}, acc: {acc:.4f}", end=", ")
    with open(epoch_info, "a") as f:
        f.write(f"{epoch:d},{loss.item():.4f},{acc:.4f},")

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            valid_acc = correct / total
            valid_loss = criterion(outputs, labels)
            #scheduler.step(valid_acc)
            with open(valid_info, "a") as f:
                f.write(
                    f"{epoch:d},{i:d},{valid_loss.item():.4f},{valid_acc:.4f}\n")
    print(f"valid loss: {valid_loss.item():.4f}, acc: {valid_acc:.4f}")
    with open(epoch_info, "a") as f:
        f.write(f"{valid_loss.item():.4f},{valid_acc:.4f}\n")

    # Update optimum accuracy
    if valid_acc > best_state["valid_acc"]:
        best_state["epoch"] = epoch
        best_state["weight"] = model.state_dict()
        best_state["train_loss"] = loss.item()
        best_state["train_acc"] = acc
        best_state["valid_loss"] = valid_loss.item()
        best_state["valid_acc"] = valid_acc


# save model
torch.save(best_state["weight"], f"{save_root}/best.pth")
torch.save(model.state_dict(), f"{save_root}/last.pth")
best_state["weight"] = model.__str__()
with open(f"{save_root}/best.txt", "w") as f:
    for k, v in best_state.items():
        # if not k == "weight":
        f.write(f"{k}: {v}\n")

print(best_state["valid_acc"])



