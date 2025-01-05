# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:02:27 2022

@author: admin
"""

import torch
import torchvision
import numpy as np
import utils
#import bearing_cls

from torch.utils.data import DataLoader

from dataset import BridgeDataset

#from matplotlib import pyplot as plt

root = "./output_24_3/11_05_10_37/"



# Load the model and train the weights
model = torchvision.models.shufflenet_v2_x1_0(pretrained=True) #False  True



model.fc = torch.nn.Linear(model.fc.in_features, 2)  # shufflenet_v2_x0_5, shufflenet_v2_x1_0
#model.fc = torch.nn.Linear(model.fc.in_features, 2) #resnet50
#model.classifier._modules['1'] = torch.nn.Linear(1280, 2)  # mobilenet_v2
#model.classifier._modules['3'] = torch.nn.Linear(1024, 2)  # mobilenet_v3_small
#model.classifier._modules['3'] = torch.nn.Linear(1280, 2)  # mobilenet_v3_large

#model.classifier._modules['1']=torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # squeezenet1_0, squeezenet1_1


#model.classifier._modules['6'] = torch.nn.Linear(4096, 2)  # vgg


#resnet50 prediction
# model = torchvision.models.resnet50(pretrained=True)
#model.fc = torch.nn.Linear(model.fc.in_features, 2) #resnet50,googlenet、inception_v3
#model.classifier._modules['6'] = torch.nn.Linear(4096, 2)  # vgg

model.load_state_dict(torch.load(f"{root}/best.pth"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#test
test_set = BridgeDataset('./datas/images', './datas/labels/test.txt')
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

preds_all = []
labels_all = []

model.eval()
for i, (images, labels) in enumerate(test_loader):
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #_, predicted = torch.max(outputs.logits.data, 1) #没有加载预训练时候的写法,只限于Googlenet
        predicted = predicted.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        
        preds_all += predicted
        labels_all += labels
        
preds_all = np.array(preds_all)
labels_all = np.array(labels_all)

su = 0
for i in range(len(preds_all)):
    su = su + abs(preds_all[i] - labels_all[i])
    
pre_accuracy = 1 - su/len(preds_all)
print(pre_accuracy)



# import scipy
# import sklearn
# from sklearn.metrics import confusion_matrix
# conf = confusion_matrix(labels_all, preds_all)
# print(f"Confusion Matrix: \n{conf}")
# print(f"Accuracy: {np.sum(preds_all == labels_all) / len(labels_all)}")
# print(f"Precision: {conf[1, 1] / np.sum(conf[:, 1])}")
# print(f"Recall: {conf[1, 1] / np.sum(conf[1, :])}")