import math
import sys
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from helper import Timer, eval_class, eval_total

from model_class import Net
import settings
from settings import *

import gc

def main():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
        transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
        ])

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(32, scale=(0.8, 1.1), ratio=(0.75, 1.333333)),
    #     transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
    #     transforms.RandomRotation(degrees=(-20, 20)),  #影象一半的概率翻轉，一半的概率不翻轉
    #     transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
    #     ])

    # transform_train = transforms.Compose([
    #     transforms.Resize(size=(224, 224)),
    #     transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
    #     ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    timer = Timer()
    
    def preprocess():

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
        # Return iterable which contains data in blocks, block size equals to batch size
        return trainloader, testloader
    

    trainloader, testloader = preprocess()


    model = Net()
    
    if load_model:
        model.load_state_dict(torch.load(model_dir + model_name))
    
    # del model._modules["resnet18"]
    # torch.save(model.state_dict(), model_dir + "87_92_1.pth")
    # return

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    model.to(device)
    
    # Start timer from here
    timer.timeit()
    if load_model:
        print(f"Verifying loaded model ({model_name})'s accuracy as its name suggested...")
        eval_total(model, testloader, timer, 12)
    else:
        print("Start training!")
    
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=learning_rate)
    opt2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = settings.opt_use_adam
    
    for epoch in range(epochs):
        optimizer = opts[int(opt_use_adam)]
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item() * inputs.shape[0]
            count_log = int(math.ceil(data_size/batch_size) / n_batch_logs_per_epoch) - 1
            if i % count_log == count_log - 1:
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}(batches)] loss: {running_loss / count_log:.3f}')
                running_loss = 0.0

        if adam_sgd_switch:
            if epoch % epochs_per_switch == 0:
                opt_use_adam = not opt_use_adam
                print(f"Epoch {epoch + 1}: Opt switched to {'Adam' if opt_use_adam else 'SGD'}")
        
        if epoch % epochs_per_eval == 0:
            eval_total(model, testloader, timer, epoch)

    print('Training Finished!')

    eval_total(model, testloader, timer)
    eval_class(model, testloader)


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        main()
    except KeyboardInterrupt:
        print("Exit!")