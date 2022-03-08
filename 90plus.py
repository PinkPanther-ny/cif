import sys
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from helper import Timer

from model_class import Net
import settings
from settings import *

import gc

def main():

    # ==============================================
    # SETTINGS
    cuda_n = sys.argv[1]
    # opt_use_adam = True
    # learning_rate = 1e-5
    # # learning_rate = 0.000005
    # epochs = 500

    # load_model = True
    # model_name = "89_98.pth"
    # model_save_threshold = 88.8

    # # working_dir = "/datav/alvin/CIF/"
    # # model_dir = "/datav/alvin/CIF/models/"
    # working_dir = "./"
    # model_dir = "./models/"

    # num_workers = 4

    # ==============================================

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

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

        trainset = torchvision.datasets.CIFAR10(root=working_dir + 'data/', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=working_dir + 'data/', train=False,
                                            download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)

        return trainloader, testloader
    
    def eval_total(model, t=timer):
        model.eval()
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the
        # gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images.to(device))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.cpu().data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        PATH = model_dir
        save_model = 100 * correct / total >= model_save_threshold
        if save_model:
            torch.save(model.state_dict(), PATH + f"{100 * correct / total}".replace('.', '_') + '.pth')
        print(f"Accuracy of the network on the 10000 test images: {100 * correct / float(total)} % ({'saved' if save_model else 'discarded'})")
        t = timer.timeit()
        print(f"Delta time: {t[0]}, Already: {t[1]}")
        model.train()

    def eval_class(model):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.cuda())
                _, predictions = torch.max(outputs.cpu(), 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    trainloader, testloader = preprocess()


    model = Net()
    
    if load_model:
        model.load_state_dict(torch.load(model_dir + model_name))
    
    # del model._modules["resnet18"]
    # torch.save(model.state_dict(), model_dir + "87_92_1.pth")
    # return

    device = torch.device(f'cuda:{cuda_n}' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    model.to(device)

    if load_model:
        print(f"Verifying loaded model ({model_name})'s accuracy as its name suggested...")
        eval_total(model)
    else:
        print("Start training!")
    
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=learning_rate)
    opt2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = settings.opt_use_adam
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer = opts[int(opt_use_adam)]
        
        # running_loss = 0.0
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
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0
        # if epoch % 5 == 0:
        #     opt_use_adam = not opt_use_adam
        if epoch % 2 == 0:
            eval_total(model)

    print('Training Finished!')

    eval_total(model)
    eval_class(model)


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        main()
    except KeyboardInterrupt:
        print("Exit!")