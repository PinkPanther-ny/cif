import torch
import torchvision
import torchvision.transforms as transforms

class preprocessor:

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
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
        transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __init__(self, data_dir, batch_size, n_workers) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
        
    def get_loader(self):

        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=self.n_workers)

        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                            download=True, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=self.n_workers)
        
        # Return iterable which contains data in blocks, block size equals to batch size
        return trainloader, testloader