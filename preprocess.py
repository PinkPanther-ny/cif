import torch
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy
from config import configs

class preprocessor:

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(32, scale=(0.8, 1.1), ratio=(0.75, 1.333333)),
    #     transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
    #     ])
    
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧影象隨機裁剪成32*32
    #     transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
    #     ])


    
    def __init__(self, trans = None) -> None:
        
        self.data_dir = configs.DATA_DIR
        self.batch_size = configs.BATCH_SIZE
        self.n_workers = configs.NUM_WORKERS
        if trans is None:
            # self.transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
            #     transforms.RandomRotation(degrees=(-25, 25)),  #影象一半的概率翻轉，一半的概率不翻轉
            #     ])
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="constant"),  #先四周填充0，在吧影象隨機裁剪成32*32
                transforms.RandomHorizontalFlip(),  #影象一半的概率翻轉，一半的概率不翻轉
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #R,G,B每層的歸一化用到的均值和方差
                ])
            # self.transform_train = transforms.Compose([
            #     transforms.Resize(256),
            #     CIFAR10Policy(),
            #     transforms.RandomHorizontalFlip(),
            #     # transforms.Resize(32),
            #     transforms.ToTensor(),
            #     transforms.Resize(64),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # ])
        else:
            self.transform_train = trans
            
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(64),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
    def get_loader(self):

        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=self.transform_train)
        
        # 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
        #       sampler的原理，后面也会介绍。
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler)

        
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
        #                                         shuffle=True, num_workers=self.n_workers)

        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                            download=True, transform=self.transform_test)
        
        # 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
        #       sampler的原理，后面也会介绍。
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
        # testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=test_sampler)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
        
        # Return iterable which contains data in blocks, block size equals to batch size
        return trainloader, testloader