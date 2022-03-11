import time
import torch
from config import configs

class Timer:
    def __init__(self):
        self.ini = time.time()
        self.last = 0
        self.curr = 0
        
    def timeit(self)->float:
        if self.last == 0 and self.curr == 0:
            self.last = time.time()
            self.curr = time.time()
            return 0, 0
        else:
            self.last = self.curr
            self.curr = time.time()
            return round(self.curr - self.last, 2), round(self.curr - self.ini, 2)


def eval_total(model, testloader, timer, device, epoch=-1):
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
            
    save_model = 100 * correct / total >= configs.MODEL_SAVE_THRESHOLD
    print(f"{'''''' if epoch==-1 else '''Epoch ''' + str(epoch) + ''': '''}Accuracy of the network on the {total} test images: {100 * correct / float(total)} % ({'saved' if save_model else 'discarded'})")
    t = timer.timeit()
    print(f"Delta time: {t[0]}, Already: {t[1]}\n")
    if save_model:
        torch.save(model.state_dict(), configs.MODEL_DIR + f"{100 * correct / total}".replace('.', '_') + '.pth')

    model.train()
