import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from helper import Timer, eval_class, eval_total

from model_class import Net
from preprocess import preprocessor
from settings import *

import gc

def main():


    timer = Timer()
    p = preprocessor(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    trainloader, testloader = p.get_loader()
    
    model = Net()
    
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR + MODEL_NAME))

    # Assuming that we are on a CUDA machine
    model.to(DEVICE)
    
    # Start timer from here
    timer.timeit()
    
    if LOAD_MODEL:
        print(f"Verifying loaded model ({MODEL_NAME})'s accuracy as its name suggested...")
        eval_total(model, testloader, timer)
    else:
        print("Start training!")
    
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    opt2 = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = OPT_USE_ADAM
    
    for epoch in range(TOTAL_EPOCHS):
        optimizer = opts[int(opt_use_adam)]
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item() * inputs.shape[0]
            
            count_log = int(len(trainloader) / N_LOGS_PER_EPOCH)
            # print(count_log, inputs.shape)
            if i % count_log == count_log - 1:
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}(batches)] loss: {running_loss / count_log:.3f}')
                running_loss = 0.0

        if ADAM_SGD_SWITCH:
            if epoch % EPOCHS_PER_SWITCH == 0:
                opt_use_adam = not opt_use_adam
                print(f"Epoch {epoch + 1}: Opt switched to {'Adam' if opt_use_adam else 'SGD'}")
        
        if epoch % EPOCHS_PER_EVAL == 0:
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