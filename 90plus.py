import torch

import torch.nn as nn
import torch.optim as optim
import gc
from find_best import load_best3

from helper import Timer, eval_total
from model_class import Net, Net2
from preprocess import preprocessor
from config import configs


def main():

    timer = Timer()

    
    p = preprocessor()
    trainloader, testloader = p.get_loader()
    
    model = Net()
    
    try:
        if configs.LOAD_MODEL:
            model.load_state_dict(torch.load(configs.MODEL_DIR + configs.MODEL_NAME))
            # Assuming that we are on a CUDA machine
            model.to(torch.device(configs.DEVICE))
            print(f"\nVerifying loaded model ({configs.MODEL_NAME})'s accuracy as its name suggested...")
            eval_total(model, testloader, timer)
    except FileNotFoundError:
        print(f"{configs.MODEL_NAME} Model not found!")

    model.to(torch.device(configs.DEVICE))
    # Start timer from here
    timer.timeit()
    
    print(f"Start training! Total {configs.TOTAL_EPOCHS} epochs.\n")
    
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
    opt2 = optim.SGD(model.parameters(), lr=configs.LEARNING_RATE, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = configs.OPT_USE_ADAM
    
    # ========================== Train =============================
    for epoch in range(configs.TOTAL_EPOCHS):
        
        if epoch % configs.EPOCH_TO_LOAD_BEST == 0:
            best_model = load_best3()
            if best_model != '':
                model.load_state_dict(torch.load(configs.MODEL_DIR + best_model, map_location=torch.device("cpu")))
                model.to(torch.device(configs.DEVICE))
                print(f"Verifying loaded model ({best_model})'s accuracy as its name suggested...")
                eval_total(model, testloader, timer)
        
        optimizer = opts[int(opt_use_adam)]
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(torch.device(configs.DEVICE)))
            loss = criterion(outputs, labels.to(torch.device(configs.DEVICE)))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item() * inputs.shape[0]
            
            count_log = int(len(trainloader) / configs.N_LOGS_PER_EPOCH)
            if i % count_log == count_log - 1:
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}(batches)] loss: {running_loss / count_log:.3f}')
                running_loss = 0.0

        if configs.ADAM_SGD_SWITCH:
            if epoch % configs.EPOCHS_PER_SWITCH == configs.EPOCHS_PER_SWITCH - 1:
                opt_use_adam = not opt_use_adam
                print(f"Epoch {epoch + 1}: Opt switched to {'Adam' if opt_use_adam else 'SGD'}")
        
        if epoch % configs.EPOCHS_PER_EVAL == 0:
            eval_total(model, testloader, timer, epoch)

    print('Training Finished!')

    eval_total(model, testloader, timer)


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        main()
    except KeyboardInterrupt:
        print("Exit!")