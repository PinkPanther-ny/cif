import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from find_best import load_best3, remove_bad_models

from helper import Timer, eval_total
from model_class import Net, Net1, Net2
from preprocess import preprocessor
from config import configs

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    local_rank = int(os.environ["LOCAL_RANK"])

    # DDP backend initialization
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # Load model to gpu
    device = torch.device("cuda", local_rank)
    
    model = Net()
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    map_location = 'cpu'
    
    timer = Timer()
    p = preprocessor()
    trainloader, testloader = p.get_loader()
    
    # Check if load specific model or load best model in model folder
    if configs.LOAD_MODEL:
        if configs.LOAD_BEST:
            configs.MODEL_NAME = load_best3(local_rank)
        try:
            model.load_state_dict(torch.load(configs.MODEL_DIR + configs.MODEL_NAME, map_location=map_location))
            if local_rank == 0:
                print(f"\nVerifying loaded model ({configs.MODEL_NAME})'s accuracy as its name suggested...")
                eval_total(model, testloader, timer, device)
        except FileNotFoundError:
            print(f"{configs.MODEL_NAME} Model not found!")

    
    # Start timer from here
    if local_rank == 0:
        timer.timeit()
        print(f"Start training! Total {configs.TOTAL_EPOCHS} epochs.\n")
    
    # Define loss function and optimizer for the following training process
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
    opt2 = optim.SGD(model.parameters(), lr=configs.LEARNING_RATE, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = configs.OPT_USE_ADAM
    
    # Mixed precision for speed up
    # https://zhuanlan.zhihu.com/p/165152789
    scalar = torch.cuda.amp.GradScaler()
    
    # ========================== Train =============================
    for epoch in range(configs.TOTAL_EPOCHS):
        
        # To avoid duplicated data sent to multi-gpu
        trainloader.sampler.set_epoch(epoch)
        
        # Just for removing worst models
        if local_rank == 0 and configs.LOAD_BEST and epoch % configs.EPOCH_TO_LOAD_BEST == 0:
            remove_bad_models()

        # By my stategy, chose optimizer dynamically
        optimizer = opts[int(opt_use_adam)]
        
        # Counter for printing information during training
        count_log = int(len(trainloader) / configs.N_LOGS_PER_EPOCH)
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Speed up with half precision
            with torch.cuda.amp.autocast():
                # forward + backward + optimize
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
            
            # Scale the gradient
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            
            # print statistics
            running_loss += loss.item() * inputs.shape[0]
            
            if local_rank == 0 and i % count_log == count_log - 1:
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}(batches)] loss: {running_loss / count_log:.3f}')
                running_loss = 0.0
                
        # Switch to another optimizer after some epochs
        if configs.ADAM_SGD_SWITCH:
            if epoch % configs.EPOCHS_PER_SWITCH == configs.EPOCHS_PER_SWITCH - 1:
                opt_use_adam = not opt_use_adam
                print(f"Epoch {epoch + 1}: Opt switched to {'Adam' if opt_use_adam else 'SGD'}")
        
        # Evaluate model on main GPU after some epochs
        if local_rank == 0 and epoch % configs.EPOCHS_PER_EVAL == configs.EPOCHS_PER_EVAL - 1:
            eval_total(model, testloader, timer, device, epoch)

    print(f'Training Finished! ({str(datetime.timedelta(seconds=int(timer.timeit())))})')


if __name__ == '__main__':
    try:
        gc.collect()
        torch.cuda.empty_cache()
        main()
    except KeyboardInterrupt:
        print("Exit!")