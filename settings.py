# ==============================================
# GLOBAL SETTINGS
import os
import sys

import torch

if len(sys.argv) != 2:
    cuda_n = 0
    print("Run on GPU:0 by default")
else:
    cuda_n = sys.argv[1]
    print(f"Run on GPU:{cuda_n}")

device = torch.device(f'cuda:{cuda_n}' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 7
opt_use_adam = True
learning_rate = 5e-6
epochs = 500

load_model = True
model_name = "89_98.pth"
model_save_threshold = 89.6

# working_dir = "/datav/alvin/CIF/"
# model_dir = "/datav/alvin/CIF/models/"
working_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = working_dir + "/models/"
data_dir = working_dir + '/data/'

num_workers = 1

data_size = 60000
n_batch_logs_per_epoch = 3

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# ==============================================
# SPECIAL SETTINGS
epochs_per_eval = 2

adam_sgd_switch = True
epochs_per_switch = 1