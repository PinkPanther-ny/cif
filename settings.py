# import os
# import sys
# import torch
# # ==============================================
# # GLOBAL SETTINGS

# BATCH_SIZE = 256
# LEARNING_RATE = 5e-6
# TOTAL_EPOCHS = 500

# OPT_USE_ADAM = True

# LOAD_MODEL = True
# MODEL_NAME = "89_98.pth"
# MODEL_SAVE_THRESHOLD = 89.6

# NUM_WORKERS = 4
# N_LOGS_PER_EPOCH = 3

# # ==============================================
# # SPECIAL SETTINGS
# EPOCHS_PER_EVAL = 2

# ADAM_SGD_SWITCH = False
# EPOCHS_PER_SWITCH = 5

# # ==============================================
# # NOT SUPPOSED TO BE CHANGED OFTENLY

# if len(sys.argv) != 2:
#     CUDA_N = 0
#     print("Run on GPU:0 by default")
# else:
#     CUDA_N = sys.argv[1]
#     print(f"Run on GPU:{CUDA_N}")

# DEVICE = torch.device(f'cuda:{CUDA_N}' if torch.cuda.is_available() else 'cpu')

# WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
# MODEL_DIR = WORKING_DIR + "/models/"
# DATA_DIR = WORKING_DIR + '/data/'
# CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
#         'dog', 'frog', 'horse', 'ship', 'truck')