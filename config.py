import os
import sys
import torch
import json

class Config:
    def __init__(self, *dict_config) -> None:
        # ==============================================
        # GLOBAL SETTINGS

        self.BATCH_SIZE:int = 256
        self.LEARNING_RATE:float = 1e-4
        self.TOTAL_EPOCHS:int = 400

        self.OPT_USE_ADAM:bool = True

        self.LOAD_MODEL:bool = False
        self.MODEL_NAME:str = "56_19.pth"
        self.EPOCH_TO_LOAD_BEST:int = 15
        
        self.MODEL_SAVE_THRESHOLD:float = 0


        self.NUM_WORKERS:int = 4
        self.N_LOGS_PER_EPOCH:int = 3

        # ==============================================
        # SPECIAL SETTINGS
        self.EPOCHS_PER_EVAL:int = 2

        self.ADAM_SGD_SWITCH:bool = True
        self.EPOCHS_PER_SWITCH:int = 15

        # ==============================================
        # NOT SUPPOSED TO BE CHANGED OFTENLY

        if len(sys.argv) != 2:
            self.CUDA_N:int = 0
            print("Run on GPU:0 by default")
        else:
            self.CUDA_N:int = int(sys.argv[1])
            print(f"Run on GPU:{self.CUDA_N}")

        self.DEVICE:str = f'cuda:{self.CUDA_N}' if torch.cuda.is_available() else 'cpu'

        self.WORKING_DIR:str = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_DIR:str = self.WORKING_DIR + "/models_64/"
        self.DATA_DIR:str = self.WORKING_DIR + '/data/'
        self.CLASSES:list = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
        
        if len(dict_config)!=0:
            d = eval(dict_config[0])
            for k in dict(d):
                setattr(self, k, d[k])

        
        
    def save(self, fn='/config.json'):
        with open(self.WORKING_DIR + fn, 'w') as fp:
            json.dump(str(self.__dict__), fp, indent=4)
            
    def load(self, fn='/config.json'):
        try:
            
            with open(self.WORKING_DIR + fn, 'r') as fp:
                dict_config = json.load(fp)
                d = eval(dict_config)
                for k in dict(d):
                    setattr(self, k, d[k])
            print("Config file loaded successfully!")
        except:
            print("Config file does not exits, use default value instead!")
            
configs = Config()
configs.load()
# configs.save()