import os
from config import configs
from random import randrange
from os import walk

def load_best3():
    files = next(walk(configs.MODEL_DIR), (None, None, []))[2]
    if len(files) == 0:
        return ''
    acc = sorted([float(i.split('.')[0].replace('_', '.')) for i in files], reverse=True)
    best_acc = acc[:3]
    
    for i in acc[3:]:
        try:
            os.remove(configs.MODEL_DIR + "/" + str(i).replace('.', '_') + ".pth")
        except:
            continue
            
        
    model_name = str(best_acc[randrange(3)] if len(acc[:3]) == 3 else best_acc[0]).replace('.', '_') + ".pth"
    print(f"Loading one of top 3 best model: {model_name}\n")
    return "/" + model_name