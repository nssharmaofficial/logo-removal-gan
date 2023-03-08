import os
import torch

class Setup(object):
    
    def __init__(self) -> None:

        self.DEVICE = torch.device("cuda:0")
        
        self.clean_dir = os.path.join(os.path.expanduser('~'), 'NN_projects', 'LogoRemoval', 'images', 'clean') 
        self.logo_dir = os.path.join(os.path.expanduser('~'), 'NN_projects', 'LogoRemoval', 'images', 'logo') 
        self.patch_size = (256,256)
        self.whole_size = (512,512)
        
        self.BATCH = 2
        self.EPOCHS = 25
        
        self.GLR = 1e-4
        self.DLR = 4e-4
        self.LAMBDA = 200
        
        self.AUTO = False
        self.EPOCHS_model = 20
        self.BATCH_show = 10


        


        
        