import gc

import torch
import torch.utils.data
from dataset import *
from matplotlib import pyplot as plt
from model import *
from setup import *

if __name__ == '__main__':
    
    setup = Setup()
    device = setup.DEVICE
    
    print('Loading dataset...')
    train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
    train_dataset = Dataset(train_logo_paths, train_clean_paths, patches = True)
    val_dataset = Dataset(val_logo_paths, val_clean_paths, patches = True)
    train_loader = get_data_loader(train_dataset, batch_size = setup.BATCH)
    val_loader = get_data_loader(val_dataset, batch_size = setup.BATCH)

    print('Setting up the model...')
    generator = Generator().to(device)
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=list(generator.parameters()), lr = setup.GLR)

    print("Beginning training...")
    
    training_losses, val_losses = [], []

    for epoch in range(0, setup.EPOCHS):
        training_batch_losses, val_batch_losses = [], []
        
        for i, batch in enumerate(train_loader):
            generator.train()
            
            logos, cleans = batch[0], batch[1]
            
            if train_dataset.patches_bool:
                logos = torch.cat(logos, dim=0).to(device)
                cleans = torch.cat(cleans, dim=0).to(device)
            else:
                logos = logos.to(device)
                cleans = cleans.to(device)
            # logos, clenas : (BATCH*num_patches, 3, 256, 256) 

            t_loss = 0
            
            fake_images = generator.forward(logos).to(device)
            t_loss = criterion_mse(fake_images, cleans)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            if i % 200 == 0:
                print("T_Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f " % (epoch+1, setup.EPOCHS, i, len(train_loader), t_loss.item())) 
            
            t_loss = t_loss.to(torch.device("cpu"))
            training_batch_losses.append(t_loss) 
           
            
        for i, batch in enumerate(val_loader):
            generator.eval()
            with torch.no_grad():
            
                logos, cleans = batch[0], batch[1]
                
                if val_dataset.patches_bool:
                    logos = torch.cat(logos, dim=0).to(device)
                    cleans = torch.cat(cleans, dim=0).to(device)
                else:
                    logos = logos.to(device)
                    cleans = cleans.to(device)
                # logos, cleans : (BATCH*num_patches, 3, 256, 256) 
                
                v_loss = 0
                
                fake_images = generator.forward(logos).to(device)
                v_loss = criterion_mse(fake_images, cleans)                
                
                if i % 200 == 0:
                    print("V_Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f " % (epoch+1, setup.EPOCHS, i, len(val_loader), v_loss.item())) 
                
                v_loss = v_loss.to(torch.device("cpu"))
                val_batch_losses.append(v_loss) 
                
            
        # get the average results for each epoch
        training_losses.append(float(sum(training_batch_losses) / len(training_batch_losses)))
        val_losses.append(float(sum(val_batch_losses) / len(val_batch_losses)))
        
        torch.cuda.empty_cache()
        gc.collect()

        # save model after every epoch
        torch.save(generator.state_dict(), f"checkpoints/AUTOG-B{setup.BATCH}-G-{setup.GLR}-E{epoch+1}.pt")

    
    plt.plot(training_losses)
    plt.plot(val_losses)
    plt.title('Autoencoder Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"plots/AUTOG-B{setup.BATCH}-G-{setup.GLR}-E{setup.EPOCHS}.jpg")


