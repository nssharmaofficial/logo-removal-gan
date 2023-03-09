# TV- channel logo removal
 
 
 <br>
 
 ## Dataset
- images with logo in folder ```'images/logo'``` have pattern: ```'i-j-k.jpg'```
- images without logo in folder ```'images/clean'``` have pattern: ```'i-j.jpg'```
- **Note**: one clean image is having multiple corresponding logo images

<br>

Paths to images are firstly divided into train and validation sets (70/30)
```python
# get lists of paths to corresponding images (split 70/30)
train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
```

<br>

Argument ```patches``` in ```Dataset``` class can define whether the images are divided into patches or not. 
```python
train_dataset = Dataset(train_logo_paths, train_clean_paths, patches = True)
```

When called, each pair of images  ```logo```, ```clean``` is normalized and returned as tensor (when ```patches = True```, list of tensors is returned and the length of the list represents number of patches for each image)

<br>

Run  ```dataset.py```  to load first iteration of dataloader and visualize the ```logo``` and ```clean``` images as well as to understand how the batch size and the number of patches change the dimension of returned tensors.

<br>

## Model 

Consists of:
- generator 
- discriminator


<br>

### Generator

U-net structure with 7 down-sampling convolutional layers (*encoder*), bottleneck and 7 up-sampling (*decoder*) transposed-convolutional layers with relu activation function followed by final convolution layer with tanh activation function. 

<br>

Dimensions after each layer can be seen below (where 20 represents the batch size):


```
Original:  torch.Size([20, 3, 256, 256])
D1:  torch.Size([20, 64, 128, 128])
D2:  torch.Size([20, 128, 64, 64])
D3:  torch.Size([20, 256, 32, 32])
D4:  torch.Size([20, 512, 16, 16])
D5:  torch.Size([20, 512, 8, 8])
D6:  torch.Size([20, 512, 4, 4])
D7:  torch.Size([20, 512, 2, 2])
B:  torch.Size([20, 512, 1, 1])
U1:  torch.Size([20, 512, 2, 2])
U2:  torch.Size([20, 512, 4, 4])
U3:  torch.Size([20, 512, 8, 8])
U4:  torch.Size([20, 512, 16, 16])
U5:  torch.Size([20, 256, 32, 32])
U6:  torch.Size([20, 128, 64, 64])
U7:  torch.Size([20, 64, 128, 128])
Out:  torch.Size([20, 3, 256, 256])
```

<br>

### Discriminator

Basic CNN with 4 convolutional layers with batch normalization and leaky relu activation function followed by final convolutional layer with a sigmoid activation function. 

<br>

Dimensions after each layer can be seen below (where 20 represents the batch size):

```
Original:  torch.Size([20, 3, 256, 256])
Conv1:  torch.Size([20, 64, 128, 128])
Conv2:  torch.Size([20, 128, 64, 64])
Conv3:  torch.Size([20, 256, 32, 32])
Conv4:  torch.Size([20, 512, 16, 16])
Conv5:  torch.Size([20, 1, 13, 13])
Out:  torch.Size([20, 1])
```

Run  ```model.py``` to perform one forward operations of both models and visualize input and output of the generator net.

<br>

## Training

There are two options:
- generator only
- GAN

<br>

### Generator only

The generator is trained to transform input *logo* images into output *generated* images that are compared with *clean* images by the ```nn.MSELoss()``` criterion. This loss allows the network to minimize image distortion and at the same time reduce the noise, i.e. watermark within the image.

As the image is processed from one layer to another it is also compressed in size. This allows the encoder network to extract most significant features from the image. As the encoder's capacity decreases it learns to disregard certain features and compress others. The decoder will then do the work backwards by rebuilding the image to its initial state without having the logo while maintaining the quality of the image. 

<br>









