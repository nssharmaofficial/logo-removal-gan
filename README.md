# TV_Logo_Removal
 
 
 ### Dataset
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
Dataset(train_logo_paths, train_clean_paths, patches = True)
```

When called, each pair of images  ```logo```, ```clean``` is normalized and returned as tensor (when ```patches = True```, list of tensors is returned and the length of the list represents number of patches for each image)

<br>

Run  ```dataset.py```  to load first iteration of dataloader and visualize the ```logo``` and ```clean``` images as well as to understand how the batch size and the number of patches change the dimension of returned tensors.
