#Import libraries and modules
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch

def get_data_loaders(data_dir, batch_size):
    #Define the locations of the data files
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size)
    
    #Return the data loaders and train dataset to save the checkpoint
    return train_loader, valid_loader, test_loader, train_data

#Scales, crops, and normalizes a PIL image for a PyTorch model
def process_image(img_path):
    image = Image.open(img_path)
    width, height = image.size
    
    # Resize the image
    if width < height:
        new_height = int(height * 256 / width)
        image = image.resize((256, new_height))
    else:
        new_width = int(width * 256 / height)
        image = image.resize((new_width, 256))
    
    # Crop the image
    left_side = (image.width - 224) / 2
    top_side = (image.height - 224) / 2
    right_side = left_side + 224
    bottom_side = top_side + 224
    image = image.crop((left_side, top_side, right_side, bottom_side))
    
    # Convert to numpy array and normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to tensor and return
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    #Returns the image as an Numpy array
    return tensor_image

#Imshow for Tensor
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#Display the image along with the top 5 classes
def display_results(probs, labels, img_path, cat_to_name):
    figure, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=1, nrows=2)
    max_index = np.argmax(probs)
    categories = []

    #Get the image
    image = Image.open(img_path)

    ax1.axis('off')
    ax1.set_title(cat_to_name[labels[max_index]])
    ax1.imshow(image)

    #Save the catergories into a list
    for label in labels:
        categories.append(cat_to_name[label])

    #Define and label the axes
    ax2.barh(np.arange(5), probs)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(categories)
    ax2.set_title('Categories Probability')
    ax2.invert_yaxis()

    #Show the bar graph
    plt.show()

#Return the device which is available
def get_running_device(device):
    if torch.cuda.is_available() or device == "gpu":
        running_device = "cuda"
    else:
        running_device = "cpu"
        
    return running_device