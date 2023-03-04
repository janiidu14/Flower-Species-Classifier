#Import libraries and modules
import torch
import util_functions
from torch import nn, optim
from torchvision import models

import util_functions

def cutomize_model(arch, input_size, output_size, hidden_units, drop_out, learn_rate, device="gpu"):
    running_device = util_functions.get_running_device(device)
    
    if arch == "vgg16":
        #Load a pre-trained network
        model = models.vgg16(pretrained=True)
    
    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                     nn.ReLU(), 
                                     nn.Dropout(drop_out),
                                     nn.Linear(hidden_units, output_size),
                                     nn.LogSoftmax(dim=1))

    #Assign custoomized classifier to model classifier
    model.classifier = classifier
    
    #Define the loss function
    criterion = nn.NLLLoss()

    #Train only the classifier with optimzer
    optimizer = optim.Adam(model.classifier.parameters(), learn_rate)

    #Send the model to the working device
    model.to(running_device)

    return model, criterion, optimizer

#Save the checkpoint 
def save_checkpoint(train_data, arch, model, path, optimizer, input_size, hidden_layers, output_size, learn_rate, epochs, drop_out, batch_size):
    model.class_to_idx = train_data.class_to_idx

    #Define the checkpoint of the model 
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'hidden_layers': hidden_layers,
                  'output_size': output_size,
                  'learn_rate': learn_rate,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'drop_out': drop_out, 
                  'classifier' : model.classifier,
                  'state_dict': model.state_dict(),
                  'model_class_to_index': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict()}

    #Save the checkpoint
    torch.save(checkpoint, path)
    print("Checkpoint Saved")

#Loads the checkpoint of the model
def load_checkpoint(path):
    #Load the checkpoint
    checkpoint = torch.load(path)
    
    #Assign the parameters to model
    arch = checkpoint['arch']
    input_size = checkpoint['input_size']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    learn_rate = checkpoint['learn_rate']
    epochs = checkpoint['epochs']
    batch_size = checkpoint['batch_size']
    drop_out = checkpoint['drop_out']
    
    model, criterion, optimizer = cutomize_model(arch, input_size, output_size, hidden_layers, drop_out, learn_rate)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_class_to_index']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #Return the model
    return model


#Predict the class (or classes) of an image using a trained deep learning model.
def predict(image_path, model, topk = 5, device="cpu"):    
    #Implement the code to predict the class from an image file
    idx_mapping = dict(map(reversed, model.class_to_idx.items()))
     
    #Get the device on which the program runs        
    running_device = util_functions.get_running_device(device)
    
    #Get the pre-processed image as a Tensor
    processed_image = util_functions.process_image(image_path)

    model.to(running_device)
    model.eval()
    
    with torch.no_grad():
        # Forward pass pre-processed image through the model
        output_image = model(torch.unsqueeze(processed_image,0).to(running_device).float())
        
        # Calculate the probabilities and classes
        probs = torch.exp(output_image)
        top_probs, top_classes = probs.topk(topk, dim=1)
        
        top_probs = top_probs.squeeze().tolist()
        top_classes = top_classes.squeeze().tolist()
        
        # Invert the dictionariy for index mapping
        class_to_idx = model.class_to_idx.items()
        idx_to_class = {value: key for key, value in class_to_idx}
        top_classes = [idx_to_class[index] for index in top_classes]

        return top_probs, top_classes
    
#Validation on the test set
def test_validation(model, test_loader, running_device, criterion):
    total_loss = 0
    test_accuracy = 0
    #Activate the evaluation mode
    model.eval()

    #Disable gradient descents for evaluation of the model
    for test_images, test_labels in test_loader:
        #Define the parameters to be used in the evaluation of the model
        test_images, test_labels = test_images.to(running_device), test_labels.to(running_device)
        test_log_probs = model.forward(test_images)                    
        test_loss = criterion(test_log_probs, test_labels)

        total_loss += test_loss

        #Define parameters to calculate accuracy
        #Convert the output to probabilities
        probs = torch.exp(test_log_probs)
        #Get the top classs in the probabilities
        top_prob, top_class = probs.topk(1, dim=1)
        #Check if the shape is in the correct dimensions
        dim_equals = top_class == test_labels.view(*top_class.shape)
        #Convert equals_dim from BitTensor to FloatTensor
        dim_equals = dim_equals.type(torch.FloatTensor)

        test_accuracy += torch.mean(dim_equals).item()

    #Display the results of the evaluation
    print(f"Test loss: {total_loss/len(test_loader):.3f}.. "
          f"Test accuracy: {test_accuracy/len(test_loader):.3f}")