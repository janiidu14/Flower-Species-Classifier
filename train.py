#Import libraries and modules
import numpy as np
import argparse
import torch
from torch import nn, optim
from torchvision import models

import util_functions
import model_functions

#Setting up parser
parser = argparse.ArgumentParser(
    description='This is a parser for training the model',
)

parser.add_argument('data_dir', action="store", default="flowers")
parser.add_argument('--save_dir', action="store", default="vgg16_checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--gpu', action="store", default="gpu")

parser.add_argument('--learning_rate', action="store", default=0.001, type=float)
parser.add_argument('--hidden_units', action="store", default=4096, type=int)
parser.add_argument('--epochs', action="store", default=1, type=int)
parser.add_argument('--dropout', action="store", default=0.3, type=float)
parser.add_argument('--batch_size', action="store", default=64, type=int)
parser.add_argument('--input_size', action="store", default=25088, type=int)
parser.add_argument('--output_size', action="store", default=102, type=int)

args = parser.parse_args()

data_dir = args.data_dir
checkpoint_path = args.save_dir

arch = args.arch
device = args.gpu

#Set the hyperparameters recieved from the parser
learn_rate = args.learning_rate
hidden_layers = args.hidden_units
epochs = args.epochs
drop_out = args.dropout
batch_size = args.batch_size
input_size = args.input_size
output_size = args.output_size

def main():
    running_loss = 0
    evaluate_freq = 5
    steps = 0  
       
    train_loader, valid_loader, test_loader, train_data = util_functions.get_data_loaders(data_dir, batch_size)
    
    running_device = util_functions.get_running_device(device)
    
    model, criterion, optimizer = model_functions.cutomize_model(arch, input_size, output_size, hidden_layers, drop_out, learn_rate, device)
    
    #Train the model
    for epoch in range(epochs):
        for train_images, train_labels in train_loader:
            steps += 1
            # Move tensors to the running device
            train_images, train_labels = train_images.to(running_device), train_labels.to(running_device)

            #Clear the previous gradients 
            optimizer.zero_grad()

            #Get the log probabilities
            log_probs = model.forward(train_images)

            #Get the loss between labels and output
            loss = criterion(log_probs, train_labels)

            #Backpropagation
            loss.backward()
            optimizer.step()

            #Accumilate the running loss
            running_loss += loss.item()

            #Evaluate the model every 5 steps
            if steps % evaluate_freq == 0:
                eval_loss = 0
                accuracy = 0
                #Activate the evaluation mode
                model.eval()

                #Disable gradient descents for evaluation of the model
                with torch.no_grad():
                    for valid_images, valid_labels in valid_loader:
                        #Define the parameters to be used in the evaluation of the model
                        valid_images, valid_labels = valid_images.to(running_device), valid_labels.to(running_device)
                        valid_log_probs = model.forward(valid_images)                    
                        valid_loss = criterion(valid_log_probs, valid_labels)

                        eval_loss += valid_loss

                        #Define parameters to calculate accuracy
                        #Convert the output to probabilities
                        probs = torch.exp(valid_log_probs)
                        #Get the top classs in the probabilities
                        top_prob, top_class = probs.topk(1, dim=1)
                        #Check if the shape is in the correct dimensions
                        dim_equals = top_class == valid_labels.view(*top_class.shape)
                        #Convert equals_dim from BitTensor to FloatTensor
                        dim_equals = dim_equals.type(torch.FloatTensor)

                        accuracy += torch.mean(dim_equals).item()

                    #Display the results of the evaluation
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/evaluate_freq:.3f}.. "
                          f"Evaluation loss: {eval_loss/len(valid_loader):.3f}.. "
                          f"Evaluation accuracy: {accuracy/len(valid_loader):.3f}")

                    running_loss = 0
                    
                    #Switch back to training mode
                    model.train()  
     
    #Do a validation on test dataset
    model_functions.test_validation(model, test_loader, running_device, criterion)
        
    #Save the checkpoint 
    model_functions.save_checkpoint(train_data, arch, model, checkpoint_path, optimizer, input_size, hidden_layers, output_size, learn_rate, epochs, drop_out, batch_size)

if __name__ == "__main__":
    main()