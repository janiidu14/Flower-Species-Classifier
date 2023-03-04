#Import libraries and modules
import json
import argparse

import util_functions
import model_functions

#Setting up parser
parser = argparse.ArgumentParser(
    description='This is a parser for predicting the results of the model',
)

parser.add_argument('input', action="store", default='./flowers/test/12/image_03996.jpg')
parser.add_argument('checkpoint', action="store", default="vgg16_checkpoint.pth")
parser.add_argument('--top_k', action="store", default=5, type=int)
parser.add_argument('--category_names', action="store", default='cat_to_name.json')
parser.add_argument('--gpu', action="store", default="cpu")

args = parser.parse_args()

img_path = args.input
checkpoint_path = args.checkpoint
topk = args.top_k
device = args.gpu
json_mapping_pth = args.category_names

def main():
    model = model_functions.load_checkpoint(checkpoint_path)
    
    with open(json_mapping_pth, 'r') as f:
        cat_to_name = json.load(f)
        
    probs, labels = model_functions.predict(img_path, model)
    
    categories = []
    for label in labels:
        categories.append(cat_to_name[label])
        
    #Display probabilities and classes of the specific image
    print("Top {} Probabilities: {}".format(topk, probs))
    print("Top {} classes: {}".format(topk, labels))
    print("Top {} class labels: {}".format(topk, categories))
    
    print("Top Category of the predicted Flower: ", cat_to_name[labels[0]])
    print("Top Probability of the predicted Flower: ", probs[0])
    
if __name__== "__main__":
    main()