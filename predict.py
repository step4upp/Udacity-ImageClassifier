import argparse
import json
import torch
import numpy as np
import PIL
import torch.nn.functional as F
from train import gpu_select
from torchvision import models, transforms

def argparsing():
    parser = argparse.ArgumentParser(description="N_Network Settings")
    
    parser.add_argument('--image_dir',
                        type=str,
                        help='indicate the image file for prediction',
                        required=True)
    
    parser.add_argument('--checkpoint',
                        type=str,
                        help='Indicate the chekcpoint file')
    
    parser.add_argument('--category_names',
                        type=str,
                        help='Get josn file to find category names')
    
    parser.add_argument('--top_k',
                        type=int,
                        help='Indicate how many categories analized as similar')
    
    parser.add_argument('--gpu',
                        type=str,
                        help='Please press Y to use GPU')
    
    arguments = parser.parse_args()
    
    return arguments

def checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = eval("models.{}(pretrained=True)".format(checkpoint['architecture']))
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def process_image(image_path):
    image = PIL.Image.open(image_path)
    
    compose = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485,0.456,0.406],
                                                       [0.229,0.224,0.225])])
    
    img = compose(image)
    
    return img

def predict(image_path, model, device, cat_to_name, top_k):
    model.to('cpu')
    img = process_image(image_path)
    img = img.unsqueeze(0)
    img = img.type(torch.FloatTensor)
    
    with torch.no_grad():
        output = model.forward(img)
        
    tops = F.softmax(output.data, dim=1)
    
    top_ps, top_classes = tops.topk(top_k)
    
    ps = np.array(top_ps[0])
    
    top_label = []
    for idx in np.array(top_classes[0]):
        top_label.append(cat_to_name[str(idx+1)])
        
    return ps, top_label

def print_result(ps, top_label):
    for ii, results in enumerate(zip(top_label, ps)):
        print("Rank : {}".format(ii+1),
              "Flower : {}, Similarity: {:.3f}%".format(results[1], results[0]*100))
              
def main():
    argument = argparsing()
     
    with open(argument.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = checkpoint(argument.checkpoint)
    
    device = gpu_select(argument.gpu);
    
    ps, label = predict(argument.image_dir, model, device, cat_to_name, argument.top_k)
    
    print_result(label, ps)
    
if __name__ == '__main__': main()