import pandas as pd

import torch
import torch.nn as nn

from torch.autograd import Variable
from dataset import DATA_LOADER_HK

from model import ZeroShotModel
from utils import map_label, mse_custom_loss

def predict_new_instances(model, data, opt):
    model.eval()

    # Assuming you have new instances (features, attributes, etc.)
    # Replace the following lines with your actual new instance data loading/preprocessing
    new_instance_feature = torch.from_numpy(...)  # Replace with your new instance feature
    new_instance_attribute = torch.from_numpy(...)  # Replace with your new instance attribute

    # Normalize the attributes
    new_instance_attribute = nn.functional.normalize(new_instance_attribute, dim=1)

    # Forward pass to obtain predictions
    with torch.no_grad():
        pred_sem = model(new_instance_feature, new_instance_attribute)
        pred_sem = pred_sem.cpu().detach().numpy()

    #save the predictions
    save_predictions(predicted_semantics=pred_sem)

    print('Predictions generated in the results folder!!')

def save_predictions(predicted_semantics):
    df = pd.DataFrame(predicted_semantics)
    df.columns = [str(i) for i in range(750)]
    df.to_csv('../results/predicted_semantics.csv', index= True, index_label = 'ID')


if __name__ == "__main__":
    from config import PARAMS
    opt = PARAMS()

    # Instantiate the DATA_LOADER_HK class
    data = DATA_LOADER_HK(opt)

    # Instantiate the model
    net_model = ZeroShotModel() 

    # Load the trained model checkpoint
    checkpoint = torch.load('../model/model_acc_0.75.pt') 
    net_model.load_state_dict(checkpoint)

    # cast the model to right device
    net_model.to(opt.device)

    # Predict new instances
    predicted_labels = predict_new_instances(net_model, data, opt)

    # Now 'predicted_labels' contains the predicted labels for your new instances
    print(predicted_labels)
