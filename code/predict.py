import pandas as pd

import torch
import torch.nn as nn

from torch.autograd import Variable
from dataset import DATA_LOADER_HK

from model import ZeroShotModel
from utils import map_label, mse_custom_loss

def predict_new_instances(model, data, opt):
    model.eval()

    #test data
    test_res = torch.from_numpy(data.test_feature).float()
    test_att = torch.from_numpy(data.attributes).float()

    # Normalize the attributes
    test_att_normalized = nn.functional.normalize(test_att, dim=1)

    if opt.cuda:
        test_res = test_res.cuda()
        test_att = test_att_normalized.cuda()

    # Forward pass to obtain predictions
    with torch.no_grad():
        pred_sem = model(test_res, test_att_normalized)
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
    data = DATA_LOADER_HK(opt, mode='test', config_path='../path_config.json')

    # Instantiate the model
    net_model = ZeroShotModel(opt) 

    # Load the trained model checkpoint
    if opt.cuda:
        checkpoint = torch.load('../model/model_acc_0.77.pt')
    else:
        checkpoint = torch.load('../model/model_acc_0.77.pt', map_location=torch.device('cpu'))

    net_model.load_state_dict(checkpoint)

    # cast the model to right device
    # net_model.to(opt.device)

    # Predict new instances
    predicted_labels = predict_new_instances(net_model, data, opt)
