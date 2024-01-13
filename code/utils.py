import torch
import numpy as np
from config import PARAMS
from sklearn.metrics import balanced_accuracy_score

opt = PARAMS()

def map_label_hk(label, classes):
    mapped_label = np.empty((label.shape[0],), np.int8)
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i
    return mapped_label

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i], ] = i
    return mapped_label

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def sample(data, opt, input_res, input_att):
    batch_feature, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def mse_custom_loss(latent_from_vis, latent_from_att):
    L2 = torch.sum((latent_from_vis - latent_from_att) ** 2)
    return L2

def evaluate_model_performance(net, data):
    mapped_batch_label = map_label(data.val_label, data.val_classes)
    allattributes = data.allattribute.detach().cpu().numpy()
    val_att = allattributes[mapped_batch_label]
    val_att_normalized = val_att / np.linalg.norm(val_att, axis=1, keepdims=True)
    pred_att_val = net(data.val_feature.cuda(), torch.tensor(val_att_normalized).cuda())
    pred_att_val = pred_att_val.detach().cpu().numpy()
    val_label = data.val_label.detach().cpu().numpy()
    val_attributes = allattributes[len(data.train_classes):]

    # validation
    pred_val_labels = [np.argmax([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) for y in val_attributes])
                       for x in pred_att_val]
    pred_val_labels = np.array(data.val_classes)[pred_val_labels]
    val_zsl_acc = balanced_accuracy_score(val_label, pred_val_labels)
    return val_zsl_acc