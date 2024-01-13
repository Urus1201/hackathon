from model import ZeroShotModel
from dataset import DATA_LOADER_HK
from utils import sample, mse_custom_loss, evaluate_model_performance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



def train_model(opt):
	data = DATA_LOADER_HK(opt, config_path='../path_config.json')
	
    #Create an instance of the model
	net_model = ZeroShotModel(opt)
	
    #Define the torch tensors for input and output
	input_res = torch.FloatTensor(opt.batch_size, opt.res_size)
	input_att = torch.FloatTensor(opt.batch_size, opt.att_size)

    #Set up optimizer
	optimizer = optim.AdamW(net_model.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.0001)
	
	#Transfer your data and model to GPU if available
	if opt.cuda:
		net_model.cuda()
		input_res = input_res.cuda()
		input_att = input_att.cuda()

	ctr = 0
	#Train the model by passing training data opt.nepoch times
	for epoch in range(opt.nepoch):
		for i in range(0, data.ntrain, opt.batch_size):
			#sample the batch of training samples
			sample(data, opt, input_res, input_att)
			#Make input/output tensors Variable to allow backpropogation of gradients
			input_resv = Variable(input_res)
			input_attv = Variable(input_att)
			input_attv = nn.functional.normalize(input_attv, dim=1)
			pred_sem = net_model(input_resv, input_attv)
			#calculate the loss
			latent_loss = mse_custom_loss(pred_sem, input_attv)
			total_loss = latent_loss
			#Reset the gradients
			net_model.zero_grad()
			#Backpropogate the loss
			total_loss.backward()
			#Update the model gradients
			optimizer.step()
			#Set up the model in evaluation mode i.e. no more weight updates by gradients
			net_model.eval()
			if (i % (opt.batch_size*10) ==0):
				ctr+=1
				val_zsl_acc = evaluate_model_performance(net_model, data, opt)
				print('[%5d] Loss: %.4f |Acc: ZSL %.4f, %d ctr'%(epoch, total_loss, val_zsl_acc, ctr))
			#Set up the model back to training mode i.e. allows weight updates by gradients
			net_model.train()


if __name__ == "__main__":
	from config import PARAMS
	opt = PARAMS()
	train_model (opt)