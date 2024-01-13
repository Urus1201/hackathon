import torch
import torch.nn as nn
from config import PARAMS
from utils import weights_init

opt = PARAMS()

class ZeroShotModel(nn.Module):
    def __init__(self, opt):
        super(ZeroShotModel, self).__init__()

        self.image_fc = nn.Linear(opt.res_size, opt.hidden_dim)
        self.text_fc = nn.Linear(opt.att_size, opt.hidden_dim)

        self.attention = nn.MultiheadAttention(embed_dim=opt.hidden_dim, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(opt.dropout_rate)

        self.fc2 = nn.Linear(opt.hidden_dim, opt.att_size)

        #Initialize model weights/parameters
        self.apply(weights_init)

    def forward(self, image_features, text_embeddings):
        # Project image features and side information embeddings
        image_features = self.image_fc(image_features)
        text_embeddings = self.text_fc(text_embeddings)

        # Apply cross-attention mechanism
        attention_output_e2v, _ = self.attention(image_features.unsqueeze(1), text_embeddings.unsqueeze(1), text_embeddings.unsqueeze(1))
        attention_output_e2v = torch.mean(attention_output_e2v, dim = 1)

        attention_output_v2e, _ = self.attention(text_embeddings.unsqueeze(1), image_features.unsqueeze(1), image_features.unsqueeze(1))
        attention_output_v2e = torch.mean(attention_output_v2e, dim = 1)

        # Fuse the attention outputs
        fused_features =  torch.stack((attention_output_e2v, attention_output_v2e), dim = 1)
        fused_features_mean, fused_features_std = torch.std_mean(fused_features, dim = 1)
        fused_features = torch.cat((fused_features_mean, fused_features_std), dim = -1) #[batch_size, 2*hidden_dim]

        combined_features = fused_features

        # Fully connected layers
        x = self.fc1(combined_features)
        x = self.fc2(x)

        return x