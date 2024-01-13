#class to hold experimental paramters
class PARAMS(object):
  def __init__(self):
      #Model weights are updated by processing self.batch_size number of training samples
      self.batch_size=32
      #Visual feature dimension
      self.res_size=2048
      #Semantic/textual feature dimension
      self.att_size=750
      #Total number of epochs
      self.nepoch=5
      #Learning rate for model training
      self.lr=0.0001
      #True of if GPU available else false
      self.cuda=False
      #Fix the experiment seed for reproducibility
      self.manual_seed=0
      #hidden dim
      self.hidden_dim = 512
      #dropout rate
      self.dropout_rate = 0.5

opt = PARAMS()
print(opt)