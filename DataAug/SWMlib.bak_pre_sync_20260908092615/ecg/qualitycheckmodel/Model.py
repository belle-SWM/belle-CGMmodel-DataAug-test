import torch.nn as nn     
        
class CNN(nn.Module):

    def __init__(self, input_channel, data_len=2500):

        self.data_len = data_len
        self.input_channel = input_channel
        
        super(CNN, self).__init__()        

        ## define the layers
        self.conv1 = nn.Sequential(nn.Conv1d(self.input_channel,32,kernel_size=3,padding=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3))
        
        self.conv2 = nn.Sequential(nn.Conv1d(32,64,kernel_size=3,padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5))

        self.conv3 = nn.Sequential(nn.Conv1d(64,128,kernel_size=5,padding=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv1d(128,128,kernel_size=5,padding=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5))
        
        self.conv5 = nn.Sequential(nn.Conv1d(128,64,kernel_size=7,padding=3),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(5),
                                   nn.Dropout(p=0.5))

        self.dense = nn.Sequential(nn.Linear(64*(self.data_len//(5*3)), 2048),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(2048, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(-1, 64*(self.data_len//(5*3))) ## reshaping 
        x = self.dense(x)
        
        return x