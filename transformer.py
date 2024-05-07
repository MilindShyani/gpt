from utils import *

class DecoderLayer(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim        
        self.layer = nn.Linear(hidden_dim, 4*hidden_dim)


    def attention(self):
        pass

    def mlp(self):
        pass

    def forward(self,x):
        pass

class Decoder(nn.Module):
    def __init__():
        super().__init__()

    
    def forward(self,x):
        pass
    


        
class Transformer():
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        pass
