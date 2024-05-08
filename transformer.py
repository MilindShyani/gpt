from utils import *

class DecoderLayer(nn.Module):
    def __init__(self,hidden_dim, num_heads, maxlen = 512, drop = 0.2):        
        super().__init__()
        self.hidden_dim = hidden_dim     
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.head_dim = hidden_dim // num_heads   
        assert (hidden_dim % num_heads == 0)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias = False) 
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias = False) 
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias = False) 
        self.Wz = nn.Linear(hidden_dim, hidden_dim, bias = False) 
        self.fc = nn.Linear(hidden_dim, hidden_dim) 
        self.register_buffer("attention_mask", torch.tril(torch.ones(maxlen,maxlen).view(-1,maxlen.maxlen)))


        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim) 
        )
        self.dropout = nn.Dropout(drop)

    def attention(self,x):
        # x has shape (B,L,d)
        # Create attention mask that is (B,L,L)
        prods.masked_fill(self.attention_mask == 0, float("-inf"))
        



        pass

    def MLP(self,x):
        return self.mlp(x)

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
