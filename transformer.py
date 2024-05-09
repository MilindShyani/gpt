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
        self.register_buffer("attention_mask", torch.tril(torch.ones(maxlen,maxlen).view(1,1,maxlen,maxlen)))


        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim) 
        )
        self.dropout = nn.Dropout(drop)

    def attention(self,x):
        # x has shape (B,L,d)
        B,L,d = x.shape

        q = self.Wq(x).view(B,L,self.num_heads, self.hidden_dim).transpose(1,2) 
        k = self.Wk(x).view(B,L,self.num_heads, self.hidden_dim).transpose(1,2) 
        v = self.Wv(x).view(B,L,self.num_heads, self.hidden_dim).transpose(1,2) 

        # q k v has dimension (B, num_heads , L , hidden_dim)

        attn = torch.einsum("blnh, bLnh -> bnlL" ,q,k) # This multiplication has time complexity B*num_heads*hidden_dim*L^2

        # attn has shape (B,num_heads,L,L)

        # Note that if we had not use multihead, we would have (B,L,num_heads*hidden_dim). Consequently
        # attn would have been of shape (B,L,L) and time complexity would be B*L^2*num_heads*hidden_dim. So using MHA does not help with time complexity
        # But MHA does make attn_values low rank because of this split. More details to follow

        # Where the first l is for q and the second for k. That is what we sum over
        
        # Create attention mask that is (B,num_heads,L,L)
        attn.masked_fill(self.attention_mask == 0, float("-inf"))

        attn = attn / math.sqrt(d)

        attn_values = F.softmax(attn,dim=-1)        

        # attn_value has shape (B,num_heads,L,L)
        # v has shape (B,num_heads,L,hidden_dim)
        out = torch.einsum("bnlL,bnLd -> bnld",attn_values,v)
        # out has shape (B,num_heads,L, hidden_dim), where we have summer over keys as before
        out = self.fc(out)
        return

    def MLP(self,x):
        return self.mlp(x)

    def forward(self,x):
        y = self.ln1(x)
        y = self.attention(y)
        y = self.dropout(y)
        x = y + x
        x = self.ln2(x)
        x = self.mlp(x)
        return x


class positional(nn.Module):
        def __init__(self,max_len, d) -> None:
            super().__init__()            
            denom = torch.exp( 2* torch.arange(0,d//2,1) * (-math.log(1e5)/d) )
            pe = torch.zeros(max_len,d).unsqueeze(0)
            pe[:,:,::2] = torch.sin(denom*torch.arange(self.max_len))
            pe[:,:,1::2] = torch.sin(denom*torch.arange(self.max_len))
            self.register_buffer("pe",pe)

        def forward(self,x):
            x = x + self.pe[:,x.shape[1],:]
            return x


class Decoder(nn.Module):
    def __init__(self,num_layers, hidden_dim, num_heads,max_len = 512):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.emin = nn.Embedding(vocab_size,hidden_dim)
        self.emout = nn.Embedding(hidden_dim,vocab_size)
        self.stack = nn.ModuleList([DecoderLayer(hidden_dim,num_heads) for _ in range(num_layers)])
        self.max_len = max_len
        self.pos = positional(max_len,hidden_dim)


    def embed(self,x):
        # x has shape (B,L,V)
        x = self.emin(x)
        # x has shape (B,L,d)

        # And now we add positional encoding
        x = self.pos(x)    
        return x
    
    def forward(self,x):        
        x = self.embed(x)
        for layer in self.stack:
            x = layer(x)
        x = self.emout(x)
        return x
    

        
class Transformer():
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        pass
