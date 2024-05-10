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

    def attention(self,x,attn_mask):
        # x has shape (B,L,d)
        B,L,d = x.shape
        
        q = self.Wq(x)
        q = einops.rearrange(q,"B L (n h)->B n L h",h=self.head_dim) 

        k = self.Wk(x)
        k = einops.rearrange(k,"B L (n h)->B n L h",h=self.head_dim) 

        v = self.Wv(x)
        v = einops.rearrange(v,"B L (n h)->B n L h",h=self.head_dim) 


        # q = self.Wq(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 
        # k = self.Wk(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 
        # v = self.Wv(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 

        # q k v has dimension (B, num_heads , L , hidden_dim)

        attn = torch.einsum("bnlh, bnLh -> bnlL" ,q,k) # This multiplication has time complexity B*num_heads*hidden_dim*L^2

        # attn has shape (B,num_heads,L,L)

        # Note that if we had not use multihead, we would have (B,L,num_heads*hidden_dim). Consequently
        # attn would have been of shape (B,L,L) and time complexity would be B*L^2*num_heads*hidden_dim. So using MHA does not help with time complexity
        # But MHA does make attn_values low rank because of this split. More details to follow

        # Where the first l is for q and the second for k. That is what we sum over
        
        # Create attention mask that is (B,num_heads,L,L)
        
        attn = attn.masked_fill(self.attention_mask[:,:,:L,:L] == 0, float("-inf"))
                
        # attn_mask has shape (B,L)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        # now it has shape (B,1,1,L)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
            
        attn = attn / math.sqrt(d)
        attn_values = F.softmax(attn,dim=-1)        

        # attn_value has shape (B,num_heads,L,L)
        # v has shape (B,num_heads,L,hidden_dim)
        out = torch.einsum("bnlL,bnLd -> bnld",attn_values,v)
        # out has shape (B,num_heads,L, hidden_dim), where we have summer over keys as before
        out = einops.rearrange(out,"b n L d-> b L (n d)")
        out = self.fc(out)
        return out

    def MLP(self,x):
        return self.mlp(x)

    def forward(self,inputs,attn_mask):
        y = self.ln1(inputs)
        y = self.attention(y,attn_mask)
        y = self.dropout(y)
        out = y + inputs
        out = self.ln2(out)
        out = self.mlp(out)
        return out


class positional(nn.Module):
        def __init__(self,max_len, d) -> None:
            super().__init__()            
            denom = torch.exp( 2 * torch.arange(0,d//2,1) * (-math.log(1e5)/d) ).unsqueeze(0)
            # denom has shape (1,d//2)                        
            pe = torch.zeros(max_len,d).unsqueeze(0)
            pe[0,:,::2] = torch.sin(denom*torch.arange(max_len).unsqueeze(1))
            pe[0,:,1::2] = torch.cos(denom*torch.arange(max_len).unsqueeze(1))
            self.register_buffer("pe",pe)

        def forward(self,x):
            x = x + self.pe[:,x.shape[1],:]
            return x


class Decoder(nn.Module):
    def __init__(self,num_layers, hidden_dim, num_heads,vocab_size, max_len = 512):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.emin = nn.Embedding(vocab_size,hidden_dim)
        self.emout = nn.Linear(hidden_dim,vocab_size)
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
    
    def forward(self,input_ids,attn):        
        x = self.embed(input_ids)
        print("after embed shape",x.shape)
        for layer in self.stack:
            x = layer(x,attn)
        x = self.emout(x)
        return x
    

        
class GPT(nn.Module):
    def __init__(self,num_layers,hidden_dim,num_heads,max_len=512) -> None:
        super().__init__()        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = self.tokenizer.vocab_size
        self.decoder = Decoder(num_layers,hidden_dim,num_heads,vocab_size,max_len)        

    def forward_pass(self,x):
        tokens = self.tokenizer(x,return_tensors="pt", padding=True, truncation= True)
        input_ids = tokens["input_ids"][:,1:-1]                
        attn_mask = tokens["attention_mask"][:,1:-1]                                        
        x = self.decoder(input_ids,attn_mask)                
        return x
                                                                               
    def forward(self, s):
        x = self.forward_pass(s)
        x = F.softmax(x,-1)        
        samples = torch.multinomial(einops.rearrange(x,"B L D -> (B L) D"),num_samples=1)
        samples = einops.rearrange(samples,"(B L) p -> B L p", B = len(s)).squeeze()                                        
        out = self.tokenizer.batch_decode(samples) 
        return out



class train(nn.Module):
    def __init__(self,model,s,targets,batch_size=32):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()                      

    
    def forward(self):
        for i in range(len(s),batch_size):
            input_batch = s[i:i+batch_size]
            target_batch = targets[i:i+batch_size]
            output = self.forward_pass(input_batch)
            # output has shape (B,L,d). The shift in targets is for next token prediction obviously
            batch_loss = self.loss(output[:,:-1,:].transpose(0,2),target_batch[:,1:,:].transpose(0,2))
            batch_loss.backward()




if __name__ == "__main__":
    gpt = GPT(2,64,8)
    print(gpt.parameters)
    output = gpt(["quick brown fox jumped over the dog","I like physics"])
    print(output)