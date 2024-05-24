from utils import *

def backward_hook(module, grad_input, grad_output):
    # print(f"Input grads are {grad_input}")
    # print(f"Output grads are {grad_output}")    
    return


class DecoderLayer(nn.Module):
    def __init__(self,hidden_dim, num_heads, layer_num, maxlen = 512, drop = 0.2):        
        super().__init__()
        self.hidden_dim = hidden_dim     
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.layer_num = layer_num
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
        self.kv_cache = {}
        self.init_pass = 0

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim) 
        )
        self.dropout = nn.Dropout(drop)

    def attention(self,x,attn_mask):
        # x has shape (B,L,d)
        B,L,d = x.shape
        
        if self.train:
            q = self.Wq(x) 
            q = einops.rearrange(q,"B L (n h)->B n L h",h=self.head_dim) 
            # q has shape (B,nh,L,d)

            k = self.Wk(x)
            k = einops.rearrange(k,"B L (n h)->B n L h",h=self.head_dim) 

            v = self.Wv(x)
            v = einops.rearrange(v,"B L (n h)->B n L h",h=self.head_dim) 
        else:
            if self.init_pass:
                k = self.Wk(x)
                k = einops.rearrange(k,"B L (n h)->B n L h",h=self.head_dim) 

                v = self.Wv(x)
                v = einops.rearrange(v,"B L (n h)->B n L h",h=self.head_dim)

                self.kv_cache[self.layer_num] = (k,v)                
            else:
                k, v = self.kv_cache[self.layer_num]
            
            q = self.Wq(x) 
            q = einops.rearrange(q,"B L (n h)->B n L h",h=self.head_dim) 
            # q has shape (B,nh,1,d)

        # q = self.Wq(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 
        # k = self.Wk(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 
        # v = self.Wv(x).view(B,L,self.num_heads, self.head_dim).transpose(1,2) 
        # q k v has dimension (B, num_heads , L , hidden_dim)
        attn = torch.einsum("bnlh, bnLh -> bnlL" ,q,k) # This multiplication has time complexity B*num_heads*hidden_dim*L*l
        # attn has shape (B,num_heads,l|1,L)

        # Note that if we had not use multihead, we would have (B,L,num_heads*hidden_dim). Consequently
        # attn would have been of shape (B,L,L) and time complexity would be B*L^2*num_heads*hidden_dim. So using MHA does not help with time complexity
        # But MHA does make attn_values low rank because of this split. More details to follow

        # Where the first l is for q and the second for k. That is what we sum over        
        # Create attention mask that is (B,num_heads,L,L)        
        if self.train:
            # we need the causal mask only when training
            attn = attn.masked_fill(self.attention_mask[:,:,:L,:L] == 0, float("-inf"))                    
            # attn_mask has shape (B,L)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            # now it has shape (B,1,1,L)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
                    
        attn = attn / math.sqrt(d)
        attn_values = F.softmax(attn,dim=-1)        

        # attn_value has shape (B,num_heads,l|1,L)
        # v has shape (B,num_heads,L,hidden_dim)
        out = torch.einsum("bnlL,bnLd -> bnld",attn_values,v)
        # out has shape (B,num_heads,l|1, hidden_dim), where we have summer over keys as before
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
        self.stack = nn.ModuleList([DecoderLayer(hidden_dim,num_heads,layer_num) for layer_num in range(num_layers)])
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
        # print("After embed shape",x.shape)
        for layer in self.stack:
            x = layer(x,attn)
        x = self.emout(x)
        return x
                        
class GPT(nn.Module):
    def __init__(self,num_layers,hidden_dim,num_heads,max_len=512) -> None:
        super().__init__()        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.vocab_size = self.tokenizer.vocab_size
        self.decoder = Decoder(num_layers,hidden_dim,num_heads,self.vocab_size,max_len)
        self.decoder.register_backward_hook(backward_hook)
        self.train = 0        

    def tokenize(self,x):
        self.tokens = self.tokenizer(x,return_tensors="pt",padding=True, truncation=True)
        input_ids = self.tokens["input_ids"]
        attn_mask = self.tokens["attention_mask"]        
        return input_ids, attn_mask
                                                                                           
    def forward(self, s):        
        input_ids, attn_mask = self.tokenize(s)                                                             
        print(f'Input shape: {input_ids.shape}')
        x = self.decoder(input_ids,attn_mask)                       
        return x,self.tokens        

    def generate(self,s,max_tokens = 5):
        self.train = 0
        self.init_pass = 1        
        out = self.forward(s)[0][:,-2:-1,:]                
        x = F.softmax(out,-1)
        samples = torch.multinomial(einops.rearrange(x,"B L D -> (B L) D"),num_samples=1)
        samples = einops.rearrange(samples,"(B L) p -> B L p", B = len(s)).squeeze()   
        out = self.tokenizer.batch_decode(samples)         
        for i in range(len(s)):
            s[i] += out[i]
        
        self.init_pass = 0
        for i in range(1,max_tokens):
            out = self.forward(s)[0][:,-2:-1,:] 
            x = F.softmax(out,-1)
            samples = torch.multinomial(einops.rearrange(x,"B L D -> (B L) D"),num_samples=1)
            samples = einops.rearrange(samples,"(B L) p -> B L p", B = len(s)).squeeze()   
            out = self.tokenizer.batch_decode(samples)               
            for i in range(len(s)):
                s[i] += out[i]
        return s        
    


        
if __name__ == "__main__":
    # gpt = GPT(2,64,8)        
    # for name,module in gpt.named_modules():
    #     print(name,module)
    # for name, param in gpt.named_parameters():
    #     print(name,param.shape)
    # output = gpt.generate(["quick brown fox jumped over the dog","I like physics"])
    # print(output)
    pass