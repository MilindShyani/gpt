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

        self.register_buffer("attention_mask", torch.tril(torch.ones(maxlen,maxlen).view(1,1,maxlen,maxlen)))
        self.kv_cache = {}        

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim) 
        )
        self.dropout = nn.Dropout(drop)

    def attention(self,x,attn_mask,train):
        # x has shape (B,L,d)
        B,L,d = x.shape
        
        if train:            
            print("HERE")
            k = self.Wk(x)
            k = einops.rearrange(k,"B L (n h)->B n L h",h=self.head_dim) 

            v = self.Wv(x)
            v = einops.rearrange(v,"B L (n h)->B n L h",h=self.head_dim) 
        else:
            if x.shape[1] != 1:
                # print("Creating cache")
                k = self.Wk(x)
                k = einops.rearrange(k,"B L (n h)->B n L h",h=self.head_dim) 

                v = self.Wv(x)
                v = einops.rearrange(v,"B L (n h)->B n L h",h=self.head_dim)

                self.kv_cache[self.layer_num] = (k,v)                
            else:
                k, v = self.kv_cache[self.layer_num]
                
                k1 = self.Wk(x)
                k1 = einops.rearrange(k1,"B L (n h)->B n L h",h=self.head_dim) 
                
                k = torch.cat((k,k1),axis=2)

                v1 = self.Wv(x)
                v1 = einops.rearrange(v1,"B L (n h)->B n L h",h=self.head_dim)
                v = torch.cat((v,v1),axis=2)
            
                self.kv_cache[self.layer_num] = (k,v)                
                # print("Using cache and q has shape")

        q = self.Wq(x) 
        # print(q.shape)
        q = einops.rearrange(q,"B l (n h)->B n l h",h=self.head_dim)             
        # q has shape (B,nh,l|1,d)

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
        # attn_mask has now shape (B,1,1,L)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        # attn has shape (B,nh,l|1,L)
        
                    
        attn = attn / math.sqrt(self.head_dim)
        attn_values = F.softmax(attn,dim=-1)        
        attn_values = torch.nan_to_num(attn_values)
        # print(attn_values[1,0])
        # attn_value has shape (B,num_heads,l|1,L)
        # v has shape (B,num_heads,L,head_dim)
        out = torch.einsum("bnlL,bnLh -> bnlh",attn_values,v)
        
        # out has shape (B,num_heads,l|1, head_dim), where we have summer over keys as before
        out = einops.rearrange(out,"b n L h-> b L (n h)")
        out = self.Wz(out)
        # out has shape (B,L,D)
        print(f'out shape is {out.shape}')
        return out

    def MLP(self,x):
        return self.mlp(x)

    def forward(self,inputs,attn_mask,train):
        y = self.ln1(inputs)
        y = self.attention(y,attn_mask,train)
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
        x = x + self.pe[:,:x.shape[1],:]
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
    
    def forward(self,input_ids,attn_mask,train):        
        x = self.embed(input_ids)
        # print("After embed shape",x.shape)
        for layer in self.stack:
            x = layer(x,attn_mask,train) + x            
        x = self.emout(x)
        return x
                        
class GPT(nn.Module):
    def __init__(self,num_layers,hidden_dim,num_heads,max_len=512) -> None:
        super().__init__()        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.vocab_size = self.tokenizer.vocab_size
        self.train = 0        
        self.decoder = Decoder(num_layers,hidden_dim,num_heads,self.vocab_size,max_len)
        self.decoder.register_backward_hook(backward_hook)
        

    def tokenize(self,x):
        self.tokens = self.tokenizer(x,return_tensors="pt",padding=True, truncation=True)
        self.input_ids = self.tokens["input_ids"]
        self.attn_mask = self.tokens["attention_mask"]        
        return 
                                                                                           
    def forward(self, s):                
        self.tokenize(s) 
        print(f'Input shape: {self.input_ids.shape}, {self.attn_mask}')
        x = self.decoder(self.input_ids,self.attn_mask,self.train)                       
        return x,self.tokens        
    
    def freq_penalty(self,logits,penalty: float = 1):
        # logits has shape (B,V)
        # input_ids have shape (B,L)    
        for i in range(len(self.input_ids)):
            counts = torch.bincount(self.input_ids[i,:], minlength = logits.shape[-1])
            logits[i,:] -= penalty*counts
        return logits
            

    def topk(self, x, k : int = 10):
        # x has shape B,V
        idx = torch.argsort(x,dim=-1)[:,:k]
        # idx has shape (B,1,k)
        # these are the ids where prob stays         
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        x = x.masked_fill(~mask, float('-inf'))
        return x
    
    def topp(self, x, p: float = 0.9):
        # x has shape B, V
        sorted_t = torch.sort(x,axis=-1,descending=True)        
        # print(torch.cumsum(sorted_t[0][:,0,:],axis=-1))
        mask = torch.cumsum(sorted_t[0],axis=-1)>p
        # mask has dim (B,V)
        idx = torch.argmax(mask.float(),dim=-1)
        # print(idx)
        # idx has shape (B) and sorted_t[1] has shape (B,1,V)        
        for i in range(x.shape[0]):
            list_idx = sorted_t[1][i:i+1,:idx[i]]
            # list_idx has shape (1,topp)
            # x[i:i+1] has shape (1,V)

            mask = torch.zeros_like(x[i:i+1], dtype=torch.bool)
            # print(mask.shape, list_idx.shape)
            mask.scatter_(-1, list_idx, True)
            x[i:i+1] = x[i:i+1].masked_fill(~mask,float('-inf'))

        return F.softmax(x,-1)
            
        
    def generate(self,s,max_tokens = 4):               
        S = s.copy()  
        out = s               
        for i in range(0,max_tokens):
            out, _ = self.forward(out)
            print(out.shape)
            x = out[:,-1,:]
            
            x = self.freq_penalty(x)        
            
            x = self.topk(x)
            
            x = F.softmax(x,-1)                
            x = self.topp(x)

            samples = torch.multinomial(x, num_samples=1)                        
            out = self.tokenizer.batch_decode(samples)               
            for i in range(len(S)):
                S[i] += out[i]          
                       
        return S        
         
    


        
if __name__ == "__main__":
    gpt = GPT(2,64,8)        
    # for name,module in gpt.named_modules():
    #     print(name,module)
    # for name, param in gpt.named_parameters():
    #     print(name,param.shape)
    output = gpt.generate(["quick brown fox jumped over the dog","I like physics"])
    print(output)
    