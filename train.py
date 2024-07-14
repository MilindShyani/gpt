from utils import *
from transformer import *
import matplotlib.pyplot as plt
import random

def train(model,loss,optimizer,tokens,batch_size=4):    
    train_loss = []
    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    num_epochs = 10
    for epoch in range(num_epochs):
        torch.save(mymodel,f"model_{model.num_heads}_{model.num_layers}_{model.hidden_dim}_{epoch}.pt")
        for i in tqdm(range(0,len(input_ids),batch_size)):
            optimizer.zero_grad()
            input_batch = input_ids[i:i+batch_size]        
            attn_batch = attention_mask[i:i+batch_size]        
            with torch.autocast(device_type= model.device, dtype = torch.float32):
                output = model.forward(input_batch,attn_batch)
            
            tokens_onehot = F.one_hot(input_batch,num_classes = model.vocab_size).to(torch.float)
            # output has shape (B,L,V). The shift in targets is for next token prediction obviously
            # So does tokens_onehot
            batch_loss = loss(output[:,:-1,:].transpose(1,2),tokens_onehot[:,1:].transpose(1,2))
            # batch_loss has dimensions (B,L-1)
            batch_loss = batch_loss*attn_batch[:,1:] # double check this 1: vs :-1
            batch_loss = torch.mean(batch_loss)
            batch_loss.backward()
            optimizer.step()        
            train_loss.append(batch_loss.item())  
            print(batch_loss.item())  
            if not torch.isfinite(batch_loss):
                raise Exception("Loss gone crazy")
            if i % 500 == 0:          
                plt.plot(train_loss)
                plt.savefig("train_loss")
                plt.show()
    torch.save(mymodel,f"model_{model.num_heads}_{model.num_layers}_{model.hidden_dim}_final.pt")
    return model
                
if __name__ == "__main__":
    parser = argparse. ArgumentParser()
    parser.add_argument("-nh","--num_heads",type = int, default = 16)
    parser.add_argument("-nl","--num_layers",type = int, default = 4)
    parser.add_argument("-d", "--embed_dim",type = int, default = 256)    
    parser.add_argument("-fp", "--file_path",type = str,default="/home/mshyani/compression_embeddings/datasets/tinystories.pkl")    
    args = parser.parse_args()
    mymodel = GPT(args.num_layers, args.embed_dim, args.num_heads,vocab_size=GPT2Tokenizer.from_pretrained('gpt2').vocab_size)
    mymodel = torch.compile(mymodel)
    optimizer = optim.Adam(mymodel.parameters(),lr=1e-3)    
    
    loss = nn.CrossEntropyLoss(reduction="none")
    with open(args.file_path,"rb") as f:
        data = pickle.load(f)    
    mymodel.train = 1
    mytokenizer = tokenizer()    
    random.shuffle(data)
    tokens = mytokenizer.forward(data)
    mymodel = train(mymodel,loss,optimizer,tokens)
    