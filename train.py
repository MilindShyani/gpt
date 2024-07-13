from utils import *
from transformer import *
import matplotlib.pyplot as plt
import random

def train(model,loss,optimizer,input,batch_size=32):    
    train_loss = []
    print(data[:3])
    for epoch in range(10):
        torch.save(mymodel,f"model_16_2_256_{epoch}.pt")
        for i in tqdm(range(0,len(input),batch_size)):
            optimizer.zero_grad()
            input_batch = input[i:i+batch_size]        
            output, tokens = model(input_batch)
            attn_mask = tokens["attention_mask"][:,1:]
            tokens_onehot = F.one_hot(tokens["input_ids"],num_classes = model.vocab_size).to(torch.float)
            # output has shape (B,L,V). The shift in targets is for next token prediction obviously
            # So does tokens_onehot
            batch_loss = loss(output[:,:-1,:].transpose(1,2),tokens_onehot[:,1:].transpose(1,2))
            # batch_loss has dimensions (B,L)
            batch_loss = batch_loss*attn_mask
            batch_loss = torch.mean(batch_loss)
            batch_loss.backward()
            optimizer.step()        
            train_loss.append(batch_loss.item())  
            if not torch.isfinite(batch_loss):
                raise Exception("Loss gone crazy")
            if i % 500 == 0:          
                plt.plot(train_loss)
                plt.savefig("train_loss")
                plt.show()
    torch.save(mymodel,"model_16_2_256.pt")
    return model
                
if __name__ == "__main__":
    parser = argparse. ArgumentParser()
    parser.add_argument("-nh","--num_heads",type = int)
    parser.add_argument("-nl","--num_layers",type = int)
    parser.add_argument("-d", "--embed_dim",type = int)    
    parser.add_argument("-fp", "--file_path",type = str,default="/home/mshyani/compression_embeddings/datasets/tinystories.pkl")    
    args = parser.parse_args()
    mymodel = GPT(args.num_layers, args.embed_dim, args.num_heads,vocab_size=GPT2Tokenizer.from_pretrained('gpt2').vocab_size)
    optimizer = optim.Adam(mymodel.parameters(),lr=1e-3)    
    
    loss = nn.CrossEntropyLoss(reduction="none")
    with open(args.file_path,"rb") as f:
        data = pickle.load(f)    
    mymodel.train = 1
    random.shuffle(data)
    mymodel = train(mymodel,loss,optimizer,data)
    