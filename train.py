from utils import *
from transformer import *
import matplotlib.pyplot as plt

def train(model,loss,optimizer,input,batch_size=32):    
    train_loss = []
    print(data[:3])
    for i in tqdm(range(0,len(input),batch_size)):
        optimizer.zero_grad()
        input_batch = input[i:i+batch_size]        
        output, tokens = model(input_batch)
        tokens_onehot = F.one_hot(tokens["input_ids"],num_classes = model.vocab_size).to(torch.float)
        # output has shape (B,L,d). The shift in targets is for next token prediction obviously
        batch_loss = loss(output[:,:-1,:].transpose(0,2),tokens_onehot[:,1:].transpose(0,2))
        batch_loss.backward()
        optimizer.step()        
        train_loss.append(batch_loss.item())  
        if i % 1 == 0:          
            plt.plot(train_loss)
            plt.savefig("train_loss")
            plt.show()
    return model
                

if __name__ == "__main__":
    parser = argparse. ArgumentParser()
    parser.add_argument("-nh","--num_heads",type = int)
    parser.add_argument("-nl","--num_layers",type = int)
    parser.add_argument("-d", "--embed_dim",type = int)    
    parser.add_argument("-fp", "--file_path",type = str)    
    args = parser.parse_args()
    mymodel = GPT(args.num_layers, args.embed_dim, args.num_heads)
    optimizer = optim.Adam(mymodel.parameters(),lr=1e-4)    
    loss = nn.CrossEntropyLoss()
    with open(args.file_path,"rb") as f:
        data = pickle.load(f)    
    mymodel.train = 1
    mymodel = train(mymodel,loss,optimizer,data)





