from utils import *
from transformer import *
import matplotlib.pyplot as plt
import random
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

def train(model,loss,optimizer,tokens,batch_size=4,ddp=False):    
    train_loss = []
    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    if ddp:
        print("Lets use ddp")
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        is_master = ddp_local_rank == 0 
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        is_master = True

    num_epochs = 10
    mega_batch = 2**10
    assert mega_batch % (batch_size*ddp_world_size) == 0
    accum_steps  = mega_batch // (batch_size*ddp_world_size)
    model.to(device)
    model = DDP(model,device_ids=[ddp_local_rank])
    rawmodel = model.module if ddp else model
    loss_sofar = 0
    for epoch in range(num_epochs):
        # torch.save(model,f"model_{model.num_heads}_{model.num_layers}_{model.hidden_dim}_{epoch}.pt")
        # for i in tqdm(range(0,len(input_ids),batch_size)):
        for i in range(ddp_local_rank*batch_size,len(input_ids),batch_size*ddp_world_size):
            optimizer.zero_grad()
            for step in range(accum_steps):
                start_at = i*batch_size*ddp_world_size 
                end_at = (i+1)*batch_size*ddp_world_size 

                input_batch = input_ids[start_at:end_at].to(device)
                attn_batch = attention_mask[start_at:end_at].to(device) 

                
                output = model.forward(input_batch,attn_batch)
            
                tokens_onehot = F.one_hot(input_batch,num_classes = model.module.vocab_size).to(torch.float)
                # output has shape (B,L,V). The shift in targets is for next token prediction obviously
                # So does tokens_onehot
                batch_loss = loss(output[:,:-1,:].transpose(1,2),tokens_onehot[:,1:].transpose(1,2))
                # batch_loss has dimensions (B,L-1)
                batch_loss = batch_loss*attn_batch[:,1:] # double check this 1: vs :-1
                batch_loss = (1/accum_steps)*torch.mean(batch_loss)
                loss_sofar += batch_loss.detach()
                if step == accum_steps - 1 and ddp:
                    model.require_backward_grad_sync = True                    
                batch_loss.backward()
            if ddp:
                dist.all_reduce(loss_sofar,op=dist.ReduceOp.AVG)
            optimizer.step()        
            train_loss.append(loss_sofar.item())  
            print(loss_sofar.item())  
            if not torch.isfinite(batch_loss):
                raise Exception("Loss gone crazy")
            if i % 500 == 0:          
                plt.plot(train_loss)
                plt.savefig("train_loss")
                plt.show()
    # torch.save(mymodel,f"model_{model.num_heads}_{model.num_layers}_{model.hidden_dim}_final.pt")
    return model
                
if __name__ == "__main__":
    parser = argparse. ArgumentParser()
    parser.add_argument("-nh","--num_heads",type = int, default = 16)
    parser.add_argument("-nl","--num_layers",type = int, default = 4)
    parser.add_argument("-d", "--embed_dim",type = int, default = 256)    
    parser.add_argument("-fp", "--file_path",type = str,default="/home/mshyani/compression_embeddings/datasets/tinystories.pkl")    
    args = parser.parse_args()
    mymodel = GPT(args.num_layers, args.embed_dim, args.num_heads,vocab_size=GPT2Tokenizer.from_pretrained('gpt2').vocab_size)
    # mymodel = torch.compile(mymodel)
    optimizer = optim.Adam(mymodel.parameters(),lr=1e-3)    
    
    loss = nn.CrossEntropyLoss(reduction="none")
    with open(args.file_path,"rb") as f:
        data = pickle.load(f)    
    mymodel.train = 1
    mytokenizer = tokenizer()    
    random.shuffle(data)
    tokens = mytokenizer.forward(data)
    mymodel = train(mymodel,loss,optimizer,tokens,ddp=True)
    