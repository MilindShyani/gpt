from transformer import *

if __name__ == "__main__":
    gpt = torch.load("model_8_8_256_9.pt")
    output = gpt.generate(["Mary had a ","It was a bright"],max_tokens = 12)
    print(output)