from transformer import *

if __name__ == "__main__":
    gpt = torch.load("model_8_8_256_6.pt")
    output = gpt.generate(["Mary had a ","It was a "])
    # output = gpt.generate(["Mary had a "])
    print(output)