import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

chars = sorted(set("0123456789+-*/^= "))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

def encode(text):
    return [char2idx[c] for c in text]

def decode(indices):
    return "".join(idx2char[i] for i in indices)

class MathDataset(Dataset):
    def __init__(self, data):
        self.data = [encode(line) for line in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs)
    ys = nn.utils.rnn.pad_sequence(ys)
    return xs, ys

class MathTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x
      
data = [
"2+3=5","10-4=6","3*4=12","9/3=3","2^3=8","5+7=12","8-3=5","6*7=42","12/4=3","4^2=16",
"1+1=2","5-2=3","3*3=9","8/2=4","2^4=16","7+5=12","9-6=3","4*5=20","15/3=5","3^3=27",
"6+2=8","10-5=5","2*6=12","12/6=2","5^2=25","3+4=7","8-1=7","2*5=10","14/7=2","2^5=32",
"9+8=17","20-9=11","3*7=21","18/6=3","4^3=64","7+6=13","11-4=7","5*4=20","21/7=3","2^6=64",
"8+3=11","15-8=7","6*3=18","24/8=3","3^4=81","9+7=16","13-5=8","7*3=21","18/3=6","2^7=128",
"10+5=15","20-10=10","8*2=16","30/5=6","4^4=256","5+9=14","14-7=7","9*2=18","36/6=6","3^5=243",
"7+8=15","16-9=7","10*3=30","40/8=5","5^3=125","6+7=13","18-6=12","11*2=22","42/7=6","2^8=256",
"12+5=17","22-11=11","7*4=28","45/9=5","3^6=729","8+9=17","17-8=9","12*2=24","48/6=8","4^5=1024",
"9+6=15","19-10=9","5*6=30","54/9=6","6^2=36","10+7=17","21-12=9","8*5=40","60/10=6","2^9=512",
"11+8=19","23-14=9","9*3=27","63/7=9","7^2=49","13+6=19","25-15=10","6*7=42","72/8=9","3^7=2187",
"14+5=19","27-16=11","8*6=48","80/10=8","5^4=625","10+12=22","30-18=12","9*5=45","90/9=10","2^10=1024",
"15+6=21","32-20=12","7*7=49","81/9=9","4^6=4096","11+13=24","35-19=16","10*4=40","100/10=10","6^3=216"
]

dataset = MathDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

vocab_size = len(char2idx)
model = MathTransformer(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

epochs = 200
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")

def solve_math_expr(expr):
    try:
        expr = expr.replace("^", "**")
        if re.fullmatch(r"[0-9+\-*/^. ()]+", expr):
            return str(eval(expr))
    except:
        pass
    return None
  
def chat_predict(model, input_text, max_len=20):
    answer = solve_math_expr(input_text.strip("= "))
    if answer is not None:
        return answer

    model.eval()
    seq = encode(input_text)
    input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(1)
    generated = seq[:]
    
    with torch.no_grad():
        for _ in range(max_len - len(seq)):
            output = model(input_seq)
            next_token_logits = output[-1, 0]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            if idx2char[next_token] == " ":
                break
            input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(1)
    return decode(generated)

print("Chat with Victus 1M")
while True:
    user_input = input("Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_predict(model, user_input)
    print("Response:", response)
