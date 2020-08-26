from Dataset import get_dataset
from model import LSTMClassifier, train_model
from utils import make_submit
import torch
from tqdm import tqdm
import numpy as np


TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter = get_dataset()


lr = 1e-4
batch_size = 32
output_size = 2
hidden_size = 228
embedding_length = 300
num_epochs = 20

model = LSTMClassifier(vocab_size=vocab_size, 
                       output_size=output_size, 
                       embedding_dim=embedding_length,
                       hidden_dim=hidden_size,
                       n_layers=2,
                       weights=word_embeddings
)

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
    
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss()
    
train_loss, train_acc, val_loss, val_acc = train_model(model=model,
                                                       train_iter=train_iter,
                                                       val_iter=val_iter,
                                                       optim=optim,
                                                       loss=loss,
                                                       num_epochs=num_epochs,
                                                       batch_size=batch_size)

model.load_state_dict(torch.load('state_dict.pth'))
model.eval()
results_target = list()

with torch.no_grad():
    for batch in tqdm(test_iter):
        for text in zip(batch.text[0]):#, batch.id):
            text = text[0].unsqueeze(0)
            res, _ = model(text, hidden=None)

            target = np.round(res.cpu().numpy())
            
            results_target.append(target[0][1])


print("Making submit")
make_submit(results_target)
