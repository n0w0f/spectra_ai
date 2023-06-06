import torch

def train(model,loader,optimizer,criterion):
    model.train()
    for epoch in range(30):
        total_loss = 0
        for data in loader:
            edge_index, y, atom, pos = data.edge_index, data.y, data.atom, data.pos
            optimizer.zero_grad()
            out = model(torch.cat((atom, pos), dim=-1),data.batch, edge_index)
            loss = criterion(out.T, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader) 
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")