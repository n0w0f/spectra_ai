import torch
import wandb


import os
from dotenv import load_dotenv

# Set the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), os.pardir, '.env')

print(dotenv_path)
load_dotenv(dotenv_path)

# Access the keys and passwords using the environment variable names
wb_api_key = os.getenv("WB_API_KEY")
project_name = os.getenv("PROJECT_NAME")
checkpoint_path = os.getenv("CHECKPOINT_PATH")


wandb.login()

def train(model, train_loader, val_loader, optimizer, criterion, epochs, checkpoint_path=None, checkpoint_epochs=5):
    wandb.init(project=project_name, name="training-run")
    wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            edge_index, y, atom, pos = data.edge_index, data.y, data.atom, data.pos
            optimizer.zero_grad()
            out = model(torch.cat((atom, pos), dim=-1), data.batch, edge_index)
            loss = criterion(out.view(-1).squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_loss:.4f}")
        wandb.log({"Epoch": epoch + 1, "Average Training Loss": avg_loss})

        # Evaluate on the validation set
        val_loss = evaluate(model, val_loader, criterion)
        wandb.log({"Epoch": epoch + 1, "Average Validation Loss": val_loss})

        if checkpoint_path and (epoch + 1) % checkpoint_epochs == 0:
            # Save model checkpoint
            checkpoint_file = os.path.join(checkpoint_path, f"{epoch}checkpoint.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, checkpoint_file)
            print("Checkpoint saved")

    wandb.finish()
    print("Training completed")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            edge_index, y, atom, pos = data.edge_index, data.y, data.atom, data.pos
            out = model(torch.cat((atom, pos), dim=-1), data.batch, edge_index)
            loss = criterion(out.view(-1).squeeze(), y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Average Loss on Evaluation Set: {avg_loss:.4f}")
#    wandb.log({"Average Loss on Evaluation Set": avg_loss})

    # Log model weights and biases
    for name, param in model.named_parameters():
        wandb.log({"Weights/" + name: param.detach()})
        wandb.log({"Biases/" + name: param.detach()})

    return avg_loss
