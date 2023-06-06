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




def train(model, loader, optimizer, criterion, checkpoint_path=checkpoint_path, checkpoint_epochs=5):
    wandb.init(project=project_name, name="training-run")
    wandb.watch(model)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for data in loader:
            edge_index, y, atom, pos = data.edge_index, data.y, data.atom, data.pos
            optimizer.zero_grad()
            out = model(torch.cat((atom, pos), dim=-1), data.batch, edge_index)
            loss = criterion(out.T, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        wandb.log({"Epoch": epoch + 1, "Average Loss": avg_loss})

        checkpoint_file = os.path.join(checkpoint_path, f"{epoch}checkpoint.pth")

        if checkpoint_path and (epoch + 1) % checkpoint_epochs == 0:
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, checkpoint_file)
            print("Checkpoint saved")

    wandb.finish()
    print("Training completed")

