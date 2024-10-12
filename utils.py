import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def show_generated_images(images):
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
