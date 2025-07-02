import torch
from torch.utils.data import DataLoader

# Assuming your dataset class is named `YourDataset`
# Replace it with your actual dataset class

def test_dataloader(dataloader, model):
    model.eval()  # Set model to evaluation mode to avoid unnecessary operations (e.g., dropout)

    for i, (images, masks) in enumerate(dataloader):
        # Check the shapes of images and masks
        print(f"Batch {i+1}:")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")

        # Check if the images are on the right device (e.g., cuda or cpu)
        if images.device != next(model.parameters()).device:
            print(f"Warning: Images are on device {images.device}, model is on device {next(model.parameters()).device}")
        
        # Check the data types of images and masks
        print(f"Image dtype: {images.dtype}")
        print(f"Mask dtype: {masks.dtype}")

        # Check if the image is compatible with the model's input
        try:
            output = model(images)  # Pass a batch through the model
            print(f"Model output shape: {output.shape}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
        
        # Limit the check to one or two batches to avoid excessive computation
        if i == 1:  # Only check the first 2 batches
            break

# Assuming your model is already defined and your DataLoader is set up
# Replace `your_model` and `your_dataloader` with the actual objects

test_dataloader(your_dataloader, your_model)
