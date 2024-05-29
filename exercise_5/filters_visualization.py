
import numpy as np
import os
import torch
from torch.optim import Adam, SGD
from torchvision import models
from  aux_ops import preprocess_image, recreate_image, save_image


# Initialize GPU if available
use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
# Select device to work on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Total Variation Loss
def total_variation_loss(img, weight):
    #img has n,c,h,w format
    diff_i = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])**2) # differences horizontal
    diff_j = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])**2) # differences vertical
    loss = torch.sqrt(diff_i + diff_j) # according to formula

    return weight * loss

def visualise_layer_filter(model, layer_nmbr, filter_nmbr, num_optim_steps=26, print_statistics=False): # added parameter to print mean, std, etc for ex. 1.2

    # Generate a random image
    rand_img = np.uint8(np.random.uniform(low=120,
                                          high=190,
                                          size=(224, 224, 3)))

    # Process image and return variable
    processed_image = preprocess_image(rand_img, False)
    processed_image = torch.tensor(processed_image, device=device).float()
    processed_image.requires_grad = True
    # Define optimizer for the image
    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-5)
    for i in range(1, num_optim_steps):
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = processed_image
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            # Print shape, mean, min, and max of the activation
            if print_statistics is True:
                print(f"Layer {index}:")
                print(f"  Shape: {x.shape}")
                print(f"  Mean: {x.mean().item():.4f}")
                print(f"  Min: {x.min().item():.4f}")
                print(f"  Max: {x.max().item():.4f}")
            if index == layer_nmbr:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break

        conv_output = x[0, filter_nmbr]
        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        loss = -torch.mean(conv_output)
        # You may need to add total variation loss later
        # loss_tv = total_variation_loss(processed_image, 500.)
        # loss = -torch.mean(conv_output) + loss_tv*1.

        # print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image


#function customised for exercise 1.5 with the different hyperparameters
def visualise_layer_filter_1_5(model, layer_nmbr, filter_nmbr, num_optim_steps=26, lr=0.1, optimizer=None,new_img=False): # added parameter to print mean, std, etc for ex. 1.2

    # Generate a random image
    rand_img = np.uint8(np.random.uniform(low=120,
                                          high=190,
                                          size=(224, 224, 3)))
    

    if new_img is True:
            rand_img = np.uint8(np.random.uniform(low=128,
                                          high=128,
                                          size=(224, 224, 3)))

    # Process image and return variable
    processed_image = preprocess_image(rand_img, False)
    processed_image = torch.tensor(processed_image, device=device).float()
    processed_image.requires_grad = True
    # Define optimizer for the image
    if optimizer =='Adam':
        optimizer = Adam([processed_image], lr=lr, weight_decay=1e-5)
    if optimizer == 'SGD':
        optimizer = SGD([processed_image], lr=lr, weight_decay=1e-5)    
    for i in range(1, num_optim_steps):
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = processed_image
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            # Print shape, mean, min, and max of the activation
            if index == layer_nmbr:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break

        conv_output = x[0, filter_nmbr]
        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        loss = -torch.mean(conv_output)
        # You may need to add total variation loss later
        # loss_tv = total_variation_loss(processed_image, 500.)
        # loss = -torch.mean(conv_output) + loss_tv*1.

        # print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image

#helper function for 2.1
def visualise_layer_filter_2_start(model, layer_nmbr, filter_nmbr, init_img, num_optim_steps=26):
    processed_image = init_img.clone().detach().requires_grad_(True)
    
    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-5)

    for i in range(1, num_optim_steps):
        optimizer.zero_grad()

        # Forward pass layer by layer
        x = processed_image
        for index, layer in enumerate(model):
            x = layer(x)
            if index == layer_nmbr:
                break

        conv_output = x[0, filter_nmbr]

        # Loss function is the mean of the output of the selected layer/filter
        loss = -torch.mean(conv_output)

        # Compute gradients
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image

#Same as above method but with the total loss
def visualise_layer_filter_2(model, layer_nmbr, filter_nmbr, init_img, num_optim_steps=26, tv_weight=1e-6):
    processed_image = init_img.clone().detach().requires_grad_(True)
    
    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-5)

    for i in range(1, num_optim_steps):
        optimizer.zero_grad()

        # Forward pass layer by layer
        x = processed_image
        for index, layer in enumerate(model):
            x = layer(x)
            if index == layer_nmbr:
                break

        conv_output = x[0, filter_nmbr]

        # Loss function is the mean of the output of the selected layer/filter
        loss = -torch.mean(conv_output)
        # Add total variation loss to the loss function
        loss_tv = total_variation_loss(processed_image, tv_weight)
        loss += loss_tv

        # Compute gradients
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image

if __name__ == '__main__':
    layer_nmbr = 28
    filter_nmbr = 228

    # Fully connected layer is not needed
    model = models.vgg16(pretrained=True).features
    model.eval()
    # Fix model weights
    for param in model.parameters():
        param.requires_grad = False
    # Enable GPU
    if use_gpu:
        model.cuda()

    # use this output in some way
    visualise_layer_filter(model,
                           layer_nmbr=layer_nmbr,
                           filter_nmbr=filter_nmbr)
