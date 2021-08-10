import random
from random import randint, choice

import torch

import numpy as np
from PIL import Image


# Set manual seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def rand_rgb_tuple():
    val = [0, 63, 127, 191, 255]
    return choice(val), choice(val), choice(val)

def generate(sprites, latent_values, dataset_size, num_objects=None, unique=False):
    """Generator function with latent_values return"""
    # Initialise
    all_images = np.zeros((dataset_size, 64, 64, 3))
    all_instance_masks = np.zeros((dataset_size, 64, 64, 1))
    all_latent_vals = np.zeros((dataset_size, num_objects, 6))
    all_obj_counts = np.zeros((dataset_size), dtype=np.int8)
    all_obj_areas = np.zeros((dataset_size))
    all_img_areas = np.zeros((dataset_size))
    # Create images
    for i in range(dataset_size):
        if (i+1)%10000 == 0:
            print(f"Processing [{i+1} | {dataset_size}]")

        # Create background
        background_colour = rand_rgb_tuple()
        image = np.array(Image.new('RGB', (64, 64), background_colour))
        # Initialise instance masks
        instance_masks = np.zeros((64, 64, 1)).astype('int')

        img_colours = [background_colour]

        # Add objects
        if num_objects is None:
            num_sprites = randint(1, 4)
        else:
            num_sprites = num_objects
        latent_objs = []
        obj_area = 0
        for obj_idx in range(num_sprites):
            object_index = randint(0, 737279)
            latent_vals = latent_values[object_index]
            latent_objs.append(latent_vals)
            sprite_mask = np.array(sprites[object_index], dtype=bool)
            crop_index = np.where(sprite_mask == True)
            #use num pixels as a proxy for area
            obj_area += len(crop_index[0])
            object_colour = rand_rgb_tuple()
            # Optional: get new random colour if colour has already been used
            while unique and object_colour in img_colours:
                object_colour = rand_rgb_tuple()
            image[crop_index] = object_colour
            instance_masks[crop_index] = obj_idx + 1
            img_colours.append(object_colour)
        # Collate
        all_images[i] = image
        all_instance_masks[i] = instance_masks
        all_latent_vals[i] = np.array(latent_objs)
        all_obj_counts[i] = num_sprites
        all_obj_areas[i] = float(obj_area)/4096 #set pixels/total image pixels
        all_img_areas[i] = float(len(np.where(instance_masks > 0)[0]))/4096
    all_images = all_images.astype('float32') / 255.0
    return all_images, all_instance_masks, all_obj_counts, all_obj_areas, all_img_areas , all_latent_vals

