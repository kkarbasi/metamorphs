# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

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


def generate(sprites, dataset_size, num_objects=None, unique=False):
    # Initialise
    all_images = np.zeros((dataset_size, 64, 64, 3))
    all_instance_masks = np.zeros((dataset_size, 64, 64, 1))
    all_obj_counts = np.zeros((dataset_size))
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
        #collect object area before normalization in obj_area
        #zero it before the loop
        obj_area = 0
        for obj_idx in range(num_sprites):
            object_index = randint(0, 737279)
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
        all_obj_counts[i] = num_sprites
        all_obj_areas[i] = float(obj_area)/4096 #set pixels/total image pixels
        all_img_areas[i] = float(len(np.where(instance_masks > 0)[0]))/4096

    all_images = all_images.astype('float32') / 255.0
    return all_images, all_instance_masks, all_obj_counts, all_obj_areas, all_img_areas


def main():
    # Load dataset
    dataset_zip = np.load(
        'data/multi_dsprites/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
        encoding="latin1")
    sprites = dataset_zip['imgs']

    # --- Random colours ---
    # Generate training data
    print("Generate training images...")
    train_images, train_masks, train_obj_counts, train_obj_areas, train_img_areas = generate(sprites, 50000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/training_images_rand4.npy",
            train_images)
    np.save("data/multi_dsprites/processed/training_masks_rand4.npy",
            train_masks)
    np.save("data/multi_dsprites/processed/training_objcounts_rand4.npy", train_obj_counts)
    np.save("data/multi_dsprites/processed/training_objareas_rand4.npy", train_obj_areas)
    np.save("data/multi_dsprites/processed/training_imgareas_rand4.npy", train_img_areas)

    # Generate validation data
    print("Generate validation images...")
    val_images, val_masks, val_obj_counts, val_obj_areas, val_img_areas = generate(sprites, 10000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/validation_images_rand4.npy",
            val_images)
    np.save("data/multi_dsprites/processed/validation_masks_rand4.npy",
            val_masks)
    np.save("data/multi_dsprites/processed/validation_objcounts_rand4.npy", val_obj_counts)
    np.save("data/multi_dsprites/processed/validation_objareas_rand4.npy", val_obj_areas)
    np.save("data/multi_dsprites/processed/validation_imgareas_rand4.npy", val_img_areas)
    # Generate test data
    print("Generate test images...")
    test_images, test_masks, test_obj_counts, test_obj_areas, test_img_areas = generate(sprites, 10000)
    print("Saving...")
    np.save("data/multi_dsprites/processed/test_images_rand4.npy",
            test_images)
    np.save("data/multi_dsprites/processed/test_masks_rand4.npy",
            test_masks)
    np.save("data/multi_dsprites/processed/test_objcounts_rand4.npy", test_obj_counts)
    np.save("data/multi_dsprites/processed/test_objareas_rand4.npy", test_obj_areas)
    np.save("data/multi_dsprites/processed/test_imgareas_rand4.npy", test_img_areas)
    print("Done!")

    # --- Unique random colours ---
    # Generate training data
    print("Generate training images...")
    train_images, train_masks, train_obj_counts, train_obj_areas, train_img_areas = generate(sprites, 50000, unique=True)
    print("Saving...")
    np.save("data/multi_dsprites/processed/training_images_rand4_unique.npy",
            train_images)
    np.save("data/multi_dsprites/processed/training_masks_rand4_unique.npy",
            train_masks)
    np.save("data/multi_dsprites/processed/training_objcounts_rand4_unique.npy", train_obj_counts)
    np.save("data/multi_dsprites/processed/training_objareas_rand4_unique.npy", train_obj_areas)
    np.save("data/multi_dsprites/processed/training_imgareas_rand4_unique.npy", train_img_areas)
    # Generate validation data
    print("Generate validation images...")
    val_images, val_masks, val_obj_counts, val_obj_areas, val_img_areas = generate(sprites, 10000, unique=True)
    print("Saving...")
    np.save("data/multi_dsprites/processed/validation_images_rand4_unique.npy",
            val_images)
    np.save("data/multi_dsprites/processed/validation_masks_rand4_unique.npy",
            val_masks)
    np.save("data/multi_dsprites/processed/validation_objcounts_rand4_unique.npy", val_obj_counts)
    np.save("data/multi_dsprites/processed/validation_objareas_rand4_unique.npy", val_obj_areas)
    np.save("data/multi_dsprites/processed/validation_imgareas_rand4_unique.npy", val_img_areas)
    # Generate test data
    print("Generate test images...")
    test_images, test_masks, test_obj_counts, test_obj_areas, test_img_areas = generate(sprites, 10000, unique=True)
    print("Saving...")
    np.save("data/multi_dsprites/processed/test_images_rand4_unique.npy",
            test_images)
    np.save("data/multi_dsprites/processed/test_masks_rand4_unique.npy",
            test_masks)
    np.save("data/multi_dsprites/processed/test_objcounts_rand4_unique.npy", test_obj_counts)
    np.save("data/multi_dsprites/processed/test_objareas_rand4_unique.npy", test_obj_areas)
    np.save("data/multi_dsprites/processed/test_imgareas_rand4_unique.npy", test_img_areas)
    print("Done!")


if __name__ == "__main__":
    main()
