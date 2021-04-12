from tqdm import tqdm 
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imagehash
import PIL
import os
import argparse

def argument():
    parser = argparse.ArgumentParser(description ='Running Revealing Duplicates Images !!')
    parser.add_argument("-p", "--path_images", type = str,
                            metavar = "folder_path", default = "",
                            help = "The folder root images")
    parser.add_argument("-dst", "--dest_images", type = str,
                            metavar = "folder_dest", default = "",
                            help = "The folder to save image scaled !!")
    parser.add_argument("-s", "--size", type = int,
                            metavar = "Size Image Save", default = 512,
                            help = "The Size Of Scaled Images")
    parser.add_argument("-thres", "--threshold", type = float,
                            metavar = "Threshold for Choosing", default = 0.9,
                            help = "Threshold for choose")
    parser.add_argument("-se", "--seed", type = int,
                            metavar = "Seeding", default = 42,
                            help = "Seeding")
    parser.add_argument("-csv_dir", "--save_csv_dir", type = str,
                            metavar = "Save file csv in dir ", default = "",
                            help = "Dir for save csv ")
    
    return parser.parse_args()
if __name__ == "__main__":
    args = argument()
    
    # Scale Image And Save To New Folder 
    paths = os.listdir(args.path_images)
    if not os.path.exists(args.dest_images):
        print("Creating New Folder Images !!")
        os.makedirs(args.dest_images)
    for path in tqdm(paths, total=len(paths)):
        image = tf.io.read_file(os.path.join(args.path_images, path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [args.size, args.size])
        image = tf.cast(image, tf.uint8).numpy()
        if ".jpg" not in path:
            name = path.split(".")[]
            path = path + ".jpg"
        plt.imsave(os.path.join(args.dest_images ,path), image)

    
    # Hash Computing 
    hash_functions = [imagehash.average_hash,
                    imagehash.phash,
                    imagehash.dhash,
                    imagehash.whash]        
    image_ids = []
    hashes = []
    paths = tf.io.gfile.glob(os.path.join(args.dest_images ,"*.jpg"))
    for path in tqdm(paths, total=len(paths)):
        image = PIL.Image.open(path)
        hashes.append(np.array([x(image).hash for x in hash_functions]).reshape(-1,))
        image_ids.append(path.split('/')[-1])
    hashes = np.array(hashes)
    image_ids = np.array(image_ids)

    # Search Hashed Images 
    duplicate_ids = []
    for i in tqdm(range(len(hashes)), total=len(hashes)):
        similarity = (hashes[i] == hashes).mean(axis=1)
        duplicate_ids.append(list(image_ids[similarity > args.threshold]))
        
    duplicates = [frozenset([x] + y) for x, y in zip(image_ids, duplicate_ids)]
    duplicates = set([x for x in duplicates if len(x) > 1])


    # Addition pair in Plant CVPR 2021
    # duplicates_by_kingofarmy = {
    # frozenset(('8dbeda49894d522e.jpg', 'afbe5641896d522a.jpg')),
    # frozenset(('af6292db1b611d98.jpg', 'a56292dadb618d95.jpg')),
    # frozenset(('abf0b5a0df028b17.jpg', 'abf0b5819f028f0f.jpg')),
    # frozenset(('e385830ecacd2d9e.jpg', 'c335971e8acd609e.jpg')),
    # frozenset(('cebdc20f67838631.jpg', 'dfbdc047068b063d.jpg')),
    # frozenset(('f392f11919991cea.jpg', 'f196f11a99d91ce0.jpg'))}
    # duplicates |= duplicates_by_kingofarmy

    # Save the results 
    print(f'Found {len(duplicates)} duplicate pairs:')
    print('Writing duplicates to "duplicates.csv".')
    with open(os.path.join(args.save_csv_dir,'duplicates.csv'), 'w') as file:
        for row in duplicates:
            file.write(','.join(row) + '\n')

    # Print Result
    # for row in duplicates:
    #     print(', '.join(row))