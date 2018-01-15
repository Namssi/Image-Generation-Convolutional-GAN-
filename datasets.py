from __future__ import print_function


import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import pandas as pd
import pickle
import random




class CUBDataset(data.Dataset):
    def __init__(self, data_dir, train=True, embedding_type='cnn-rnn', imsize=64, transform=None, target_transform=None):
	
	self.data_dir = data_dir
	self.transform = transform
	self.target_transform = target_transform
	self.img_names = self.get_img_names(data_dir, train)
	self.imsize = imsize
	self.bbox = self.load_bbox()
	split = 'train'
	if not train:
	    split = 'test'
	split_dir = os.path.join(data_dir, split)
	self.embeddings = self.load_embedding(split_dir, embedding_type)


    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        #elif embedding_type == 'cnn-gru':
        #    embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        #elif embedding_type == 'skip-thought':
        #    embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape, data_dir)
        return embeddings


    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
	#print(filename_bbox)
        return filename_bbox


    def get_img_names(self, root, train):
        images_file_path = os.path.join(root, 'images/')

        all_images_list_path = os.path.join(root, 'images.txt')
        all_images_list = np.genfromtxt(all_images_list_path, dtype=str)
        train_test_list_path = os.path.join(root, 'train_test_split.txt')
        train_test_list = np.genfromtxt(train_test_list_path, dtype=int)

        imgs = []

        for idx,fname in all_images_list:
            #full_path = os.path.join(images_file_path, fname)
            idx = int(idx)-1
	    if train_test_list[idx, 1] == 1 and train:
                imgs.append(fname)#, int(fname[0:3]) - 1))
            elif train_test_list[idx, 1] == 0 and not train:
                imgs.append(fname)#, int(fname[0:3]) - 1))
        return imgs



    def __len__(self):
	return len(self.img_names)



    def __getitem__(self, index):
	key = self.img_names[index][:-4]

	if self.bbox is not None:#no need if else since it is only for CUB data
            bbox = self.bbox[key]
            data_dir = self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir


	embeddings = self.embeddings[index, :, :]
	img_name = '%s/images/%s.jpg' % (data_dir, key)
	img = self.get_img(img_name, bbox)

	embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding





