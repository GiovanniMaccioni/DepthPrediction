import json
from collections import defaultdict
from pathlib import Path

#import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision
import torchvision.transforms.functional as TVF

import os

"""
Comment on dataset:

For each run, for each point of view, 20 sequences are recorded.
In the training split there are the 20 sequences divided in 10 subsequences, corresponding
to the 10 pick-n-place moves performed by the robot, each sampled at 10 FPS.
For each of the 20 sequences, only a point of view is chosen.
TOCHECK further explore the split composition
"""
def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class RecDataset(Dataset):
    def __init__(self, dataset_dir: Path, init_mode: str = 'train',
                 img_size: tuple = None, sequence_length: int = 5, fps: int = 2, seq_transform = None, img_transform = None, fps_augmentation=False) :
        """Load Baxter robot synthetic dataset

        Parameters
        ----------
        dataset_dir: Path
            Dataset path.
        init_mode: str
            Loading mode (train -> train/val set, test -> test set).
        img_size: str
            Image dimensions to which resize dataset images. If None,  the image is not resized, so it will the be 768x432
        norm_type: str
            Type of normalization (min_max, mean_std supported).
        sequence_length: int
            number of frames to pick from the dataset. It can correspond to past+next(depth prediction) or past+current(depth reconstruction/estimation)
        output_fps: int            
        """
        assert init_mode in ['train', 'test']

        self.dataset_dir = Path(dataset_dir)

        self._mode = init_mode
  
        self.img_size = img_size

        self.sequence_length = sequence_length

        self.fps_augmentation = fps_augmentation

        #Load the data
        if fps_augmentation:
            self.data = self.load_data("_all_cut_2fps.txt")
            self.data1fps = self.load_data("_all_cut_1fps.txt")
            self.data5fps = self.load_data("_all_cut_5fps.txt")
        else:
            self.data = self.load_data(f"_all_cut_{fps}fps.txt")
            print(f"FPS:{fps}")
        
        self.seq_transform = seq_transform
        self.img_transform = img_transform
        self.fps = fps

    def __len__(self):
        #TOCHECK Don't know if modifying the length like this is a BAD PRACTICE
        # self.mode contains 'train', 'val', or 'test'
        return self.data[self.mode]["len"]

    def __getitem__(self, idx):
        if self.sequence_length != 1:
            #TOCHECK Assuming that the sequence length is a dividor for the dataset length
            sample_sequence = []
            data = self.data

            r = np.random.random()  
            if self.fps_augmentation:
                if r < 0.5:
                    data = self.data
                else:
                    if r < 0.75:
                        data = self.data1fps
                    else:
                        data = self.data5fps
            
            index_act = 5#3*self.fps??


            if self._mode == 'train':
                #random action indices. It will be between 0 and len(action) - self.sequence_length
                len_action = len(data[self.mode]["depth8_file"][idx])
                index_act = np.random.randint(0, len_action - self.sequence_length + 1)
                """else:
                #splits = ['test']
                print("ciao")"""
            
            for i in range(self.sequence_length):
                #FIXME now I have to access the key depth8_file
                #Before there was copy because it was an element of the dictionary
                sample_sequence.append(data[self.mode]["depth8_file"][idx][index_act+i])#.copy())
                #sample = self.data[self.mode][idx].copy()

            depth_sequence = []
            #depthvis_sequence = []
            for sample in sample_sequence:

                #image loading (depth and/or RGB)
                #depth8_img = torchvision.io.read_image(sample['depth8_file']).to(torch.float32)
                #Read directly depth images with pixel values from 0 to 255
                depth8_img = torchvision.io.read_image(sample).to(torch.float32)
                
                depth8_img = TVF.center_crop(depth8_img, (424, 512))#(392, 598)
                if self.img_size != None:
                    depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)


                #Bring the values to the range 0 to 1
                depth8_img = depth8_img/255.

                #depth_sequence.append(torch.from_numpy(depth8_img[None, ...]))#Convert the numpy array to torch
                depth_sequence.append(depth8_img[None, ...])
                #depthvis_sequence.append(depth8_img_vis)

            depth_sequence = torch.cat(depth_sequence, dim=0)#concat the sequence in the first dimension

            #transform depth sequence with custom sequence augmentation
            if self.img_transform:
                depth_sequence = self.transform_images(depth_sequence)
            if self.seq_transform:
                depth_sequence = self.transform_sequence(depth_sequence)

            output = {
                'depth8': depth_sequence#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
        
        else:

            sample = self.data[self.mode]["depth8_file"][idx]

            #Read directly depth images with pixel values from 0 to 255
            depth8_img = torchvision.io.read_image(sample).to(torch.float32)

            depth8_img = TVF.center_crop(depth8_img, (424, 512))#(392, 598)
            if self.img_size != None:
                depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)

            #Bring the values to the range 0 to 1
            depth8_img = depth8_img/255.

            if self.img_transform:#FIXME a variable used to contain two different types of object
                depth8_img = self.img_transform(depth8_img)

            output = {
                'depth8': depth8_img#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
        
    def transform_sequence(self, depths_sequence):
        #Invert sequence
        if np.random.random() > 0.5:
            depths_sequence = depths_sequence.flip(dims=(0,))

        return depths_sequence
    
    def transform_images(self, depths_sequence):

        if np.random.random() > 0.5:
            for i in range(depths_sequence.shape[0]):
                depths_sequence[i] = TVF.hflip(depths_sequence[i])

        if np.random.random() > 0.5:
            corners = torchvision.transforms.RandomPerspective.get_params(width=self.img_size[1], height=self.img_size[0], distortion_scale=0.1)
            for i in range(depths_sequence.shape[0]):
                depths_sequence[i] = TVF.perspective(depths_sequence[i], corners[0], corners[1], interpolation=TVF.InterpolationMode.NEAREST)#, interpolation=TVF.InterpolationMode.NEAREST)

        if np.random.random() > 0.5:
            angle = torchvision.transforms.RandomRotation.get_params(degrees=(-45, 45))
            for i in range(depths_sequence.shape[0]):
                depths_sequence[i] = TVF.rotate(depths_sequence[i], angle)

        return depths_sequence
            

    def train(self):
        self.mode = 'train'
        """if self._aug_mode:
            self.aug_mode = True"""

    def eval(self):
        self.mode = 'val'
        #self.aug_mode = False

    def test(self):
        self.mode = 'test'
        #self.aug_mode = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'test']
        self._mode = value

    def load_data(self, end_file_name):
        if self._mode == 'train':
            splits = ['train', 'val']
        else:
            splits = ['test']

        #this initailization serves to instantiate an empty dictionary, to which the keys and values will
        #be added in the following for cycles
        data = defaultdict(list)

        for split in splits:
            actions = [] #this

            #with (self.dataset_dir / 'splits' / f'{split}_all_cut.txt').open('r') as fd:
            #with (self.dataset_dir / 'splits' / f'{split}{end_file_name}').open('r') as fd:
            with open(os.path.join(self.dataset_dir, 'splits',f'{split}'+end_file_name), 'r') as fd:
                depth_files = fd.read().splitlines()

            first_file = True
            current_camera = ""
            current_seq = ""
            current_sub = ""
            number_of_actions = 0 #This will indicate the total number of actions recorded for the robot.

            for depth_file in tqdm(depth_files, f'Loading {split}'):
                    
                info = depth_file.split('/')
                camera = info[-4]
                seq = info[-3]
                sub_seq = info[-2]
                #img_name = Path(depth_file).stem

                #If it is the first iteration, initialize the "current" variables to do comparison
                if first_file:
                    current_camera = camera
                    current_seq = seq
                    current_sub = sub_seq
                    actions.append([])
                    first_file = False

                #if any of the directory name changes in the directory path, it means it is a different action of the robot(or the same from another view
                # but counted as different for training purposes)
                if (current_camera != camera) or (current_seq != seq) or (current_sub != sub_seq):
                    current_camera = camera
                    current_seq = seq
                    current_sub = sub_seq
                    actions.append([])
                    number_of_actions += 1

                actions[number_of_actions].append(depth_file)

            #TODO unwrap the list if sequence length == 1
            if self.sequence_length == 1:
                x = actions
                actions = []#TOCHECK
                #actions = [actions + elem for elem in x]
                actions = [item for sublist in x for item in sublist]

            sample = {
                    'depth8_file': actions,
                    'len': len(actions)
                }

            data[split] = sample

        return data