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

"""
Comment on dataset:

For each run, for each point of view, 20 sequences are recorded.
In the training split there are the 20 sequences divided in 10 subsequences, corresponding
to the 10 pick-n-place moves performed by the robot, each sampled at 10 FPS.
For each of the 20 sequences, only a point of view is chosen.
TOCHECK further explore the split composition
"""


def apply_depth_normalization_16bit_image(img, norm_type):
    """Applies normalization over 16bit depth images

    Parameters
    ----------
    img: np.array
        16bit depth image.
    norm_type: str
        Type of normalization (min_max, mean_std supported).

    Returns
    -------
    tmp: np.array
        Normalized and 16bit depth image (zero-centered in the case of mean_std normalization).

    """
    if norm_type == "min_max":
        min_value = 0
        max_value = 255#TOCHECK before there was 5000
        tmp = (img - min_value) / (max_value - min_value)
    elif norm_type == "mean_std":
        tmp = (img - img.mean()) / img.std()
    elif norm_type == "batch_mean_std":
        raise NotImplementedError
    elif norm_type == "dataset_mean_std":
        raise NotImplementedError
    elif norm_type == None:
        tmp = img
    else:
        raise NotImplementedError
    return tmp


def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class BaxterJointsSynthDataset(Dataset):
    def __init__(self, dataset_dir: Path, run: list, init_mode: str = 'train', norm_type: str = 'mean_std',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', demo: bool = False,
                 img_size: tuple = None, sequence_length: int = 5, time_horizon: int = 0, output_fps: int = 2) :
        """Load Baxter robot synthetic dataset

        Parameters
        ----------
        dataset_dir: Path
            Dataset path.
        run: list
            Synthetic run to load (0, 1, or both).
        init_mode: str
            Loading mode (train -> train/val set, test -> test set).
        img_size: str
            Image dimensions to which resize dataset images. If None,  the image is not resized, so it will the be 768x432
        norm_type: str
            Type of normalization (min_max, mean_std supported).
        TOCHECK surrounding_removals: bool
            Activate or deactivate removal of surrounding walls/objects.
        demo: bool
            Useful for loading a portion of the dataset when debugging.
        sequence_length: int
            number of frames to pick from the dataset. It can correspond to past+next(depth prediction) or past+current(depth reconstruction/estimation)
        time_horizion: float
            these parameter will indicate
        output_fps: int
            
        """
        assert init_mode in ['train', 'test']

        self.dataset_dir = Path(dataset_dir)
        self.run = run
        self._mode = init_mode
  
        self.img_size = img_size

        self.norm_type = norm_type


        self.depth_range = depth_range
        self.depth_range_type = depth_range_type

        """self.aug_type = aug_type
        self._aug_mode = aug_mode
        self.aug_mode = aug_mode
        self.noise = noise"""
        self.demo = demo

        self.sequence_length = sequence_length

        #Load the data
        self.data = self.load_data()

    def __len__(self):
        #TOCHECK Don't know if modifying the length like this is a BAD PRACTICE
        # self.mode contains 'train', 'val', or 'test'
        return self.data[self.mode]["len"]

    def __getitem__(self, idx):

        if self.sequence_length != 1:
            #TOCHECK Assuming that the sequence length is a dividor for the dataset length
            sample_sequence = []
            len_action = len(self.data[self.mode]["depth8_file"][idx])
            #random action indices. It will be between 0 and len(action) - self.sequence_length
            #FIXME set seeds!!
            index_act = np.random.randint(0, len_action - self.sequence_length + 1)

            for i in range(self.sequence_length):
                #FIXME now I have to access the key depth8_file

                #Before there was copy because it was an element of the dictionary
                """
                sample_sequence.append(self.data[self.mode][idx][index_act+i].copy())"""
                sample_sequence.append(self.data[self.mode]["depth8_file"][idx][index_act+i])#.copy())
                #sample = self.data[self.mode][idx].copy()

            depth_sequence = []
            #depthvis_sequence = []
            for sample in sample_sequence:

                # image loading (depth and/or RGB)
                #depth8_img = torchvision.io.read_image(sample['depth8_file']).to(torch.float32)
                depth8_img = torchvision.io.read_image(sample).to(torch.float32)

                depth8_img = TVF.center_crop(depth8_img, (392, 598))#(432, 512)

                if self.img_size != None:
                    depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)

                # image size divided by 32 should be an even value (for SH network)
                #TOCHECK I think other than to have the values divisible by 32, this removes the
                # black bands on the sides(but only if the slice operates on the width!!!) 
                #depth8_img = depth8_img[12:-12, :]

                # adapt depth image to "padding" depth range type
                """if self.depth_range_type == 'padding':
                    Z_min, Z_max, dZ = self.depth_range
                    new_img_h = (Z_max - Z_min) // dZ
                    padding = int(np.abs(depth8_img.shape[0] - new_img_h) // 2)
                    depth8_img = cv2.copyMakeBorder(depth8_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)"""

                #TODO convert this code to work with depth8img as a tensor
                # create visible depth map
                """depth8_img_vis = ((depth8_img * 255) / depth8_img.max()).astype(np.uint8)[..., None]
                depth8_img_vis = np.concatenate([depth8_img_vis, depth8_img_vis, depth8_img_vis], -1)"""
                #depth8_img_vis = None#FIXME for compatibility

                # depth map and keypoints normalization
                depth8_img = apply_depth_normalization_16bit_image(depth8_img, self.norm_type)

                #depth_sequence.append(torch.from_numpy(depth8_img[None, ...]))#Convert the numpy array to torch
                depth_sequence.append(depth8_img[None, ...])
                #depthvis_sequence.append(depth8_img_vis)

            depth_sequence = torch.cat(depth_sequence, dim=0)#concat the sequence in the first dimension

            output = {
                'depth8': depth_sequence#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
        
        else:

            sample = self.data[self.mode]["depth8_file"][idx]

            depth8_img = torchvision.io.read_image(sample).to(torch.float32)

            depth8_img = TVF.center_crop(depth8_img, (392, 598))#(432, 512)

            if self.img_size != None:
                depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)

            depth8_img = apply_depth_normalization_16bit_image(depth8_img, self.norm_type)

            output = {
                'depth8': depth8_img#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
            

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

    def load_data(self):
        if self._mode == 'train':
            splits = ['train', 'val']
        else:
            splits = ['test']

        #this initailization serves to instantiate an empty dictionary, to which the keys and values will
        #be added in the following for cycles
        data = defaultdict(list)

        for split in splits:
            actions = [] #this
            iter = 0
            for run in self.run:
                with (self.dataset_dir / 'splits' / f'{split}_rgb_run_{run}.txt').open('r') as fd:
                    rgb_files = fd.read().splitlines()
                if self.demo:
                    #I chose the dimensions to be divisible by 30, as every pick-and-move subsequence is composed by 30 depth images
                    if split == 'train':
                        rgb_files = rgb_files[:900]
                    elif split == 'val':
                        rgb_files = rgb_files[:300]
                    else:
                        rgb_files = rgb_files[:600]
                
                first_file = True
                current_camera = ""
                current_seq = ""
                current_sub = ""
                number_of_actions = 0 #This will indicate the total number of actions recorded for the robot.

                for rgb_file in tqdm(rgb_files, f'Loading {split} set run {run}'):
                        
                    info = rgb_file.split('/')
                    camera = info[-4]
                    seq = info[-3]
                    sub_seq = info[-2]
                    img_name = Path(rgb_file).stem

                    #If it is the first iteration, initialize the "current" variables to do comparison
                    if first_file:
                        current_camera = camera
                        current_seq = seq
                        current_sub = sub_seq
                        actions.append([])
                        first_file = False

                    #if any of the directory name changes, it means it is a different action of the robot(or the same from another view
                    # but counted as different for training purposes)
                    if (current_camera != camera) or (current_seq != seq) or (current_sub != sub_seq):
                        current_camera = camera
                        current_seq = seq
                        current_sub = sub_seq
                        actions.append([])
                        number_of_actions += 1


                    #TODO From here take the parent directory of the image file(.jpg) and use it to divide the sequences
                    #The data structure could be a nested list where each element of the main list will be a sequence 
                    #

                    # camera, depth, joints and picknplace file and data
                    #TOCHECK Had to add these
                    depth8_file = rgb_file.replace('/nas/softechict-nas-3/asimoni/Baxter/simulation', './data/dataset')
                    depth8_file = Path(depth8_file.replace('rgb', 'depth8_registered'))
                    depth8_img_name = f"{img_name.replace('color', 'depth')}.png"
                    depth8_file = str(depth8_file.parents[0] / f'{depth8_img_name}')

                    iter += 1

                    actions[number_of_actions].append(depth8_file)#FIXME if I pass both run ([0,1]) here I have an error

                #TODO unwrap the list if sequence length == 1
                if self.sequence_length == 1:
                    x = actions
                    actions = []
                    #actions = [actions + elem for elem in x]
                    actions = [item for sublist in x for item in sublist]

                sample = {
                        'depth8_file': actions,
                        'len': len(actions)
                    }

                data[split] = sample

        return data
    

class RecDataset(Dataset):
    def __init__(self, dataset_dir: Path, run: list, init_mode: str = 'train', norm_type: str = 'mean_std',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', demo: bool = False,
                 img_size: tuple = None, sequence_length: int = 5, time_horizon: int = 0, output_fps: int = 2, transforms = None) :
        """Load Baxter robot synthetic dataset

        Parameters
        ----------
        dataset_dir: Path
            Dataset path.
        run: list
            Synthetic run to load (0, 1, or both).
        init_mode: str
            Loading mode (train -> train/val set, test -> test set).
        img_size: str
            Image dimensions to which resize dataset images. If None,  the image is not resized, so it will the be 768x432
        norm_type: str
            Type of normalization (min_max, mean_std supported).
        TOCHECK surrounding_removals: bool
            Activate or deactivate removal of surrounding walls/objects.
        demo: bool
            Useful for loading a portion of the dataset when debugging.
        sequence_length: int
            number of frames to pick from the dataset. It can correspond to past+next(depth prediction) or past+current(depth reconstruction/estimation)
        time_horizion: float
            these parameter will indicate
        output_fps: int
            
        """
        assert init_mode in ['train', 'test']

        self.dataset_dir = Path(dataset_dir)
        self.run = run
        self._mode = init_mode
  
        self.img_size = img_size

        self.norm_type = norm_type


        self.depth_range = depth_range
        self.depth_range_type = depth_range_type

        """self.aug_type = aug_type
        self._aug_mode = aug_mode
        self.aug_mode = aug_mode
        self.noise = noise"""
        self.demo = demo

        self.sequence_length = sequence_length

        #Load the data
        self.data = self.load_data()

        self.transforms = transforms

    def __len__(self):
        #TOCHECK Don't know if modifying the length like this is a BAD PRACTICE
        # self.mode contains 'train', 'val', or 'test'
        return self.data[self.mode]["len"]

    def __getitem__(self, idx):

        if self.sequence_length != 1:
            #TOCHECK Assuming that the sequence length is a dividor for the dataset length
            sample_sequence = []
            len_action = len(self.data[self.mode]["depth8_file"][idx])
            #random action indices. It will be between 0 and len(action) - self.sequence_length
            #FIXME set seeds!!
            index_act = np.random.randint(0, len_action - self.sequence_length + 1)

            for i in range(self.sequence_length):
                #FIXME now I have to access the key depth8_file

                #Before there was copy because it was an element of the dictionary
                """
                sample_sequence.append(self.data[self.mode][idx][index_act+i].copy())"""
                sample_sequence.append(self.data[self.mode]["depth8_file"][idx][index_act+i])#.copy())
                #sample = self.data[self.mode][idx].copy()

            depth_sequence = []
            #depthvis_sequence = []
            for sample in sample_sequence:

                # image loading (depth and/or RGB)
                #depth8_img = torchvision.io.read_image(sample['depth8_file']).to(torch.float32)
                depth8_img = torchvision.io.read_image(sample).to(torch.float32)

                depth8_img = TVF.center_crop(depth8_img, (392, 598))#(432, 512)

                if self.img_size != None:
                    depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)

                # image size divided by 32 should be an even value (for SH network)
                #TOCHECK I think other than to have the values divisible by 32, this removes the
                # black bands on the sides(but only if the slice operates on the width!!!) 
                #depth8_img = depth8_img[12:-12, :]

                # adapt depth image to "padding" depth range type
                """if self.depth_range_type == 'padding':
                    Z_min, Z_max, dZ = self.depth_range
                    new_img_h = (Z_max - Z_min) // dZ
                    padding = int(np.abs(depth8_img.shape[0] - new_img_h) // 2)
                    depth8_img = cv2.copyMakeBorder(depth8_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)"""

                #TODO convert this code to work with depth8img as a tensor
                # create visible depth map
                """depth8_img_vis = ((depth8_img * 255) / depth8_img.max()).astype(np.uint8)[..., None]
                depth8_img_vis = np.concatenate([depth8_img_vis, depth8_img_vis, depth8_img_vis], -1)"""
                #depth8_img_vis = None#FIXME for compatibility

                # depth map and keypoints normalization
                depth8_img = apply_depth_normalization_16bit_image(depth8_img, self.norm_type)

                #depth_sequence.append(torch.from_numpy(depth8_img[None, ...]))#Convert the numpy array to torch
                depth_sequence.append(depth8_img[None, ...])
                #depthvis_sequence.append(depth8_img_vis)

            depth_sequence = torch.cat(depth_sequence, dim=0)#concat the sequence in the first dimension

            output = {
                'depth8': depth_sequence#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
        
        else:

            sample = self.data[self.mode]["depth8_file"][idx]

            depth8_img = torchvision.io.read_image(sample).to(torch.float32)

            depth8_img = TVF.center_crop(depth8_img, (392, 598))#(432, 512)

            if self.img_size != None:
                depth8_img = TVF.resize(depth8_img, (self.img_size[1], self.img_size[0]), interpolation = TVF.InterpolationMode.NEAREST)

            depth8_img = apply_depth_normalization_16bit_image(depth8_img, self.norm_type)

            if self.transforms:
                depth8_img = self.transforms(depth8_img)

            output = {
                'depth8': depth8_img#,#Add the batch dimension
                #'depth8vis': depth8_img_vis
            }

            return output
            

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

    def load_data(self):
        if self._mode == 'train':
            splits = ['train', 'val']
        else:
            splits = ['test']

        #this initailization serves to instantiate an empty dictionary, to which the keys and values will
        #be added in the following for cycles
        data = defaultdict(list)

        for split in splits:
            actions = [] #this
            iter = 0

            with (self.dataset_dir / 'splits' / f'{split}_rec.txt').open('r') as fd:
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

                #if any of the directory name changes, it means it is a different action of the robot(or the same from another view
                # but counted as different for training purposes)
                if (current_camera != camera) or (current_seq != seq) or (current_sub != sub_seq):
                    current_camera = camera
                    current_seq = seq
                    current_sub = sub_seq
                    actions.append([])
                    number_of_actions += 1


                #TODO From here take the parent directory of the image file(.jpg) and use it to divide the sequences
                #The data structure could be a nested list where each element of the main list will be a sequence 
                #

                # camera, depth, joints and picknplace file and data
                #TOCHECK Had to add these
                """depth8_file = rgb_file.replace('/nas/softechict-nas-3/asimoni/Baxter/simulation', './data/dataset')
                depth8_file = Path(depth8_file.replace('rgb', 'depth8_registered'))
                depth8_img_name = f"{img_name.replace('color', 'depth')}.png"
                depth8_file = str(depth8_file.parents[0] / f'{depth8_img_name}')"""

                iter += 1

                actions[number_of_actions].append(depth_file)#FIXME if I pass both run ([0,1]) here I have an error

            #TODO unwrap the list if sequence length == 1
            if self.sequence_length == 1:
                x = actions
                actions = []
                #actions = [actions + elem for elem in x]
                actions = [item for sublist in x for item in sublist]

            sample = {
                    'depth8_file': actions,
                    'len': len(actions)
                }

            data[split] = sample

        return data