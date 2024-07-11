import torch
import torchvision

import data as D

import os

def create_new_image_list(image_list):
    new_list = []
    for i in range(1, len(image_list)):
        img1 = torchvision.io.read_image(os.path.join("./", image_list[i-1])).to(torch.float32)
        img2 = torchvision.io.read_image(os.path.join("./", image_list[i])).to(torch.float32)
        diff = torch.nn.functional.l1_loss(img1, img2)

        if diff > 0:
            #TODO Keep only the relative path to the folder doiwnloaded from the dataset site
            new_list.append(image_list[i-1])

    return new_list

def two_fps(image_list):
    new_list = []
    counter = 0

    first_file = True
    current_camera = ""
    current_seq = ""
    current_sub = ""

    for i in range(0, len(image_list)):
        
        t1 = image_list[i].split("/")

        if first_file:
            current_camera = t1[4]
            current_seq = t1[5]
            current_sub = t1[6]
            first_file = False

        if (current_camera != t1[4]) or (current_seq != t1[5]) or (current_sub != t1[6]):
            current_camera = t1[4]
            current_seq = t1[5]
            current_sub = t1[6]

            new_list = new_list[:-2]

        t2 = t1[7].split("_")
        t3 = int(t2[0])
        t4 = str(int(t3+2e9))
        t5 = str(int(t3+7e9))
        t2[0] = t4
        final1 = "_".join(t2)
        t2[0] = t5
        final2 = "_".join(t2)
        
        final1 = "/".join(t1[:-1]+[final1])
        final2 = "/".join(t1[:-1]+[final2])

        new_list.append(final1)
        new_list.append(final2)

        counter = counter+1

        #if any of the directory name changes in the directory path, it means it is a different action of the robot(or the same from another view
        # but counted as different for training purposes)
        

    return new_list

def five_fps(image_list):
    new_list = []
    counter = 0

    first_file = True
    current_camera = ""
    current_seq = ""
    current_sub = ""

    for i in range(0, len(image_list)):
        
        t1 = image_list[i].split("/")

        if first_file:
            current_camera = t1[4]
            current_seq = t1[5]
            current_sub = t1[6]
            first_file = False

        if (current_camera != t1[4]) or (current_seq != t1[5]) or (current_sub != t1[6]):
            current_camera = t1[4]
            current_seq = t1[5]
            current_sub = t1[6]

            new_list = new_list[:-5]

        t2 = t1[7].split("_")
        t3 = int(t2[0])
        t4 = str(int(t3+2e9))
        t5 = str(int(t3+4e9))
        t6 = str(int(t3+6e9))
        t7 = str(int(t3+8e9))
        t3 = str(t3)

        t2[0] = t3
        final1 = "_".join(t2)
        t2[0] = t4
        final2 = "_".join(t2)
        t2[0] = t5
        final3 = "_".join(t2)
        t2[0] = t6
        final4 = "_".join(t2)
        t2[0] = t7
        final5 = "_".join(t2)
        
        final1 = "/".join(t1[:-1]+[final1])
        final2 = "/".join(t1[:-1]+[final2])
        final3 = "/".join(t1[:-1]+[final3])
        final4 = "/".join(t1[:-1]+[final4])
        final5 = "/".join(t1[:-1]+[final5])

        new_list.append(final1)
        new_list.append(final2)
        new_list.append(final3)
        new_list.append(final4)
        new_list.append(final5)

        counter = counter+1

        #if any of the directory name changes in the directory path, it means it is a different action of the robot(or the same from another view
        # but counted as different for training purposes)
        

    return new_list



def write_list_to_txt(list_to_write, new_txt_path):
    with open(new_txt_path, 'w') as f:
        for elem in list_to_write:
            f.write(f"{elem}\n")
    return



"""t1 =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, sequence_length=1)
t2 =  D.BaxterJointsSynthDataset("./data/dataset", [1], "train", demo = False, sequence_length=1)

data_t1_train = t1.data['train']['depth8_file']
data_t1_val = t1.data['val']['depth8_file']

data_t2_train = t2.data['train']['depth8_file']
data_t2_val = t2.data['val']['depth8_file']"""

t = D.RecDataset("./data/dataset", "train", sequence_length=1)
data_t_train = t.data['train']['depth8_file']
data_t_val = t.data['val']['depth8_file']

"""new_list1 = two_fps(data_t1_train)
new_list2 = two_fps(data_t2_train)

new_list3 = two_fps(data_t2_val)
new_list4 = two_fps(data_t2_val)"""

"""new_list1 = create_new_image_list(data_t1_train)
new_list2 = create_new_image_list(data_t2_train)

new_list3 = create_new_image_list(data_t1_val)
new_list4 = create_new_image_list(data_t2_val)"""

"""write_list_to_txt(new_list1 + new_list2, "./train_all_cut.txt")
write_list_to_txt(new_list3 + new_list4, "./val_all_cut.txt")"""


new_list1 = five_fps(data_t_train)
new_list2 = five_fps(data_t_val)


write_list_to_txt(new_list1, "./train_all_cut_5fps.txt")
write_list_to_txt(new_list2, "./val_all_cut_5fps.txt")
