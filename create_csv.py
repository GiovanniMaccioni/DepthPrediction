import torch
import torchvision

import data as D

def create_new_image_list(image_list):
    new_list = []
    for i in range(1, len(image_list)):
        img1 = torchvision.io.read_image(image_list[i-1]).to(torch.float32)
        img2 = torchvision.io.read_image(image_list[i]).to(torch.float32)
        diff = torch.nn.functional.l1_loss(img1, img2)

        if diff > 0:
            #TODO Keep only the relative path to the folder doiwnloaded from the dataset site
            new_list.append(image_list[i-1])

    return new_list

def write_list_to_txt(list_to_write, new_txt_path):
    with open(new_txt_path, 'w') as f:
        for elem in list_to_write:
            f.write(f"{elem}\n")
    return



t1 =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, sequence_length=1)
t2 =  D.BaxterJointsSynthDataset("./data/dataset", [1], "train", demo = False, sequence_length=1)

data_t1_train = t1.data['train']['depth8_file']
data_t1_val = t1.data['val']['depth8_file']

data_t2_train = t2.data['train']['depth8_file']
data_t2_val = t2.data['val']['depth8_file']

new_list1 = create_new_image_list(data_t1_train)
new_list2 = create_new_image_list(data_t2_train)

new_list3 = create_new_image_list(data_t1_val)
new_list4 = create_new_image_list(data_t2_val)

write_list_to_txt(new_list1 + new_list2, "./train_rec.txt")
write_list_to_txt(new_list3 + new_list4, "./val_rec.txt")



