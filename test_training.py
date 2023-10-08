import torch
import models as M
import data as D
import train as T


#Hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#x = torch.randn(1, 5*1, 424, 512)#[batch_size, sequence_len*channels, height, width]

trainset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = True, img_size=(512, 424), sequence_length=1)
trainset.train()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        shuffle=True, num_workers=4,
                                        worker_init_fn=D.init_worker, drop_last=True),


valset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = True, img_size=(512, 424), sequence_length=1)
valset.eval()
valloader =  torch.utils.data.DataLoader(valset, batch_size=1,
                                    shuffle=False, num_workers=4,
                                    worker_init_fn=D.init_worker, drop_last=True)


#TOCHECK I don't know if I have to do the XYZ transformation as said in the paper
encoder = M.Encoder_3d()
decoder = M.Decoder_3d()
model = M.Autoencoder_3d(encoder, decoder).to(device)

#I will test pdepth estimation first. To change the test for now, change the function call in the train function in train.py

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), 3*1e-3)

#FIXME Don't knw why I have a tuple as the trainloader and have to access it by addressing it
T.train(model, trainloader[0], valloader, criterion, optimizer, device, 5)




