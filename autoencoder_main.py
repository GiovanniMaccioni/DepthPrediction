import argparse
import autoencoder_training

config ={
    'batch_size': 256,
    'seed': 10,
    'lr': 1e-3,
    'sequence_length':1,
    'train_img_size': (256, 256),#(512, 432), #(512, 424),None, (392, 392)
    'val_img_size': (256, 256),#(512, 432), #(512, 424),None, (392, 392)
    'epochs': 30,
    'loss_weigths': [1., 0., 0.],
    'scheduler': "None",
    'scheduler_parameters':[],
    'random_crop': False,
    'dir_path': "",
    'transforms': False
}

if __name__=="__main__":

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Autoencoder2D')
    parser.add_argument("--dir_path", type=str, default="runs/temp", help='')

    #Arguments
    parser.add_argument("--seed", type=int, default=10, help='Set the seed for reproducibility')
    parser.add_argument("--batch_size", type=int, default=256, help='Batch Size')
    parser.add_argument("--epochs", type=int, default=25, help='Number of Epochs')
    parser.add_argument("--lr", type=float, default=5e-4, help='Learning rate')
    parser.add_argument("--scheduler", type=str, default="None", help='Scheduler to use')
    parser.add_argument("--scheduler_parameters", type=float, nargs='+', default=None, help='parameter for the chosen scheduler')
    parser.add_argument("--loss_weigths", type=float, nargs='+', default=[1., 1., 1.], help='Weights for the losses')

    #Arguments for images
    parser.add_argument("--train_img_size", type=int, nargs='+', default=(256, 256), help='')
    parser.add_argument("--val_img_size", type=int,nargs='+', default=(256, 256), help='')
    parser.add_argument("--img_transform", type=bool, default=False, help='')
    parser.add_argument("--random_crop", type=bool, default=False, help='')

    args = parser.parse_args()

    config['dir_path']=args.dir_path

    #Populate config
    config['seed'] = args.seed
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['scheduler'] = args.scheduler
    config['scheduler_parameters'] = args.scheduler_parameters
    config['loss_weigths'] = args.loss_weigths
    
    config['train_img_size']=args.train_img_size
    config['val_img_size']=args.val_img_size
    config['img_transform']=args.img_transform
    config['random_crop']=args.random_crop
    
    autoencoder_training.run(config)