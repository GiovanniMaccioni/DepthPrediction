import argparse
import lstm_training

config ={
    'batch_size': 32,
    'seed': 10,
    'lr': 5e-4,
    'sequence_length':15,
    'train_img_size': (256, 256),#(512, 432), #(512, 424),None, (392, 392)
    'val_img_size': (256, 256),#(512, 432), #(512, 424),None, (392, 392)
    'epochs': 2,
    'scheduler': "",
    'scheduler_parameters':[],
    'dir_path': "./run_lstm/test",
    'hid_dim': 1024,
    'n_layers': 2,
    'seq_transform': False,
    'fps_augmentation': False
}

if __name__=="__main__":

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Autoencoder2D')
    parser.add_argument("--dir_path", type=str, default="run_lstm/temp", help='')

    #Arguments
    parser.add_argument("--seed", type=int, default=10, help='Set the seed for reproducibility')
    parser.add_argument("--batch_size", type=int, default=16, help='Batch Size')
    parser.add_argument("--epochs", type=int, default=100, help='Number of Epochs')
    parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate')
    parser.add_argument("--scheduler", type=str, default="None", help='Scheduler to use')
    parser.add_argument("--scheduler_parameters", type=float, nargs='+', default=None, help='parameter for the chosen scheduler')

    

    #Arguments for images
    parser.add_argument("--train_img_size", type=int, nargs='+', default=(256, 256), help='')
    parser.add_argument("--val_img_size", type=int,nargs='+', default=(256, 256), help='')
    parser.add_argument("--random_crop", type=bool, default=False, help='')
    parser.add_argument("--img_transform", type=bool, default=False, help='')
    parser.add_argument("--img_loss_weights", type=float, nargs='+', default=[1.0, 0.1, 0.1], help='Weights for the losses')
    parser.add_argument("--img_diff_loss", type=str, default="None", help='Weights for the losses')
    parser.add_argument("--img_diff_loss_weight", type=float, default=0., help='Weights for the losses')

    #Arguments for LSTM
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden dimension LSTM')
    parser.add_argument("--n_layers", type=int, default=1, help='Number of Layers LSTM')
    parser.add_argument("--teacher_forcing", type=float, default=1.0, help='Number of Layers LSTM')

    #Sequence Augmentation
    parser.add_argument("--sequence_length", type=int, default=10, help='Sequence length')
    parser.add_argument("--time_step", type=int, default=4, help='Time Steps')
    parser.add_argument("--seq_transform", type=bool, default=False, help='')
    parser.add_argument("--fps", type=int, default=2, help='')
    parser.add_argument("--fps_augmentation", type=bool, default=False, help='')
    parser.add_argument("--feature_loss", type=str, default="None", help='')
    parser.add_argument("--feature_loss_weight", type=float, default=0., help='')
    #parser.add_argument("--diff_feat_loss", type=str, default=None, help='')

    args = parser.parse_args()

    config['dir_path']=args.dir_path

    #Populate config
    config['seed'] = args.seed
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['scheduler'] = args.scheduler
    config['scheduler_parameters'] = args.scheduler_parameters
    
    config['train_img_size']=args.train_img_size
    config['val_img_size']=args.val_img_size
    config['img_loss_weights'] = args.img_loss_weights
    config['img_diff_loss'] = args.img_diff_loss
    config['img_diff_loss_weight'] = args.img_diff_loss_weight

    config['hid_dim'] = args.hid_dim
    config['n_layers'] = args.n_layers

    config['sequence_length'] = args.sequence_length
    config['time_step'] = args.time_step
    config['seq_transform'] = args.seq_transform
    config['fps_augmentation'] = args.fps_augmentation
    config['feature_loss'] = args.feature_loss
    config['feature_loss_weight'] = args.feature_loss_weight
    config["teacher_forcing"] = args.teacher_forcing
    #config['diff_feat_loss'] = args.diff_feat_loss

    
    lstm_training.run(config)