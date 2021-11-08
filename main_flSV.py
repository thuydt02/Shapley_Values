import argparse

from flSV import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--run', dest='run', help='run number, to create a folder for outputs', type=int, default=0)
    parser.add_argument('--num-clients', dest='num_clients', help='Number of clients', type=int, default=100)
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=0.003)
    parser.add_argument('--lr-decay', dest='lr_decay', help='LEARNING RATE DECAY', type=float, default=0.995)
    parser.add_argument('--client-epochs', dest='client_epochs', help='Number local epochs, default = 10', type=int, default=10)
    parser.add_argument('--server-epochs', dest='server_epochs', help='Number of epoch to train the FL model in server, default = 2000', type=int, default=2000)
    
    parser.add_argument('--dataset', dest='dataset', help='synthetic_regression', type=str, default='synthetic_regression')
    parser.add_argument('--input-dim', dest='input_dim', help='input_dim', type=int, default=3)
    parser.add_argument('--output-dim', dest='output_dim', help='output_dim', type=int, default=1)
    
    
    parser.add_argument('--data-size-file', dest='data_size_file', help='The data distribution (in batches, csv format) of clients', type=str, default="") 

    return parser.parse_args()

def main(args):


    BATCH_SIZE = 10 #10 for mnist 20 for cifar10

    DATASET = args.dataset #"mnist"#"mnist"#"femnist" #"mnist"#"femnist"#"mnist"#"cifar100"#"femnist"
    #OUT_DIR = "/project/umb_duc_tran/thuy/flSV/output/" on ghpcc06
    #OUT_DIR = "./output/" # local

    OUT_DIR = "/content/drive/MyDrive/flSV/output/"
    OUT_DIR = "/content/drive/MyDrive/flSV/output/"
    
    dataset_dir = "/content/drive/MyDrive/flSV/data/"
    dataset_dir = "/content/drive/MyDrive/flSV/data/"
    w_DIR = OUT_DIR + DATASET + "/weight/"
    z_DIR = OUT_DIR + DATASET + "/z_ass/"
    acc_DIR = OUT_DIR + DATASET + "/acc/"


    CL_PERCENTAGES = 1

    PRE_TRAINED_W_FILE = None

    #w_DIR ="../../weight/" # local
    #z_DIR = "../../z_ass/" # local
    #acc_DIR = "../../acc/" # local

    train_fname = ''
    test_fname = ''

    if args.dataset == 'synthetic_regression':
        train_fname = dataset_dir + args.dataset + "/" + "train_XY_linear_sample4800_feature3.csv"
        test_fname = dataset_dir + args.dataset + "/" + "test_XY_linear_sample960_feature3.csv"

    
    trainer = Trainer(n_clients = args.num_clients, learning_rate = args.lr, lr_decay = args.lr_decay, batch_size = BATCH_SIZE, 
        epochs = args.server_epochs, n_local_epochs = args.client_epochs, cl_percentages = CL_PERCENTAGES,  data_size_file = args.data_size_file,
        w_dir = w_DIR, acc_dir = acc_DIR, z_dir = z_DIR, 
        train_fname = train_fname, test_fname = test_fname, input_dim = args.input_dim, output_dim=args.output_dim,
        pre_trained_w_file = PRE_TRAINED_W_FILE)
    trainer.run_train()

if __name__ == '__main__':
    main(parse_args())



