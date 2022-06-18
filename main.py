from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from model import MCEVAE as mtie_model
from train import Train
from load_data import LoadData

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--transformation", help = "Transformation type: use so2 or se2", default = "so2")
    # parser.add_argument("--loss_type", help = "Reconstruction Loss type: use mse or bce", default = "bce")
    # parser.add_argument("--nCat", help = "Number of Categories", default = 10)
    # parser.add_argument("--nVar", help = "Number of Variational Latent Dimensions", default = 3)
    # parser.add_argument("--nBatch", help = "Batch size", default = 100)
    # parser.add_argument("--nEpochs", help = "Number of Epochs", default = 60)
    # parser.add_argument("--nHiddenCat", help = "Number of Nodes in Hidden Layers for Categorical Latent Space", default = 512)
    # parser.add_argument("--nHiddenVar", help = "Number of Nodes in Hidden Layers for Variational Latent Space", default = 512)
    # parser.add_argument("--nHiddenTrans", help = "Number of Nodes in Hidden Layers for Transformational Latent Space", default = 32)
    # parser.add_argument("--tag", help = "tag for model name", default = "default")
    # parser.add_argument("--training_mode", help = "Training mode: use supervised or unsupervised", default = "supervised")
    # parser.add_argument("--beta", help = "Beta for beta-VAE training", default = 1.0)
    # args = parser.parse_args()
    #ld = LoadData()
    #ld.get_datasets()

    print('loading data...')
    c = '/DAT/'
    transformation = "se2"
    nBatch = 256
    tag = "default"
    nCat = 10
    nVar = 3
    loss_type = "bce"
    nHiddenCat = 512
    nHiddenVar = 512
    nHiddenTrans = 32
    training_mode = "unsupervised"
    nEpochs = 10
    beta = 1.0
    
    mnist_SE2 = np.load('MTIE-VAE/DAT/mnist_se2.npy')
    mnist_SE2_test = np.load('MTIE-VAE/DAT/mnist_se2_test.npy')[:1000]
    mnist_SE2_init = np.load('MTIE-VAE/DAT/mnist_se2_init.npy')
    mnist_SE2_init_test = np.load('MTIE-VAE/DAT/mnist_se2_init_test.npy')[:1000]

    print('preparing dataset')
    batch_size = int(nBatch)
    trans_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2), torch.from_numpy(mnist_SE2_init))
    trans_loader = torch.utils.data.DataLoader(trans_dataset, batch_size=batch_size)
    trans_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2_test),
                                                        torch.from_numpy(mnist_SE2_init_test))
    trans_test_loader = torch.utils.data.DataLoader(trans_test_dataset, batch_size=batch_size)
    in_size = aug_dim = 28 * 28
    mode = transformation.upper()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = str(tag)
    mtie_model_var = mtie_model(in_size=in_size,
                                aug_dim=aug_dim,
                                latent_z_c=int(nCat),
                                latent_z_var=int(nVar),
                                mode=mode,
                                invariance_decoder='gated',
                                rec_loss=str(loss_type),
                                div='KL',
                                in_dim=1,
                                out_dim=1,
                                hidden_z_c=int(nHiddenCat),
                                hidden_z_var=int(nHiddenVar),
                                hidden_tau=int(nHiddenTrans),
                                activation=nn.Sigmoid,
                                training_mode=str(training_mode),
                                device=device,
                                tag=tag).to(device)

    train_obj = Train(mtie_model_var=mtie_model_var)

    lr = 1e-3
    optim = torch.optim.Adam(train_obj.model.parameters(), lr=lr)
    train_obj.train(
        optim=optim,
        train_data=trans_loader,
        test_data=trans_test_loader,
        num_epochs=int(nEpochs),
        tr_mode='new',
        beta=float(beta))
