# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
# from sklearn.manifold import TSNE


class Train:
    def __init__(self, mtie_model_var):
        self.model = mtie_model_var
        self.prefix = 'MTIE-VAE/DAT'

    # def create_image(self, z_var_q, epoch, modelname):
    #     y = torch.from_numpy(np.load(self.prefix + '/mnist_' + 'se2' + '_' + 'target' + '.npy'))[:10000]

    #     if z_var_q.shape[0] > 0:
    #         tsne_features = TSNE(n_components=2).fit_transform(z_var_q.cpu().numpy())

    #         fig = plt.figure(figsize=(10, 6))

    #         plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y[:tsne_features.shape[0]], marker='o',
    #                     edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s=10)
    #         plt.grid(False)
    #         plt.axis('off')
    #         plt.colorbar()
    #         plt.savefig("images/" + "epoch_" + str(epoch) + "_" + modelname.replace("_checkpoint", "_clustermap.png"))
    #         plt.show()


    def calc_loss(self, x, x_init, beta=1., n_sample=4):
        # print('x is ', x.size())
        # x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, \
        # z_c_q, z_c_q_mu, z_c_q_logvar, z_c_q_L, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = self.model(x)
        x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = self.model(x)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = x.view(-1, self.model.in_size).to(device)
        x_hat = x_hat.view(-1, self.model.in_size)
        x_rec = x_rec.view(-1, self.model.in_size)

        if self.model.rec_loss == 'mse':
            RE = torch.sum((x - x_hat) ** 2)
            if self.model.tau_size > 0 and self.model.training_mode == 'supervised':
                RE_INV = torch.sum((x_rec - x_init) ** 2)
            elif self.model.tau_size > 0 and self.model.training_mode == 'unsupervised':
                RE_INV = torch.FloatTensor([0.]).to(device)
                for jj in range(25):
                    with torch.no_grad():
                        x_arb = self.model.get_x_ref(
                            x.view(-1, 1, int(np.sqrt(self.model.in_size)), int(np.sqrt(self.model.in_size))),
                            tau_q)
                        z_aug_arb = self.model.aug_encoder(x_arb)
                        z_c_q_mu_arb, z_c_q_logvar_arb, _ = self.model.q_z_c(z_aug_arb)
                        z_c_q_arb = self.model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                        z_var_q_mu_arb, z_var_q_logvar_arb = self.model.q_z_var(z_aug_arb)
                        z_var_q_arb = self.model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                        x_init, _ = self.model.reconstruct(z_var_q_arb, z_c_q_arb)
                        x_init = x_init.view(-1, self.model.in_size).to(device)
                        x_init = torch.clamp(x_init, 1.e-5, 1 - 1.e-5)
                    RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q) ** 2)
                    #RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q) ** 2)
                    RE_INV = RE_INV + torch.sum((x_rec - x_init) ** 2)
                RE_INV = RE_INV / 25.0
            else:
                RE_INV = torch.FloatTensor([0.]).to(device)
        elif self.model.rec_loss == 'bce':
            x_hat = torch.clamp(x_hat, 1.e-5, 1 - 1.e-5)
            x = torch.clamp(x, 1.e-5, 1 - 1.e-5)
            x_init = torch.clamp(x_init, 1.e-5, 1 - 1.e-5)
            x_rec = torch.clamp(x_rec, 1.e-5, 1 - 1.e-5)
            RE = -torch.sum((x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat)))
            if self.model.tau_size > 0 and self.model.training_mode == 'supervised':
                x_init = x_init.view(-1, self.model.in_size).to(device)
                RE_INV = -torch.sum((x_init * torch.log(x_rec) + (1 - x_init) * torch.log(1 - x_rec)))
            elif self.model.tau_size > 0 and self.model.training_mode == 'unsupervised':
                RE_INV = torch.FloatTensor([0.]).to(device)
                for jj in range(25):
                    with torch.no_grad():
                        x_arb = self.model.get_x_ref(
                            x.view(-1, 1, int(np.sqrt(self.model.in_size)), int(np.sqrt(self.model.in_size))),
                            tau_q)
                        z_aug_arb = self.model.aug_encoder(x_arb)
                        # z_c_q_mu_arb, z_c_q_logvar_arb, _ = self.model.q_z_c(z_aug_arb)
                        # z_c_q_arb = self.model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                        z_var_q_mu_arb, z_var_q_logvar_arb = self.model.q_z_var(z_aug_arb)
                        z_var_q_arb = self.model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                        x_init, _ = self.model.reconstruct_b(z_var_q_arb)
                        x_init = x_init.view(-1, self.model.in_size).to(device)
                        x_init = torch.clamp(x_init, 1.e-5, 1 - 1.e-5)
                    RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q) ** 2)
                    # RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q)**2)
                    RE_INV = RE_INV - torch.sum((x_init * torch.log(x_rec) + (1 - x_init) * torch.log(1 - x_rec)))
                RE_INV = RE_INV / 25.0
            else:
                RE_INV = torch.FloatTensor([0.]).to(device)
        else:
            raise NotImplementedError

        if z_var_q.size()[0] == 0:
            log_q_z_var, log_p_z_var = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        else:
            log_q_z_var = -torch.sum(0.5 * (1 + z_var_q_logvar))
            log_p_z_var = -torch.sum(0.5 * (z_var_q ** 2))

        if tau_q.size()[0] == 0:
            log_q_tau, log_p_tau = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        else:
            log_q_tau = -torch.sum(0.5 * (1 + tau_q_logvar))
            log_p_tau = -torch.sum(0.5 * (tau_q ** 2))
        # if z_c_q.size()[0] == 0:
        #     log_q_z_c, log_p_z_c = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
        # else:
        #     log_q_z_c = -torch.sum(0.5*(1 + z_c_q_logvar/self.model.latent_z_c + \
        #                                    (self.model.latent_z_c -1)*z_c_q**2/self.model.latent_z_c))
        #     log_p_z_c = -torch.sum(0.5*(z_c_q**2 )) + torch.sum(z_c_q)/self.model.latent_z_c

        likelihood = - (RE + RE_INV) / x.shape[0]
        # divergence_c = (log_q_z_c - log_p_z_c)/x.shape[0]
        divergence_var_tau = (log_q_z_var - log_p_z_var) / x.shape[0] + (log_q_tau - log_p_tau) / x.shape[0]

        divergence_var = (log_q_z_var - log_p_z_var) / x.shape[0]
        divergence_tau = (log_q_tau - log_p_tau) / x.shape[0]

        # loss = - likelihood + beta * divergence_var_tau + divergence_c
        # return loss, RE/x.shape[0], divergence_var_tau, divergence_c
        # loss = - likelihood + beta * divergence_var_tau
        loss = - likelihood + beta * (divergence_var + divergence_tau)
        return loss, RE / x.shape[0], divergence_var, divergence_tau

    def load_checkpoint(self, optimizer, filename='checkpoint.pth.tar'):
        # Note: Input self.model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return optimizer, start_epoch

    # def plot_grad_flow(self, named_parameters):
    #     '''Plots the gradients flowing through different layers in the net during training.
    #     Can be used for checking for possible gradient vanishing / exploding problems.

    #     Usage: Plug this function in Trainer class after loss.backwards() as
    #     "plot_grad_flow(self.self.model.named_parameters())" to visualize the gradient flow'''
    #     ave_grads = []
    #     max_grads = []
    #     layers = []
    #     for n, p in named_parameters:
    #         if (p.requires_grad) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grads.append(p.grad.abs().mean().cpu())
    #             max_grads.append(p.grad.abs().max().cpu())
    #     # print(ave_grads)
    #     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    #     plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    #     plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(left=0, right=len(ave_grads))
    #     plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.legend([Line2D([0], [0], color="c", lw=4),
    #                 Line2D([0], [0], color="b", lw=4),
    #                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    # def plot_grad_flow(named_parameters):
    #     ave_grads = []
    #     layers = []
    #     for n, p in named_parameters:
    #         if(p.requires_grad) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grads.append(p.grad.abs().mean().cpu())
    #     plt.plot(ave_grads, alpha=0.3, color="b")
    #     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(xmin=0, xmax=len(ave_grads))
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)

    def train_epoch(self, data, optim, epoch, num_epochs, N, beta):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.train()
        train_loss = 0
        train_reco_loss = 0
        train_div_var_tau = 0
        train_div_c = 0
        c = 0
        for (x, x_init) in data:
            b = x.size(0)
            x = x.view(-1, 1, int(np.sqrt(self.model.in_size)), int(np.sqrt(self.model.in_size))).to(device).float()
            optim.zero_grad()
            loss, reco_loss, divergence_var_tau, divergence_c = self.calc_loss(x, x_init, beta=beta)
            loss.backward()
            # plot_grad_flow(self.model.named_parameters())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.05)
            optim.step()
            c += 1
            train_loss += loss.item()
            train_reco_loss += reco_loss.item()
            train_div_var_tau += divergence_var_tau.item()
            train_div_c += divergence_c.item()
            template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Reco Error={:.5f}, Disent KL={:.5f}, Ent KL={:.5f}'
            line = template.format(epoch + 1, num_epochs, c / N, train_loss / c, train_reco_loss / c,
                                   train_div_var_tau / c,
                                   train_div_c / c)
            print(line, end = '\r', file=sys.stderr)
        print(' ' * 80, end = '\r', file=sys.stderr)
        return train_loss / c, train_reco_loss / c, train_div_var_tau / c, train_div_c / c

    def test_epoch(self, data, beta):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        train_loss = 0
        train_reco_loss = 0
        train_div_var_tau = 0
        train_div_c = 0
        c = 0
        for (x, x_init) in data:
            b = x.size(0)
            x = x.view(-1, 1, int(np.sqrt(self.model.in_size)), int(np.sqrt(self.model.in_size))).to(device).float()
            with torch.no_grad():
                loss, reco_loss, divergence_var_tau, divergence_c = self.calc_loss(x, x_init, beta=beta)
            c += 1
            train_loss += loss.item()
            train_reco_loss += reco_loss.item()
            train_div_var_tau += divergence_var_tau.item()
            train_div_c += divergence_c.item()
        return train_loss / c, train_reco_loss / c, train_div_var_tau / c, train_div_c / c

    def train(self, optim, train_data, test_data, num_epochs=20,
              tr_mode='new', beta=1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        modelname = "model_{}_{}_dEnt_{}_ddisEnt_{}_{}_{}_{}_checkpoint".format(self.model.mode,
                                                                                self.model.invariance_decoder,
                                                                                self.model.latent_z_c,
                                                                                self.model.latent_z_var,
                                                                                self.model.tag,
                                                                                self.model.training_mode,
                                                                                self.model.rec_loss)
        # print(modelname)
        if tr_mode == 'resume' and os.path.exists('MTIE-VAE/models/' + self.modelname):
            print("Loading old model")
            self.model, optim, epoch = self.load_checkpoint(self, optim, 'MTIE-VAE/models/' + self.modelname)
            train_loss_record = np.load('MTIE-VAE/losses/trainloss_' + modelname.replace("_checkpoint", "") + ".npy")
            test_loss_record = np.load('MTIE-VAE/losses/testloss_' + modelname.replace("_checkpoint", "") + ".npy")
            n_trainrecord_old = len(train_loss_record)
            n_testrecord_old = len(test_loss_record)
            train_loss_record = np.append(train_loss_record, np.zeros(num_epochs))
            test_loss_record = np.append(test_loss_record, np.zeros(num_epochs))
        else:
            n_trainrecord_old = 0
            n_testrecord_old = 0
            train_loss_record = np.zeros(num_epochs)
            test_loss_record = np.zeros(num_epochs)
        print('training...')
        N = len(train_data)
        print('Training data  ', N)
        RE_best = 10000
        output = sys.stdout

        # no training latent structure
        mnist_SE2 = torch.from_numpy(np.load(self.prefix + '/mnist_' + 'se2' + '_' + 'test' + '.npy'))[:10000]
        with torch.no_grad():
            x = mnist_SE2.to(device)
            x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = self.model(x)
            # self.create_image(z_var_q, 0, modelname)

        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            train_loss, train_RE, train_div_var_tau, train_div_c = self.train_epoch(train_data,
                                                                               optim, epoch, num_epochs, N, beta)
            line = '\t'.join(
                [str(epoch + 1), 'train', str(train_loss), str(train_RE), str(train_div_var_tau), str(train_div_c)])
            print(line)
            output.flush()
            train_loss_record[n_trainrecord_old + epoch] = train_RE
            test_loss, test_RE, test_div_var_tau, test_div_c = self.test_epoch(test_data, beta)
            line = '\t'.join(
                [str(epoch + 1), 'test', str(test_loss), str(test_RE), str(test_div_var_tau), str(test_div_c)])
            print(line)
            output.flush()
            test_loss_record[n_testrecord_old + epoch] = test_RE

            # no training latent structure
            mnist_SE2 = torch.from_numpy(np.load(self.prefix + '/mnist_' + 'se2' + '_' + 'test' + '.npy'))[:10000]
            with torch.no_grad():
                x = mnist_SE2.to(device)
                x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = self.model(x)
                # self.create_image(z_var_q, epoch, modelname)

            if abs(RE_best) > abs(train_RE):
                RE_best = train_RE
                state = {'epoch': epoch + 1,
                         'state_dict': self.model.state_dict(),
                         'optimizer': optim.state_dict()}
                torch.save(state, 'MTIE-VAE/models/' + modelname)
        print('saving...')
        # plt.savefig('/content/MCE-VAE-hyptune-orig-hyptune-EDIT/gradplot.png', bbox_inches='tight')
        np.save('MTIE-VAE/losses/trainloss_' + modelname.replace("_checkpoint", ""), train_loss_record)
        np.save('MTIE-VAE/losses/testloss_' + modelname.replace("_checkpoint", ""), test_loss_record)

