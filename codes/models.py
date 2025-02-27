import torch
import numpy as np
import yaml
import torch.nn.functional as F
from matplotlib import pyplot as plt
from util import generate_torus_point_cloud
from autoregressive_blocks import PixelCNN, ConditionalPixelCNN
from gan_blocks import LinearSFT, Generator, Discriminator
from diffusion_blocks import ContextUnet
from transformers import VisionTransformer
from vqvae_blocks import VQVAE
import itertools
from tqdm import tqdm


class AutoEncoderBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        pass

    def forward(self):
        pass
    
    def feed_data(self):
        pass
    
    def test(self):
        pass

    def optimize_parameters(self):
        pass

    def compute_loss(self):
        pass
    
    def tb_write_losses(self, tb_write, iter_idx):
        pass

    def eval_reconstructed_image(self, tb_writer, epoch, test_dataLoader):
        n = 10
        digit_size = 28
        image_width = digit_size * n * 2
        image_height = digit_size * n
        image = np.zeros((image_height, image_width))
        
        test_dataLoader_iterator = iter(test_dataLoader)
        
        for i in range(10):
            with torch.no_grad():
                test_data = next(test_dataLoader_iterator)
                self.feed_data(test_data)
                self.test()

            input_img = test_data['image'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            recon_img = self.output_dict['imgs'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            in_re_img = np.concatenate([input_img, recon_img], axis=2)
            for b in range(in_re_img.shape[0]):
                image[i * digit_size: (i+1) * digit_size, b * in_re_img.shape[2]: (b+1) * in_re_img.shape[2]] = in_re_img[b, :, :]
            
        tb_writer.add_image('reconstructed_image'.format(epoch), image, epoch, dataformats="HW")





class AutoEncoder(AutoEncoderBaseModel):
    # def __init__(self, input_dim, hidden_dims, use_sigmoid, device, optimizerConfigs):
    def __init__(self, configs):
        super().__init__()

        modelConfigs = configs['modelConfigs']
        optimizerConfigs = configs['optimizerConfigs']

        input_dim = modelConfigs['input_dim']
        hidden_dims = modelConfigs['hidden_dims']
        use_sigmoid = modelConfigs['use_sigmoid']
        self.device = configs['device']
        

        assert hidden_dims[-1] == 2, "always use 2 as the latent dimension for generating a 2D image grid during evaluation"

        encoder_list = []
        decoder_list = []

        num_hiddenLayers = len(hidden_dims)
        encoder_list += [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        # decoder_list = decoder_list + [torch.nn.Linear(hidden_dims[0], input_dim), torch.nn.ReLU()]
        # encoder_list += [torch.nn.Linear(input_dim, hidden_dims[0])]
        decoder_list = decoder_list + [torch.nn.Linear(hidden_dims[0], input_dim)]

        prev_dim = hidden_dims[0]
        for i in range(1, num_hiddenLayers):
            if i == num_hiddenLayers - 1:
                encoder_list += [torch.nn.Linear(prev_dim, hidden_dims[i])]
            else:
                encoder_list += [torch.nn.Linear(prev_dim, hidden_dims[i]), torch.nn.ReLU()]
            decoder_list = [torch.nn.Linear(hidden_dims[i], prev_dim), torch.nn.ReLU()] + decoder_list
            prev_dim = hidden_dims[i]
        
        if use_sigmoid:
            decoder_list = decoder_list + [torch.nn.Sigmoid()]
        self.encoder = torch.nn.Sequential(*encoder_list)
        self.decoder = torch.nn.Sequential(*decoder_list)



        # init self modules, loss and optimizers
        self.input_x = None
        self.output_dict = {}
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']),
                                     weight_decay=float(optimizerConfigs['weight_decay']))
        self.to(self.device)


    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decode(encoded)
        return {"imgs": decoded}
    
    def feed_data(self, train_data):
        self.input_x = train_data['image']
    
    def test(self):
        self.output_dict = self.forward(self.input_x)

    def optimize_parameters(self):
        self.output_dict = self.forward(self.input_x)

        self.loss = self.compute_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def compute_loss(self):
        mse_err = self.loss_fn(self.output_dict['imgs'], self.input_x)
        return mse_err
    
    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('training loss/MSE loss',
                            self.loss.item(),
                            iter_idx)



    def plot_latent_images(self, tb_writer, epoch):
        n = 10
        digit_size = 28
        grid_x = np.linspace(-2, 2, n)
        grid_y = np.linspace(-2, 2, n)

        image_width = digit_size * n
        image_height = digit_size * n
        image = np.zeros((image_height, image_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    x_decoded = self.decode(z)
                digit = x_decoded.view(digit_size, digit_size).cpu().numpy()
                image[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit
        image = np.clip(image, 0, 1)

        tb_writer.add_image('latent_images_epoch'.format(epoch), image, epoch, dataformats="HW")
    

    def eval_reconstructed_image(self, tb_writer, epoch, test_dataLoader):
        n = 10
        digit_size = 28
        image_width = digit_size * n * 2
        image_height = digit_size * n
        image = np.zeros((image_height, image_width))
        
        test_dataLoader_iterator = iter(test_dataLoader)
        test_data = next(test_dataLoader_iterator)
        for i in range(10):
            with torch.no_grad():
                self.feed_data(test_data)
                self.test()

            input_img = test_data['image'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            recon_img = self.output_dict['imgs'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            in_re_img = np.concatenate([input_img, recon_img], axis=2)
            for b in range(in_re_img.shape[0]):
                image[i * digit_size: (i+1) * digit_size, b * in_re_img.shape[2]: (b+1) * in_re_img.shape[2]] = in_re_img[i, :, :]
            
        tb_writer.add_image('reconstructed_image'.format(epoch), image, epoch, dataformats="HW")

    def validation(self, tb_writer, epoch, test_dataLoader):
        self.plot_latent_images(tb_writer, epoch)
        self.eval_reconstructed_image(tb_writer, epoch, test_dataLoader)


class ConditionalVariationalAutoEncoder(AutoEncoderBaseModel):
    def __init__(self, configs):        
        super().__init__()

        modelConfigs = configs['modelConfigs']

        input_dim = modelConfigs['input_dim']
        hidden_dims = modelConfigs['hidden_dims']
        self.z_size = hidden_dims[-1] // 2
        if modelConfigs['decode_dim'] > 0:
            decode_dim = modelConfigs['decode_dim'] 
        else:
            decode_dim = input_dim

        self.reg_coff = float(modelConfigs['reg_coff'])
        self.conditional = modelConfigs['conditional']
        self.use_sigmoid = modelConfigs['use_sigmoid']
        self.use_torus_latent_code = modelConfigs['use_torus_latent_code']
        self.torus_R, self.torus_r = float(configs['TestingDataSetConfigs']['R']), float(configs['TestingDataSetConfigs']['r'])
        if self.conditional:
            self.n_classes = modelConfigs['n_classes']
            self.conditional_encoder = [torch.nn.Linear(10, hidden_dims[0]), torch.nn.ReLU()]
            self.conditional_encoder = torch.nn.Sequential(*self.conditional_encoder)
            input_dim = input_dim + hidden_dims[0]

        self.device = configs['device']
        

        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()


        encoder_list = []
        decoder_list = []

        num_hiddenLayers = len(hidden_dims)
        encoder_list += [torch.nn.Linear(input_dim, hidden_dims[0]), torch.nn.ReLU()]
        decoder_list = decoder_list + [torch.nn.Linear(hidden_dims[0], decode_dim)]

        prev_dim = hidden_dims[0]
        for i in range(1, num_hiddenLayers):
            if i == num_hiddenLayers - 1:
                encoder_list += [torch.nn.Linear(prev_dim, hidden_dims[i])]
                decoder_list = [torch.nn.Linear(hidden_dims[i]//2, prev_dim), torch.nn.ReLU()] + decoder_list
            else:
                encoder_list += [torch.nn.Linear(prev_dim, hidden_dims[i]), torch.nn.ReLU()]
                decoder_list = [torch.nn.Linear(hidden_dims[i], prev_dim), torch.nn.ReLU()] + decoder_list
            prev_dim = hidden_dims[i]
        if self.use_sigmoid:
            decoder_list = decoder_list + [torch.nn.Sigmoid()]
        self.encoder = torch.nn.Sequential(*encoder_list)
        self.decoder = torch.nn.Sequential(*decoder_list)
        


        self.input_x = None
        self.output_dict = {}
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                lr = 1e-4,
                                weight_decay = 1e-8)

        self.to(self.device)


    def feed_data(self, train_data):
        self.input_x = train_data['image'].to(self.device)
        self.label = train_data['label'].to(self.device)
    
    def test(self):
        self.output_dict = self.forward(self.input_x, self.label)

    def reparameterize(self, mean, logvar):
      var = torch.exp(logvar)
      eps = torch.randn_like(var)
      return mean + torch.sqrt(var) * eps


    def forward(self, x, label):
        if self.conditional:
            label = F.one_hot(label, num_classes=self.n_classes).float()
            conditional_embbeding = self.conditional_encoder(label)
            x = torch.cat([x, conditional_embbeding], dim=-1)

        mean_logvar = self.encoder(x)
        mean, logvar = torch.split(mean_logvar, split_size_or_sections=[self.z_size, self.z_size], dim=-1)
        z = self.reparameterize(mean, logvar)
        x_probs = self.decode(z)
        

        return {"imgs": x_probs,
                "z": z,
                "mean": mean,
                "logvar": logvar}

    def compute_loss(self):
        mse_err = self.loss_fn(self.output_dict['imgs'], self.input_x)
        logpx_z = -1.0 * torch.sum(mse_err)


        mean, logvar = self.output_dict['mean'], self.output_dict['logvar']
        var = torch.exp(logvar)
        kl_divergence = -0.5 * torch.sum(torch.pow(mean, 2)
                            + var - 1.0 - logvar,
                            dim=[1])

        self.loss = -1.0 * torch.mean(logpx_z + self.reg_coff * kl_divergence)
    

    def optimize_parameters(self):
        self.output_dict = self.forward(self.input_x, self.label)

        self.compute_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def decode(self, z):
        z = self.decoder(z)
        if self.use_torus_latent_code:
            R, r = self.torus_R, self.torus_r
            x_soph = (R + r * torch.cos(z[:,0])) * torch.cos(z[:,1])
            y_soph = (R + r * torch.cos(z[:,0])) * torch.sin(z[:,1])
            z_soph = r * torch.sin(z[:, 0])
            x_probs = torch.cat([x_soph[:, None], y_soph[:, None], z_soph[:, None]], dim=-1)
        return x_probs

    
    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('training loss/MSE loss',
                            self.loss.item(),
                            iter_idx)
    
    def validation(self, tb_writer, epoch, test_dataLoader):
        self.eval_reconstructed_image(tb_writer, epoch, test_dataLoader)
    


class PositionalEncoding3D(torch.nn.Module):
    def __init__(self, num_frequencies=10):
        """
        Initializes the positional encoding for 3D coordinates.

        Args:
            num_frequencies (int): The number of different frequencies to use for encoding.
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = 2 ** torch.arange(num_frequencies, dtype=torch.float32)

    def forward(self, points):
        """
        Applies positional encoding to the 3D points.

        Args:
            points (torch.Tensor): N x 3 tensor of 3D coordinates.

        Returns:
            torch.Tensor: N x (6*num_frequencies) tensor of encoded coordinates.
        """
        encoded_points = []
        for i in range(points.shape[1]):  # For each dimension (x, y, z)
            for freq in self.frequencies:
                encoded_points.append(torch.sin(freq * points[:, i:i+1]))
                encoded_points.append(torch.cos(freq * points[:, i:i+1]))
        return torch.cat(encoded_points, dim=-1)



class PointVAE(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.device = configs['device']
        self.reg_coff = configs['modelConfigs']['reg_coff']
        self.pos_enc = PositionalEncoding3D()
        #############
        ### Problem 4(c): Create your own VAE
        self.vae = ConditionalVariationalAutoEncoder(configs)
        #############

        self.input_x = None
        self.output_dict = {}
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                lr=float(configs['optimizerConfigs']['learning_rate']),
                                weight_decay=float(configs['optimizerConfigs']['weight_decay']))
        self.to(self.device)


    def feed_data(self, train_data):
        self.input_x = train_data['image'].to(self.device)
        self.label = train_data['label'].to(self.device)

    def forward(self, x, label):
        emb = self.pos_enc(x)
        return self.vae(emb, label)
    
    def compute_loss(self):
        mse_err = self.loss_fn(self.output_dict['imgs'], self.input_x)
        logpx_z = -1.0 * torch.sum(mse_err)


        mean, logvar = self.output_dict['mean'], self.output_dict['logvar']
        var = torch.exp(logvar)
        kl_divergence = -0.5 * torch.sum(torch.pow(mean, 2)
                            + var - 1.0 - logvar,
                            dim=[1])

        self.loss = -1.0 * torch.mean(logpx_z + self.reg_coff * kl_divergence)
        # print(self.loss.item())
    

    def optimize_parameters(self):
        self.output_dict = self.forward(self.input_x, self.label)

        self.compute_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('training loss/MSE loss',
                            self.loss.item(),
                            iter_idx)
        

    def plot_torus_point_cloud(self, x, y, z, ax, color='b', name='Training Data'):
        """
        Plots the 3D point cloud of a torus.
        """
        ax.scatter(x, y, z, c=color, marker='o', s=5)

        # Set equal scaling for all axes
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(name)
        return ax

    def validation(self, tb_writer, epoch, test_dataLoader):
        if epoch % self.configs['TestingDataSetConfigs']['plot_interval'] == 0:

            with torch.no_grad():
                R = self.configs['TestingDataSetConfigs']['R']
                r = self.configs['TestingDataSetConfigs']['r']
                num_points = self.configs['TestingDataSetConfigs']['num_points']
                points = generate_torus_point_cloud(R, r, num_points)
                points = torch.tensor(points, dtype=torch.float32)
                self.output_dict = self.forward(points.to(self.device), 0)
                decoded_pc = self.output_dict['imgs'].cpu().numpy()

                # fig = plt.figure(figsize=(10, 7))
                # ax = fig.add_subplot(111, projection='3d')
                # self.plot_torus_point_cloud(decoded_pc[:, 0], decoded_pc[:, 1], decoded_pc[:, 2], ax, name="Decoded Data")
                # fig.canvas.draw()

                # # Now we can save it to a numpy array.
                # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # plt.close(fig)
                # tb_writer.add_image('cloud_point', data, epoch, dataformats="HWC")


                z0 = self.forward(torch.tensor([[0.0, 1.0, 0.]]).to(self.device), None)['z']
                z1 = self.forward(torch.tensor([[0.0, -1.0, 0.]]).to(self.device), None)['z']

                num_steps = 100

                weights = torch.linspace(0, 1, num_steps).view(-1, 1).to(self.device)
                latent_vecs = weights * z0 + (1 - weights) * z1

                with torch.no_grad():
                    outputs = self.vae.decode(torch.tensor(latent_vecs).to(self.device))

                lin_traj = outputs.cpu().numpy()

                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                self.plot_torus_point_cloud(lin_traj[:, 0], lin_traj[:, 1], lin_traj[:, 2], ax, color='r', name="Decoded Data")
                self.plot_torus_point_cloud(decoded_pc[:, 0], decoded_pc[:, 1], decoded_pc[:, 2], ax, name="Decoded Data")
                fig.canvas.draw()

                # Now we can save it to a numpy array.
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                tb_writer.add_image('trajectory', data, epoch, dataformats="HWC")



class PixelCNN_model(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        # num_input_c=1, num_inner_c=64, num_output_c=1, num_masked_convs=4

        modelConfigs = configs['modelConfigs']
        optimizerConfigs = configs['optimizerConfigs']


        num_input_c = modelConfigs['input_chn']
        num_inner_c = modelConfigs['inner_chn']
        num_output_c = modelConfigs['output_chn']
        self.validation_interval = configs['TestingDataSetConfigs']['validation_interval']
        self.conditional = modelConfigs['conditional']
        self.n_classes = configs['TestingDataSetConfigs']['n_classes']
        self.device = configs['device']

        if self.conditional:
            self.pixelCNN = ConditionalPixelCNN(num_input_c=num_input_c, num_inner_c=num_inner_c, num_output_c=num_output_c)
        else:
            self.pixelCNN = PixelCNN(num_input_c=num_input_c, num_inner_c=num_inner_c, num_output_c=num_output_c)


        # init self modules, loss and optimizers
        self.input_x = None
        self.output_dict = {}
        self.loss_fn = torch.nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(params=self.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))
        self.to(self.device)



    def forward(self):

        if self.conditional:
            self.output_dict['imgs'] = self.pixelCNN(self.input_x, self.label)
        else:
            self.output_dict['imgs'] = self.pixelCNN(self.input_x)
        return self.output_dict['imgs']
        
    
    def feed_data(self, train_data):
        self.input_x = train_data['image'].to(self.device)
        self.label = train_data['label'].to(self.device)
        self.gt = self.input_x.clone()
    
    def test(self):
        self.forward()

    def optimize_parameters(self):
        self.train()
        self.forward()
        self.compute_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        

    def compute_loss(self):
        output_img = self.output_dict['imgs']
        self.loss = self.loss_fn(output_img, self.gt)
        
    
    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('training loss/BCE loss',
                            self.loss.item(),
                            iter_idx)
    

    def reconstruction_test(self, tb_writer, epoch, test_dataLoader):
        H, W = 28, 28
        self.eval()
        with torch.no_grad():
            for iter, test_data in enumerate(test_dataLoader):
                # images = images.to(device)
                self.feed_data(test_data)
                self.test()
                pred = self.output_dict['imgs']

                for i in range(H):
                    for j in range(W):
                        pred[:, :, i, j] = torch.bernoulli(pred[:, :, i, j], out=pred[:, :, i, j])
                break

        samples = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
        fig, axes = plt.subplots(8, 8, figsize=(15, 15))

        for i in range(64):
            sample = samples[i]
            row, col = divmod(i, 8)
            axes[row, col].imshow(sample, cmap='gray')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        tb_writer.add_image('Reconstruction', data, epoch, dataformats="HWC")
    

    def generation_test(self, tb_writer, epoch):
        H, W = 28, 28
        if self.conditional:
            samples = torch.zeros(size=(60, 1, H, W)).to(self.device)
            label = np.sort(np.array([np.arange(self.n_classes)] * 6).flatten())
            label = F.one_hot(torch.tensor(label), num_classes=self.n_classes).to(self.device).float()
        else:
            samples = torch.zeros(size=(64, 1, H, W)).to(self.device)
            label = torch.ones(size=(64, 1)).to(self.device)
        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    if j > 0 and i > 0:
                        test_data = {}
                        test_data['image'] = samples
                        test_data['label'] = label
                        self.feed_data(test_data)
                        self.test()
                        out = self.output_dict['imgs']
                        samples[:, :, i, j] = torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

        samples = samples.cpu().numpy().transpose(0, 2, 3, 1)

        if self.conditional:
            fig, axes = plt.subplots(10, 6, figsize=(15, 30))
            for i in range(60):
                sample = samples[i]
                row, col = divmod(i, 6)
                axes[row, col].imshow(sample, cmap='gray')
                axes[row, col].axis('off')
        else:
            fig, axes = plt.subplots(8, 8, figsize=(15, 15))
            for i in range(64):
                sample = samples[i]
                row, col = divmod(i, 8)
                axes[row, col].imshow(sample, cmap='gray')
                axes[row, col].axis('off')

        plt.tight_layout()
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        tb_writer.add_image('Generation', data, epoch, dataformats="HWC")

    def validation(self, tb_writer, epoch, test_dataLoader):

        # if (epoch + 1) % self.validation_interval == 0 or epoch == 0 :
    
        self.reconstruction_test(tb_writer, epoch, test_dataLoader)
        self.generation_test(tb_writer, epoch)

        


        
class GAN_model(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        modelConfigs = configs['modelConfigs']
        optimizerConfigs = configs['optimizerConfigs']
        self.device = configs['device']

        self.d_init_chn_num = modelConfigs['d_init_chn_num']
        self.dim_z = modelConfigs['dim_z']
        self.g_chn_num = modelConfigs['g_chn_num']
        self.d_chn_num = modelConfigs['d_chn_num']
        self.with_condition = modelConfigs['with_condition']
        self.num_test_samples = configs['TestingDataSetConfigs']['num_test_samples']

        self.generator = Generator(dim_z=self.dim_z, channels=self.g_chn_num, with_condition=self.with_condition) # dim_z=100, channels = [128, 256, 512], with_condition=False
        self.discriminator = Discriminator(init_chn_num=self.d_init_chn_num, channels=self.d_chn_num, with_condition=self.with_condition) # channels = [512, 256, 128], with_condition=False

        

        # init self modules, loss and optimizers
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer_g = torch.optim.Adam(params=self.generator.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))
        
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))

        self.to(self.device)


    def feed_data(self, train_data):
        self.gt_image = train_data['image'].to(self.device)
        self.labels = train_data['label'].to(self.device)
        batch_size = self.gt_image.shape[0]
        self.input_noise = torch.randn([batch_size, self.dim_z]).to(self.device)

    def optimize_parameters(self):
        fake_image = self.generator(self.input_noise, self.labels)
        fake_image_logits = self.discriminator(fake_image, self.labels)
        fake_image_target = torch.ones_like(fake_image_logits)

        g_loss = self.loss_fn(fake_image_logits, fake_image_target)

        fake_image = fake_image.detach()
        fake_image_logits = self.discriminator(fake_image, self.labels)
        fake_image_target = torch.zeros_like(fake_image_logits)
        d_loss_fake = self.loss_fn(fake_image_logits, fake_image_target)

        real_image_logits = self.discriminator(self.gt_image, self.labels)
        real_image_target = torch.ones_like(real_image_logits)
        d_loss_real = self.loss_fn(real_image_logits, real_image_target)
        d_loss = d_loss_fake + d_loss_real


        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()

        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()

        self.fake_image_logits = torch.mean(fake_image_logits.detach()).item()
        self.real_image_logits = torch.mean(real_image_logits.detach()).item()
        self.g_loss = g_loss.item()
        self.d_loss = d_loss.item()
        self.d_loss_fake = d_loss_fake.item()
        self.d_loss_real = d_loss_real.item()


    
    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('Training loss/g_loss',
                            self.g_loss,
                            iter_idx)
        tb_write.add_scalar('Training loss/d_loss',
                            self.d_loss,
                            iter_idx)
        tb_write.add_scalar('Training loss/d_loss_fake',
                            self.d_loss_fake,
                            iter_idx)
        tb_write.add_scalar('Training loss/d_loss_real',
                            self.d_loss_real,
                            iter_idx)
        
        tb_write.add_scalar('G_D logits/fake_image_logits',
                            self.fake_image_logits,
                            iter_idx)
        tb_write.add_scalar('G_D logits/real_image_logits',
                            self.real_image_logits,
                            iter_idx)

    def validation(self, writer, epoch, test_dataLoader):
        self.monitor_images(writer, epoch)


    def monitor_images(self, tb_writer, epoch):
        
        test_noise = torch.randn(self.num_test_samples, self.dim_z).to(self.device)
        size_figure_grid = int(np.sqrt(self.num_test_samples))
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)

        if self.with_condition:
            cls_labels = torch.randint(0, 10, (self.num_test_samples,)).to(self.device)
            cls_labels_oh = F.one_hot(cls_labels, num_classes=10).float()
            test_images = self.generator(test_noise, label = cls_labels_oh)
        else:
            test_images = self.generator(test_noise)

        for k in range(self.num_test_samples):
            i = k//4
            j = k%4
            ax[i,j].cla()
            ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
            if self.with_condition:
                ax[i, j].set_title(f'Class: {cls_labels[k].item()}', fontsize=10)
        
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        tb_writer.add_image('Generation', data, epoch, dataformats="HWC")




class DDPM_model(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        
        modelConfigs = configs['modelConfigs']
        schedules_configs = configs['schedules_configs']
        optimizerConfigs = configs['optimizerConfigs']
        TrainingDataSetConfigs = configs['TrainingDataSetConfigs']
        TestingDataSetConfigs = configs['TestingDataSetConfigs']
        self.device = configs['device']

        
        in_channels = modelConfigs['in_channels']
        n_feat = modelConfigs['n_feat']
        self.n_classes = modelConfigs['n_classes']
        self.context_unet_model = ContextUnet(in_channels=in_channels, n_feat=n_feat, n_classes=self.n_classes)

        self.sample_type = modelConfigs['sample_type']
        self.ddim_step = modelConfigs['ddim_step']
        self.drop_prob = modelConfigs['drop_prob']
        self.test_size = TestingDataSetConfigs['test_size']
        self.guided_weights_list = TestingDataSetConfigs['guided_weights_list']


        # schedules init
        self.n_T = schedules_configs['n_T']
        self.beta1 = float(schedules_configs['beta1'])
        self.beta2 = float(schedules_configs['beta2'])
        self.schedules_init(self.beta1, self.beta2, self.n_T)
        

        # init self modules, loss and optimizers
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.context_unet_model.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))
        
        self.to(self.device)

    def schedules_init(self, beta1, beta2, T):
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        alpha_t = 0.0
        oneover_sqrta = 0.0
        sqrt_beta_t = 0.0
        alphabar_t= 0.0
        sqrtab = 0.0
        sqrtmab = 0.0
        mab_over_sqrtmab_inv  = 0.0

        with torch.no_grad():
            t_list = torch.arange(0, T + 1)
            beta_t = beta1 + t_list * (beta2 - beta1) / T
            alpha_t = 1 - beta_t
            oneover_sqrta = 1 / torch.sqrt(alpha_t)
            sqrt_beta_t = torch.sqrt(beta_t)
            alphabar_t = torch.cumprod(alpha_t, dim=0)
            sqrtab = torch.sqrt(alphabar_t)
            sqrtmab = torch.sqrt(1 - alphabar_t)
            mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        res_dict = {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

        for k, v in res_dict.items():
            self.register_buffer(k, v)

    
    def feed_data(self, input_data):
        self.input_x = input_data['image'].to(self.device)
        self.label = input_data['label'].to(self.device) # one hot now

    def optimize_parameters(self):
        self.train()

        bs, c, h, w = self.input_x.shape
        current_t = torch.randint(low=1, high=self.n_T, size=[bs, 1, 1, 1]).to(self.device)
        context_mask = torch.tensor([self.drop_prob] * bs)
        # context_mask = context_mask[:, None]
        context_mask = torch.bernoulli(context_mask).to(self.device)

        sqrtab = self.sqrtab[current_t].to(self.device)
        sqrtmab = self.sqrtmab[current_t].to(self.device)
        standard_noise = torch.randn_like(self.input_x).to(self.device)
        
        xt = sqrtab * self.input_x + sqrtmab * standard_noise
        current_t = current_t / self.n_T
        est_noise = self.context_unet_model(xt, self.label, current_t, context_mask)

        self.loss = self.loss_fn(est_noise, standard_noise)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    

    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('Training loss/mse_loss',
                            self.loss.item(),
                            iter_idx)


    def ddpm_sample(self, guided_w):

        x_t = torch.randn(self.test_size).to(self.device)
        
        c = torch.arange(0, 10)
        context = torch.zeros_like(c)

        c = c.repeat(2)
        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context = context.repeat(2)
        
        bs, _, h, w = self.test_size

        c = c.repeat(2, 1)
        c = c.to(self.device)
        context = context.repeat(2)
        context[bs:] = 1.0
        context = context.to(self.device)

        

        for i in range(self.n_T, -1, -1):
            
            oneover_sqrta = self.oneover_sqrta[i].to(self.device)
            mab_over_sqrtmab = self.mab_over_sqrtmab[i].to(self.device)
            beta_t = self.sqrt_beta_t[i].to(self.device)

            x_t = x_t.repeat(2, 1, 1, 1).to(self.device)
            t = torch.tensor(i / self.n_T).to(self.device)
            est_noise = self.context_unet_model(x_t, c, t, context)

            context_noise = est_noise[:bs, :, :, :]
            uncontext_noise = est_noise[bs:, :, :, :]

            if i != 0:
                z = torch.randn(self.test_size).to(self.device)
            else:
                z = 0

            interp_noise = (1 + guided_w) * context_noise - guided_w * uncontext_noise
            x_t = x_t[:bs, :, :, :]

            x_t = oneover_sqrta * (x_t - mab_over_sqrtmab * interp_noise) + beta_t * z

        return x_t

    
    def sample_ddim(self, guided_w):
        x_t = torch.randn(self.test_size).to(self.device)
        
        c = torch.arange(0, 10)
        context = torch.zeros_like(c)

        c = c.repeat(2)
        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context = context.repeat(2)
        
        bs, _, h, w = self.test_size

        c = c.repeat(2, 1)
        c = c.to(self.device)
        context = context.repeat(2)
        context[bs:] = 1.0
        context = context.to(self.device)

        sample_indexes = list(range(self.n_T, -1, self.ddim_step))

        for i in range(len(sample_indexes)-1):

            cur_t = sample_indexes[i]
            pre_t = sample_indexes[i+1]
            pre_t = max(pre_t, 1)

            sqrt_alphabar_pre = self.alphabar_t[pre_t].to(self.device)
            sqrt_alphabar_cur = self.alphabar_t[cur_t].to(self.device)
            beta_pre = torch.sqrt(1 - sqrt_alphabar_pre).to(self.device)
            beta_cur = torch.sqrt(1 - sqrt_alphabar_cur).to(self.device)
            

        #     res_dict = {
        #     "alpha_t": alpha_t,  # \alpha_t
        #     "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        #     "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        #     "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        #     "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        #     "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        #     "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        # }
            
            

            x_t = x_t.repeat(2, 1, 1, 1).to(self.device)
            t = torch.tensor(cur_t / self.n_T).to(self.device)
            est_noise = self.context_unet_model(x_t, c, t, context)

            context_noise = est_noise[:bs, :, :, :]
            uncontext_noise = est_noise[bs:, :, :, :]

            interp_noise = (1 + guided_w) * context_noise - guided_w * uncontext_noise
            x_t = x_t[:bs, :, :, :]

            x_t = (sqrt_alphabar_pre / sqrt_alphabar_cur) * (x_t - beta_cur * interp_noise) + beta_pre * interp_noise

        return x_t


    def plot_sample_res(self, tb_writer, epoch):
        # ws_test = [0.0, 0.5, 2.0]
        ws_test = self.guided_weights_list
        n_sample = self.test_size[0]
        self.eval()
        with torch.no_grad():
            fig, ax_all = plt.subplots(3, 1, figsize=(15, 7.2))
            for w_i, w in enumerate(ws_test):
                if self.sample_type == 'ddpm':
                    x_gen = self.ddpm_sample(w)
                elif self.sample_type == 'ddim':
                    x_gen = self.sample_ddim(w)
                else:
                    raise NotImplemented("Sample type {} is not implemneted!".format(self.sample_type))

                # fig, ax = plt.subplots(2, 10, figsize=(15, 2.4))
                # for i, j in itertools.product(range(2), range(10)):
                #     ax[i,j].get_xaxis().set_visible(False)
                #     ax[i,j].get_yaxis().set_visible(False)
                x_gen_flatten = torch.zeros([2*28, 28*10])

                for k in range(n_sample):
                    i = k//10
                    j = k%10
                    # ax[i,j].cla()
                    # ax[i,j].imshow(x_gen[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')  
                    x_gen_flatten[i*28:(i+1)*28, j*28:(j+1)*28] = x_gen[k, :]
                ax_all[w_i].cla()
                ax_all[w_i].get_xaxis().set_visible(False)
                ax_all[w_i].imshow(x_gen_flatten.data.cpu().numpy(), cmap='Greys')
                ax_all[w_i].set_title('guided wight = {}'.format(w), fontsize=16)
        
        # Now we can save it to a numpy array.
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        tb_writer.add_image('Sample Res', data, epoch, dataformats="HWC")
        

    def validation(self, writer, epoch, test_dataLoader):
        self.plot_sample_res(writer, epoch)



class VisionTransformer_model(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        self.configs = configs
        modelConfigs = configs['modelConfigs']
        optimizerConfigs = configs['optimizerConfigs']
        TrainingDataSetConfigs = configs['TrainingDataSetConfigs']
        TestingDataSetConfigs = configs['TestingDataSetConfigs']
        self.device = configs['device']

        # image_channel, image_size, patch_size, num_transformer, num_head, embed_size, num_class
        image_channel = modelConfigs['image_channel']
        image_size = modelConfigs['image_size']
        patch_size = modelConfigs['patch_size']
        num_transformer = modelConfigs['num_transformer']
        num_head = modelConfigs['num_head']
        embed_size = modelConfigs['embed_size']
        num_class = TrainingDataSetConfigs['n_classes']

        self.vit = VisionTransformer(image_channel=image_channel, image_size=image_size, patch_size=patch_size, num_transformer=num_transformer, num_head=num_head, embed_size=embed_size, num_class=num_class)


        # init self modules, loss and optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.vit.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))
        
        self.save_model_interval = modelConfigs['save_model_interval']
        
        self.to(self.device)

    def feed_data(self, input_data):
        self.input_x = input_data['image'].to(self.device)
        self.label = input_data['label'].to(self.device) # one hot now

    def optimize_parameters(self):
        self.train()

        pred = self.vit(self.input_x)

        self.loss = self.loss_fn(pred, self.label)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def test(self):
        pred = self.vit(self.input_x)
        probs = torch.nn.functional.softmax(pred, dim=-1)
        pred_label = torch.argmax(probs, dim=-1)
        return pred_label

    def tb_write_losses(self, tb_write, iter_idx):
        tb_write.add_scalar('Training loss/CrossEntorpy_loss',
                            self.loss.item(),
                            iter_idx)


    def cal_acurracy(self, test_dataLoader):
        self.eval()
        correct_num = 0
        total_num = 0

        

        with torch.no_grad():
            val_batch_progress = tqdm(test_dataLoader, desc='Val_batch', leave=False)
            test_iter_rand = np.random.randint(0, len(test_dataLoader))
            for iter_idx, test_data in enumerate(val_batch_progress):
                self.feed_data(test_data)
                pred_label = self.test()
                gt_label = test_data['label']
                gt_label = torch.argmax(gt_label, dim=-1)

                pred_label = pred_label.cpu().numpy()
                gt_label = gt_label.numpy()
                num_cor = np.sum((pred_label - gt_label) == 0)
                correct_num += num_cor
                total_num += pred_label.shape[0]

                if iter_idx == test_iter_rand:
                    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
                    for k in range(16):
                        i = k//4
                        j = k%4
                        ax[i,j].cla()
                        ax[i,j].get_yaxis().set_visible(False)
                        ax[i,j].get_xaxis().set_visible(False)
                        ax[i,j].imshow(self.input_x[k,:,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
                        ax[i, j].set_title(f'Pred Class: {pred_label[k]}', fontsize=10)


        return correct_num / total_num, correct_num, total_num, fig




    def validation(self, tb_writer, epoch, test_dataLoader):
        acc, num_cor, total_num, fig = self.cal_acurracy(test_dataLoader=test_dataLoader)

        tb_writer.add_scalar('Validation/Accuracy', acc, epoch)
        # Now we can save it to a numpy array.
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        tb_writer.add_image('Validation Samples', data, epoch, dataformats="HWC")

        if (epoch + 1) % self.save_model_interval == 0:
            save_path = './expriments_save/{}/model_epoch_{}.pth'.format(self.configs['Configuration_name'], epoch)
            torch.save(self.state_dict(), save_path)
    
    def inference(self, tb_writer, test_dataLoader):
        self.eval()
        correct_num = 0
        total_num = 0
        with torch.no_grad():
            val_batch_progress = tqdm(test_dataLoader, desc='Val_batch', leave=False)
            test_iter_rand = np.random.randint(0, len(test_dataLoader), 10)
            for iter_idx, test_data in enumerate(val_batch_progress):
                self.feed_data(test_data)
                pred_label = self.test()
                gt_label = test_data['label']
                gt_label = torch.argmax(gt_label, dim=-1)

                pred_label = pred_label.cpu().numpy()
                gt_label = gt_label.numpy()
                num_cor = np.sum((pred_label - gt_label) == 0)
                correct_num += num_cor
                total_num += pred_label.shape[0]

                if iter_idx in test_iter_rand:
                    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
                    for k in range(16):
                        i = k//4
                        j = k%4
                        ax[i,j].cla()
                        ax[i,j].get_yaxis().set_visible(False)
                        ax[i,j].get_xaxis().set_visible(False)
                        ax[i,j].imshow(self.input_x[k,:,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
                        ax[i, j].set_title(f'Pred Class: {pred_label[k]}', fontsize=10)

                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    tb_writer.add_image('Validation Samples', data, iter_idx, dataformats="HWC")


        acc = correct_num / total_num
        tb_writer.add_scalar('Test/Accuracy', acc, -1)



class VQVAE_model(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        self.configs = configs
        modelConfigs = configs['modelConfigs']
        optimizerConfigs = configs['optimizerConfigs']
        self.TrainingDataSetConfigs = configs['TrainingDataSetConfigs']
        self.TestingDataSetConfigs = configs['TestingDataSetConfigs']
        self.device = configs['device']

        # image_channel, image_size, patch_size, num_transformer, num_head, embed_size, num_class
        image_channel = modelConfigs['input_channel']
        inner_channel = modelConfigs['inner_channel']
        output_channel = modelConfigs['output_channel']
        num_embedding = modelConfigs['num_embedding']
        num_class = self.TrainingDataSetConfigs['n_classes']
        conditional = self.TrainingDataSetConfigs['conditional']

        self.w_recon, self.w_embedding, self.w_commitment = optimizerConfigs['w_recon'], optimizerConfigs['w_embedding'], optimizerConfigs['w_commitment']

        self.vqvae_module = VQVAE(image_channel, inner_channel, output_channel, num_embedding, num_class, conditional)
        # self.pixelcnn_module = PixelCNN()
        if self.conditional:
            self.pixelcnn_module = ConditionalPixelCNN(num_input_c=inner_channel, num_inner_c=inner_channel, num_output_c=inner_channel)
        else:
            self.pixelcnn_module = PixelCNN(num_input_c=inner_channel, num_inner_c=inner_channel, num_output_c=inner_channel)

        # init self modules and optimizers
        self.optimizer_vqvae = torch.optim.Adam(params=self.vqvae_module.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))

        self.optimizer_pixelcnn = torch.optim.Adam(params=self.pixelcnn_module.parameters(), 
                                     lr=float(optimizerConfigs['learning_rate']))
        
        self.mse_loss = torch.nn.MSELoss()

        self.save_model_interval = modelConfigs['save_model_interval']
        self.to(self.device)

    def feed_data(self, input_data):
        self.input_x = input_data['image'].to(self.device)
        self.label = input_data['label'].to(self.device) # one hot now
    

    # vqvae
    def optimize_parameters_vqvae(self):
        self.train()

        # ze = self.vqvae_module.encode(self.input_x)
        # zq = self.vqvae_module.embbed(ze)
        # recon_img = self.vqvae_module.decode(zq)
        ze, zq, recon_img = self.vqvae_module(self.input_x)

        self.loss_recon = self.mse_loss(recon_img, self.input_x)
        self.loss_embedding = self.mse_loss(ze.detach(), zq)
        self.loss_commitment = self.mse_loss(ze, zq.detach())

        self.loss_vqvae = self.w_recon * self.loss_recon + self.w_embedding * self.loss_embedding + self.w_commitment * self.loss_commitment

        self.optimizer_vqvae.zero_grad()
        self.loss_vqvae.backward()
        self.optimizer_vqvae.step()
    
    def tb_write_losses_vqvae(self, tb_write, iter_idx):
        tb_write.add_scalar('Training loss/totoal_vqvae',
                            self.loss.item(),
                            iter_idx)
        
        tb_write.add_scalar('Training loss/loss_Reconstruction',
                            self.loss_recon.item(),
                            iter_idx)
        
        tb_write.add_scalar('Training loss/loss_Embedding',
                            self.loss_embbeding.item(),
                            iter_idx)
        
        tb_write.add_scalar('Training loss/loss_Commitment',
                            self.loss_commitment.item(),
                            iter_idx)

    def validation_vqvae(self, tb_writer, epoch, test_dataLoader):
        self.eval()
        n = 10
        digit_size = 28
        image_width = digit_size * n * 2
        image_height = digit_size * n
        image = np.zeros((image_height, image_width))
        
        test_dataLoader_iterator = iter(test_dataLoader)
        test_data = next(test_dataLoader_iterator)
        for i in range(10):
            with torch.no_grad():
                self.feed_data(test_data)
                self.test()

            input_img = test_data['image'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            recon_img = self.output_dict['imgs'].view(test_data['image'].shape[0], digit_size, digit_size).cpu().numpy()
            in_re_img = np.concatenate([input_img, recon_img], axis=2)
            for b in range(in_re_img.shape[0]):
                image[i * digit_size: (i+1) * digit_size, b * in_re_img.shape[2]: (b+1) * in_re_img.shape[2]] = in_re_img[i, :, :]
            
        tb_writer.add_image('reconstructed_image'.format(epoch), image, epoch, dataformats="HW")

    # pixelcnn
    def optimize_parameters_pixelcnn(self):
        self.train()

        with torch.no_grad():
            ze = self.vqvae_module.encoder(self.input_x)
            zq = self.vqvae_module.embbed(ze)

        pred_zq = self.pixelcnn_module(zq)
        self.loss_pixelcnn_recont = self.mse_loss(pred_zq, zq)

        self.optimizer_pixelcnn.zero_grad()
        self.loss_pixelcnn_recont.backward()
        self.optimizer_pixelcnn.step()

    def tb_write_losses_pixelcnn(self, tb_write, iter_idx):
        tb_write.add_scalar('Training loss/totoal_pixelcnn',
                            self.loss_pixelcnn_recont.item(),
                            iter_idx)
    
    def validation_pixelcnn(self, tb_writer, epoch):

        target_size = self.TestingDataSetConfigs['test_sample_size']
        samples_shape = [1] + target_size # [1, 28, 28]
        samples = torch.zeros(samples_shape)
        H, W = samples_shape[1:]

        with torch.no_grad():
            # samples_shape = [10*10] + target_size # [100, 28, 28]
            # label = torch.arange(0, 10).repeat(10, 1).transpose(1, 0)
            # samples = torch.zeros(samples_shape)
            zq = self.vqvae_module.encoder(samples)
            H, W = zq.shape[1:]

            latent_samples_shape = [100] + samples_shape 
            latent_samples = torch.zeros(latent_samples_shape) # [100, 28, 28]
            label = torch.arange(0, 10).repeat(10, 1).transpose(1, 0).flatten() # [100, 1]

            for i in range(H):
                for j in range(W):
                    if j > 0 and i > 0:
                        # test_data = {}
                        # test_data['image'] = latent_samples
                        # test_data['label'] = label
                        pred_code = self.pixelcnn_module(latent_samples)
                        prob_dist = F.softmax(pred_code, dim=1)
                        pixel = torch.multinomial(prob_dist, 1)
                        latent_samples[:, :, i, j] = pixel[:, 0]

        
            samples = self.vqvae_module.decoder(latent_samples)
        
        samples = samples.cpu().numpy().transpose(0, 2, 3, 1)
        samples = samples.squeeze()
        samples_flatten_img = np.zeros([10*28, 10*28])
        for i in range(10):
            c_chunk = samples[i*10:(i+1)*10, :, :, :] # [10, 28, 28]
            c_chunk = np.concatenate(c_chunk, axis=2)
            samples_flatten_img[i*28:(i+1)*28, :] = c_chunk

        
        tb_writer.add_image('generated_image'.format(epoch), samples_flatten_img, epoch, dataformats="HW")

if __name__ == '__main__':

    
    ### Test
    device = 'cuda'
    hidden_dims = [128, 64, 36, 18, 2]
    input_dim = 256
    test_tensor = torch.randn([1, input_dim]).to(device)
    
    train_data = {'image': test_tensor}

    with open('train_config.yaml') as f:
        configs = yaml.safe_load(f)
    optimizerConfigs = configs['optimizerConfigs']

    ae_test = AutoEncoder(input_dim, hidden_dims, device, optimizerConfigs).to(device)
    print(ae_test)

    with torch.no_grad():
        ae_test.feed_data(train_data)
        ae_test.optimize_parameters()