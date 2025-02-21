import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import cv2



def create_model(configs):
    model_class_name = configs['modelConfigs']['model_class_name']

    if model_class_name == 'AutoEncoder':
        from models import AutoEncoder
        model = AutoEncoder(configs)

    elif model_class_name == 'ConditionalVariationalAutoEncoder':
        from models import ConditionalVariationalAutoEncoder
        model = ConditionalVariationalAutoEncoder(configs)

    elif model_class_name == 'PointVAE':
        from models import PointVAE
        model = PointVAE(configs)
    
    elif model_class_name == 'PixelCNN':
        from models import PixelCNN_model
        model = PixelCNN_model(configs)

    elif model_class_name == 'GAN':
        from models import GAN_model
        model = GAN_model(configs)

    elif model_class_name == 'Diffusion':
        from models import DDPM_model
        model = DDPM_model(configs)
    
    elif model_class_name == 'Transformer':
        from models import VisionTransformer_model
        model = VisionTransformer_model(configs)

    else:
        raise(NotImplemented('model {} is not registered!'.format(model_class_name)))
    print(model)
    return model



def create_dataLoader(dataSetConfigs):
    dataset_name = dataSetConfigs['dataset_name']
    dataset_path = dataSetConfigs['dataset_path']
    

    if dataset_name == 'MNIST':

        # input transforms
        tensor_transform = [transforms.ToTensor()]

        if dataSetConfigs['binarize_image']:
            tensor_transform.append(transforms.Lambda(binarize_image))
        
        if dataSetConfigs['Normalize']:
            tensor_transform.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))

        tensor_transform = transforms.Compose(tensor_transform)


        # 
        from dataset import MNIST
        batch_size = dataSetConfigs['batch_size']
        reshape_to_vector = dataSetConfigs['reshape_to_vector']
        n_classes = dataSetConfigs['n_classes']
        if dataSetConfigs['dataset_type'].lower() == 'train':
            
            dataset = MNIST(root=dataset_path,
                            train=True,
                            download = True,
                            transform = tensor_transform,
                            data_device='cuda',
                            reshape_to_vector=reshape_to_vector,
                            n_classes=n_classes)
            
        elif dataSetConfigs['dataset_type'].lower() == 'test':
            dataset = MNIST(root=dataset_path,
                            train=False,
                            download = True,
                            transform = tensor_transform,
                            data_device='cuda',
                            reshape_to_vector=reshape_to_vector,
                            n_classes=n_classes)
            
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    elif dataset_name == 'PointCloudTorus':
        from dataset import TorusPointCloudDataset
        torus_pc_dataset = TorusPointCloudDataset(dataSetConfigs)
        dataLoader = torch.utils.data.DataLoader(torus_pc_dataset, batch_size=3000, shuffle=True)

    else:
        raise(NotImplemented('dataset_name {} is not registered!'.format(dataset_name)))

    return dataLoader



def generate_torus_point_cloud(R, r, num_points=1000, seed=1234):
    """
    Parameters:
    - R: Major radius of the torus (distance from the center of the tube to the center of the torus).
    - r: Minor radius of the torus (radius of the tube).
    - num_points: Number of points to sample in the point cloud.

    Returns:
    - x, y, z: Arrays containing the x, y, and z coordinates of the sampled points.
    """
    np.random.seed(seed)
    u = np.random.uniform(0, 2 * np.pi, num_points)
    np.random.seed(seed+1)
    v = np.random.uniform(0, 2 * np.pi, num_points)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    points = np.vstack((x, y, z)).T

    return points



def plot_torus_point_cloud(x, y, z, ax, color='b', name='Training Data'):
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


def binarize_image(tensor):
    return (tensor > 0.5).float()



if __name__ == '__main__':
    image = plot_latent_images(None, 10)
    cv2.imwrite('./test.png', image * 255)
    # plt.imshow(figure_data)
