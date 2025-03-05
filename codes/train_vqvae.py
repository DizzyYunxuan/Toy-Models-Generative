'''
vqvae
pixel cnn
'''
import torch
import os
from tqdm import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from util import create_model, create_dataLoader

def train_vqvae(configs):
    # init dataset
    TrainingDataSetConfigs = configs['TrainingDataSetConfigs']
    TestingDataSetConfigs = configs['TestingDataSetConfigs']
    train_dataLoader = create_dataLoader(TrainingDataSetConfigs)
    test_dataLoader = create_dataLoader(TestingDataSetConfigs)


    # init model
    model = create_model(configs)

    # init tensorboard
    writer = SummaryWriter('expriments_tb/{}'.format(configs['Configuration_name']))

    # save path
    os.makedirs('./expriments_save/{}'.format(configs['Configuration_name']), exist_ok=True)


    # total_epochs = configs['TrainingDataSetConfigs']['total_epochs']
    total_vqvae_epochs = configs['TrainingDataSetConfigs']['total_vqvae_epochs']
    total_pixelcnn_epochs = configs['TrainingDataSetConfigs']['total_pixelcnn_epochs']
    # for epoch in tqdm(range(total_vqvae_epochs), desc='Epochs'):
    #     train_batch_progress = tqdm(train_dataLoader, desc='Train_Batch', leave=False)
    #     for iter_idx, train_data in enumerate(train_batch_progress):
    #         model.feed_data(train_data)
    #         model.optimize_parameters_vqvae()
    #         model.tb_write_losses_vqvae(writer, epoch * len(train_batch_progress) + iter_idx)

    #     model.validation_vqvae(writer, epoch, test_dataLoader)
    
    model.vqvae_module.load_state_dict(torch.load('expriments_save/20250226_train_VQVAE_conditional/model_epoch_19.pth'), strict=True)
    model.validation_vqvae(writer, 100, test_dataLoader)

    for epoch in tqdm(range(total_pixelcnn_epochs), desc='Epochs'):
        train_batch_progress = tqdm(train_dataLoader, desc='Train_Batch', leave=False)
        for iter_idx, train_data in enumerate(train_batch_progress):
            model.feed_data(train_data)
            model.optimize_parameters_pixelcnn()
            model.tb_write_losses_pixelcnn(writer, epoch * len(train_batch_progress) + iter_idx)

        model.validation_pixelcnn(writer, epoch)


with open('options/train/train_config_vqvae.yaml') as f:
    configs = yaml.safe_load(f)


train_vqvae(configs)