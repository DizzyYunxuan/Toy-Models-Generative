import os
from tqdm import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from util import create_model, create_dataLoader
    
def train(configs):
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


    total_epochs = configs['TrainingDataSetConfigs']['total_epochs']
    for epoch in tqdm(range(total_epochs), desc='Epochs'):
        train_batch_progress = tqdm(train_dataLoader, desc='Train_Batch', leave=False)
        for iter_idx, train_data in enumerate(train_batch_progress):
            model.feed_data(train_data)
            model.optimize_parameters()
            model.tb_write_losses(writer, epoch * len(train_batch_progress) + iter_idx)

        model.validation(writer, epoch, test_dataLoader)
              

# with open('train_config_ae.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('train_config_vae.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('train_config_pointCloudVae.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('options/train/train_config_arPixelCNN.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('train_config_unconGAN.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('train_config_conGAN.yaml') as f:
#     configs = yaml.safe_load(f)

# with open('train_config_diffusion.yaml') as f:
#     configs = yaml.safe_load(f)

with open('options/train/train_config_transformer.yaml') as f:
    configs = yaml.safe_load(f)



train(configs)








