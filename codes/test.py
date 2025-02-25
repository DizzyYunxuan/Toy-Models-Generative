import os
import torch
from tqdm import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from util import create_model, create_dataLoader


def test(configs):
    # init dataset
    TestingDataSetConfigs = configs['TestingDataSetConfigs']
    test_dataLoader = create_dataLoader(TestingDataSetConfigs)

    # init model
    model = create_model(configs)

    # load model
    pretrained_model_path = configs['inferenceConfigs']['load_path']
    model.load_state_dict(torch.load(pretrained_model_path))

    # init tensorboard
    writer = SummaryWriter('expriments_tb/{}'.format(configs['Configuration_name'] + '_Inference'))
    
    model.inference(writer, test_dataLoader)
              
with open('options/test/test_config_transformer.yaml') as f:
    configs = yaml.safe_load(f)

test(configs)