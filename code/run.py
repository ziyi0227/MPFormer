import os
import shutil
import argparse
import cv2
import numpy as np
import torch
from mpformer.data_provider import datasets_factory
from mpformer.models.model_factory import Model
import mpformer.evaluator as evaluator
import mpformer.train as train
import time
import sys

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MPFormer')

parser.add_argument('--device', type=str, default='cuda:0,cuda:1')
parser.add_argument('--worker', type=int, default=2)
parser.add_argument('--cpu_worker', type=int, default=2)
parser.add_argument('--dataset_name', type=str, default='radar')    # radar, radar_future, radar_nwp
parser.add_argument('--input_length', type=int, default=9)
parser.add_argument('--total_length', type=int, default=29)
parser.add_argument('--img_height', type=int, default=512)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--case_type', type=str, default='extreme')  # normal, extreme, all
parser.add_argument('--model_name', type=str, default='mpformer')
parser.add_argument('--gen_frm_dir', type=str, default='results/mpformer')
parser.add_argument('--pretrained_model', type=str, default='mpformer.ckpt')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--adapter', type=bool, default=True)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--ngf', type=int, default=32)  # number of generator filters in first conv layer
parser.add_argument('--dataset_path', type=str, default='data/dataset/mrms/figure') 
parser.add_argument('--dataset_path_test', type=str, default='data/dataset/mrms/figure_test') 
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--step_size', type=int, default=4000) # 学习率调度器的步长，每个step_size个epoch衰减gamma倍
parser.add_argument('--checkpoint_dir', type=str, default='data/checkpoints/train02')
parser.add_argument('--temperature', type=int, default=0.07)

args = parser.parse_args()

args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.ic_feature = args.ngf * 10

def test_wrapper_pytorch_loader(model):
    batch_size_test = args.batch_size
    test_input_handle = datasets_factory.data_provider(args)
    args.batch_size = batch_size_test
    evaluator.test_pytorch_loader(model, test_input_handle, args, 'test_result')

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('Initializing models')

train.train_pytorch_loader(args)