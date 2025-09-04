import os
import sys
import numpy as np
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
import argparse

from core.models.model_factory import Model
from core.data_provider import datasets_factory
from core.utils import preprocess
from configs import configs

def parse_args():
    parser = argparse.ArgumentParser(description='PredRNN Training')
    parser.add_argument('--is_training', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--train_data_paths', type=str, required=True)
    parser.add_argument('--valid_data_paths', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--gen_frm_dir', type=str, required=True)
    parser.add_argument('--input_length', type=int, required=True)
    parser.add_argument('--total_length', type=int, required=True)
    parser.add_argument('--img_width', type=int, required=True)
    parser.add_argument('--img_height', type=int, required=True)
    parser.add_argument('--img_channel', type=int, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--num_hidden', type=str, required=True)
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=100)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_iterations', type=int, default=100)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=100)
    parser.add_argument('--snapshot_interval', type=int, default=100)
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--save_output', type=int, default=1)
    # Tambahkan argumen lain sesuai kebutuhan
    return parser.parse_args()

def get_configs(args):
    configs.is_training = args.is_training
    configs.device = args.device
    configs.dataset_name = args.dataset_name
    configs.train_data_paths = args.train_data_paths
    configs.valid_data_paths = args.valid_data_paths
    configs.save_dir = args.save_dir
    configs.gen_frm_dir = args.gen_frm_dir
    configs.input_length = args.input_length
    configs.total_length = args.total_length
    configs.img_width = args.img_width
    configs.img_height = args.img_height
    configs.img_channel = args.img_channel
    configs.model_name = args.model_name
    configs.pretrained_model = args.pretrained_model
    configs.num_hidden = args.num_hidden
    configs.filter_size = args.filter_size
    configs.stride = args.stride
    configs.patch_size = args.patch_size
    configs.layer_norm = args.layer_norm
    configs.scheduled_sampling = args.scheduled_sampling
    configs.sampling_stop_iter = args.sampling_stop_iter
    configs.sampling_start_value = args.sampling_start_value
    configs.sampling_changing_rate = args.sampling_changing_rate
    configs.lr = args.lr
    configs.reverse_input = args.reverse_input
    configs.batch_size = args.batch_size
    configs.max_iterations = args.max_iterations
    configs.display_interval = args.display_interval
    configs.test_interval = args.test_interval
    configs.snapshot_interval = args.snapshot_interval
    configs.num_save_samples = args.num_save_samples
    configs.save_output = args.save_output
    # Tambahkan konfigurasi lain sesuai kebutuhan
    return configs

def main():
    args = parse_args()
    configs = get_configs(args)
    
    # Buat direktori hasil jika belum ada
    if not os.path.exists(configs.gen_frm_dir):
        os.makedirs(configs.gen_frm_dir)
    
    # Inisialisasi model
    model = Model(configs)
    if configs.pretrained_model != '':
        model.load(configs.pretrained_model)
    model.network.eval()
    
    # Pindahkan model ke device yang sesuai
    device = torch.device(configs.device)
    model.network.to(device)
    
    # Tambahkan kode untuk pelatihan atau pengujian model
    # ...

if __name__ == '__main__':
    main()
