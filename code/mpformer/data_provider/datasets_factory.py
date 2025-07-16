from mpformer.data_provider import loader
from torch.utils.data import Dataset, DataLoader
import numpy as np

datasets_map = {
    'radar': loader,
}

def data_provider(configs):
    if configs.dataset_name == 'radar':
        train_input_param = {
            'image_width': configs.img_width,
            'image_height': configs.img_height,
            'input_data_type': 'float32',
            'is_output_sequence': True,
            'name': configs.dataset_name + 'train iterator',
            'total_length': configs.total_length,
            'data_path': configs.dataset_path,
            'type': 'train',
        }
        train_dataset = datasets_map[configs.dataset_name].InputHandle(train_input_param)
        train_input_handle = DataLoader(
            train_dataset,
            batch_size=configs.batch_size,
            shuffle=False,
            num_workers=configs.cpu_worker,
            drop_last=True
        )

    elif configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)
    
    return train_input_handle

def data_provider_val(configs):
    if configs.dataset_name == 'radar':
        valid_input_param = {
            'image_width': configs.img_width,
            'image_height': configs.img_height,
            'input_data_type': 'float32',
            'is_output_sequence': True,
            'name': configs.dataset_name + 'train iterator',
            'total_length': configs.total_length,
            'data_path': configs.dataset_path_test,
            'type': 'valid',
        }
        valid_dataset = datasets_map[configs.dataset_name].InputHandle(valid_input_param)
        valid_input_handle = DataLoader(
            valid_dataset,
            batch_size=configs.batch_size,
            shuffle=False,
            num_workers=configs.cpu_worker,
            drop_last=True
        )

    elif configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)
    
    return valid_input_handle

def test_data_provider(configs):
 
    if configs.dataset_name == 'radar':
        test_input_param = {
                            'image_width': configs.img_width,
                            'image_height': configs.img_height,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': configs.dataset_name + 'test iterator',
                            'total_length': configs.total_length,
                            'data_path': configs.dataset_path_test,
                            'type': 'test',
                            }
        test_input_handle = datasets_map[configs.dataset_path_test].InputHandle(test_input_param)
        print(len(test_input_handle))
        test_input_handle = DataLoader(test_input_handle,
                                       batch_size=configs.batch_size,
                                       shuffle=False,
                                       num_workers=configs.cpu_worker,
                                       drop_last=True)


    elif configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)

    return test_input_handle

