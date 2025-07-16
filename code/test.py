import os
import glob
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm
from mpformer.data_provider import datasets_factory
from mpformer.models.model_factory import Model
import mpformer.evaluator as evaluator
import mpformer.evaluator_time as evaluator_time

def parse_args():
    parser = argparse.ArgumentParser(description='MPFormer Test')

    # reuse training args
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--worker', type=int, default=2)
    parser.add_argument('--cpu_worker', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default='radar')
    parser.add_argument('--input_length', type=int, default=9)
    parser.add_argument('--total_length', type=int, default=29)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_ch', type=int, default=2)
    parser.add_argument('--case_type', type=str, default='extreme')
    parser.add_argument('--model_name', type=str, default='mpformer')
    parser.add_argument('--gen_frm_dir', type=str, default='results/mpformer_val9')
    parser.add_argument('--checkpoint_dir', type=str, default='model',
                        help='Directory containing saved .ckpt files')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--adapter', type=bool, default=False)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--dataset_path', type=str, default='data/mrms_test')
    parser.add_argument('--dataset_path_test', type=str, default='data/mrms_demo')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--num_save_samples', type=int, default=100)

    args = parser.parse_args()
    # derived args
    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10
    return args


def find_best_checkpoint(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not paths:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    # select latest by modification time
    return max(paths, key=os.path.getmtime)

def model_test(self, test_ims: np.ndarray) -> np.ndarray:
    """
    Run inference on numpy input array of shape [batch, time, height, width, channels]
    and return numpy predictions of shape [batch, output_time, height, width, channels].
    """
    self.eval()
    with torch.no_grad():
        # Convert input to torch tensor and move to model device
        input_tensor = torch.from_numpy(test_ims).float().to(next(self.parameters()).device)
        # Forward pass
        preds = self(input_tensor)
        # Move back to CPU and convert to numpy
        return preds.cpu().numpy()

# Attach to Model class
Model.test = model_test

def main():
    args = parse_args()

    # prepare output directory
    if os.path.exists(args.gen_frm_dir):
        shutil.rmtree(args.gen_frm_dir)
    os.makedirs(args.gen_frm_dir, exist_ok=True)

    # load model from checkpoint
    ckpt_path = find_best_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    model = Model.load_from_checkpoint(checkpoint_path=ckpt_path, configs=args)
    model.freeze()
    model = model.to(args.device)

    # Create validation DataLoader and wrap with tqdm for progress bar
    raw_loader = datasets_factory.data_provider_val(args)
    total_batches = len(raw_loader)
    val_loader = tqdm(raw_loader, desc='Validation', total=total_batches)

    # run test
    print("Starting validation test...")
    evaluator.test_pytorch_loader2(model, val_loader, args, itr='val')
    # evaluator_time.test_pytorch_loader(model, val_loader, args, itr='val')
    print("Validation testing complete.")

if __name__ == '__main__':
    main()
