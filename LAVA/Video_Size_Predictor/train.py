import numpy as np
import pandas as pd
import os
import torch
cuda = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import ConcatDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

NUM = 10000

INPUT_STEPS = 8
FEATURES = 5
OUTPUT_STEPS = 1
OUTPUT = 176


df1 = pd.read_hdf(f'../latency_cst_2/Results/Driving3_bit.h5', key='encoding_data')
df2 = pd.read_hdf(f'../latency_cst_2/Results/PPE1_bit.h5', key='encoding_data')
df3 = pd.read_hdf(f'../latency_cst_2/Results/PPE2_bit.h5', key='encoding_data')
df4 = pd.read_hdf(f'../latency_cst_2/Results/Camera1_count.h5', key='encoding_data')

class VideoDataset(Dataset):
    def __init__(self, df, num, input_steps, OUTPUT_STEPS, len, seed=42):
        self.df = df
        self.num = num
        self.input_steps = input_steps
        self.output_steps = OUTPUT_STEPS
        self.len = len
        self.seed = seed

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)
        start_chunk = rng.randint(0, self.len - self.input_steps)
        X = []
        for i in range(self.input_steps):
            bit = np.random.randint(11)
            skip = np.random.randint(4)
            re = np.random.randint(4)
            x = self.df.loc[(start_chunk+i, bit, skip, re), ['Small', 'Mid', 'Large', 'Move']].values.tolist()
            x.append(bit)
            x.append(skip)
            x.append(re)
            X.append(x)
        Y = []
        for i in range(self.input_steps, self.input_steps + 1):
            for bit in range(11):
                for skip in range(4):
                    for re in range(4):
                        y = self.df.loc[(start_chunk + i, bit, skip, re), 'Ratio']
                        Y.append(y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.mode = mode
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        # 判断指标是否改善
        if (self.mode == 'min' and current_value < self.best_value - self.min_delta) or \
                (self.mode == 'max' and current_value > self.best_value + self.min_delta):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


if __name__ == '__main__':
    from val_6 import val
    os.makedirs('Results_o_1', exist_ok=True)
    log_file = open('Results_o_1/training_log_17.txt', 'w')
    log_header = (
        "Epoch | Train Loss | Val Loss 1 | Val Loss 2 | Val Loss 3 \n"
    )
    log_file.write(log_header)
    log_file.flush()

    dataset1 = VideoDataset(df1, NUM, INPUT_STEPS, OUTPUT_STEPS, 1800)
    dataset2 = VideoDataset(df2, int(NUM*0.3), INPUT_STEPS, OUTPUT_STEPS, 454)
    dataset3 = VideoDataset(df3, int(NUM*0.05), INPUT_STEPS, OUTPUT_STEPS, 64)
    dataset4 = VideoDataset(df4, NUM, INPUT_STEPS, OUTPUT_STEPS, 1800)
    dataset_train = ConcatDataset([dataset1, dataset2, dataset3, dataset4])

    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    beta_model = BetaModel(model_dim=128, output_dim=OUTPUT).to(device)
    # model_name = 'Results_o_1/100_beta_14.pth'
    # beta_model.load_state_dict(torch.load(model_name))
    beta_optimizer = optim.Adam(beta_model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(beta_optimizer, T_max=200)

    early_stopper = EarlyStopper(patience=30, mode='min')
    num_epochs = 300
    for epoch in range(num_epochs):
        beta_model.train()
        beta_loss_total = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for X, y in progress:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            beta_pred = beta_model(X)
            loss_b = F.l1_loss(beta_pred, y)

            beta_optimizer.zero_grad()
            loss_b.backward()
            beta_optimizer.step()
            beta_loss_total += loss_b.item()
            progress.set_postfix({'Loss': f'{loss_b.item():.4f}'})

        with torch.no_grad():
            val1, val2, val3 = val(beta_model, cuda)
        log_content = (
            f"[{epoch + 1:3d}/{num_epochs}]"
            f"  {beta_loss_total / len(train_loader):.4f}"
            f"  {val1:.4f}"
            f"  {val2:.4f}"
            f"  {val3:.4f}\n"
        )
        log_file.write(log_content)
        log_file.flush()
        if (epoch+1) % 10 == 0:
            torch.save(beta_model.state_dict(), f'Results_o_1/{epoch + 1:03d}_beta_17.pth')
        if early_stopper(val1):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    log_file.close()