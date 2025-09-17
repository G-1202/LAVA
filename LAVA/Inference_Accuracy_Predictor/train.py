import numpy as np
import pandas as pd
import os
cuda = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from val import val
from network import I_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

NUM = 20000

INPUT_STEPS = 8
FEATURES = 5
OUTPUT_STEPS = 1
OUTPUT = 176

frame_path = f'frame_dataset'

class VideoDataset(Dataset):
    def __init__(self, df, name, num, input_steps, OUTPUT_STEPS, len=1800, seed=52):
        self.df = df
        self.name = name
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
        X1 = []
        vf_img = None
        for i in range(self.input_steps):
            bit = np.random.randint(11)
            skip = np.random.randint(4)
            re = np.random.randint(4)
            x = self.df.loc[(start_chunk+i, bit, skip, re), ['Small', 'Mid', 'Large', 'Move']].values.tolist()
            x.append(bit)
            x.append(skip)
            x.append(re)
            X1.append(x)

            if i == self.input_steps - 1:
                file_name = f"{start_chunk + i + 1}_{bit * 16 + skip * 4 + re}.jpg"
                image_path = os.path.join(frame_path, f"frame_{self.name}", file_name)
                img = cv2.imread(image_path)
                if img is None:
                    raise FileNotFoundError(f"Unable to read image file: {image_path}")
                img = cv2.resize(img, (224, 224))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_proc = img_rgb.astype(np.float32) / 255.0
                vf_img = torch.tensor(img_proc, dtype=torch.float32).permute(2, 0, 1)

        X1 = torch.tensor(X1, dtype=torch.float32)
        Y = []
        for i in range(self.input_steps, self.input_steps + 1):
            for bit in range(11):
                for skip in range(4):
                    for re in range(4):
                        y = self.df.loc[(start_chunk + i, bit, skip, re), 'Accuracy']
                        Y.append(y)
        Y = torch.tensor(Y, dtype=torch.float32)
        return X1, vf_img, Y

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
    os.makedirs('Results_a', exist_ok=True)
    log_file = open('Results_a/training_log.txt', 'a')
    log_header = (
        "Epoch | Train Loss | Val Loss \n"
    )
    log_file.write(log_header)
    log_file.flush()

    # Create your own dataset
    df1 = pd.read_hdf(f'dataset/train.h5', 'encoding_data')
    dataset_train = VideoDataset(df1, 'train', NUM, INPUT_STEPS, OUTPUT_STEPS)

    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    acc_model = I_Model(model_dim=256, output_dim=OUTPUT).to(device)
    acc_optimizer = optim.Adam(acc_model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(acc_optimizer, T_max=200)

    early_stopper = EarlyStopper(patience=30, mode='min')
    num_epochs = 500
    for epoch in range(num_epochs):
        acc_model.train()
        acc_loss_total = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for X1, X2, y in progress:
            X1 = X1.to(device, non_blocking=True)
            X2 = X2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            acc_pred = acc_model(X1, X2)
            loss_a = F.l1_loss(acc_pred, y)

            acc_optimizer.zero_grad()
            loss_a.backward()
            acc_optimizer.step()
            acc_loss_total += loss_a.item()
            progress.set_postfix({'Loss': f'{loss_a.item():.4f}'})

        with torch.no_grad():
            val = val(acc_model, cuda)
        log_content = (
            f"[{epoch + 1:3d}/{num_epochs}]"
            f"  {acc_loss_total / len(train_loader):.4f}"
            f"  {val:.4f}\n"
        )
        log_file.write(log_content)
        log_file.flush()
        if (epoch + 1) % 10 == 0:
            torch.save(acc_model.state_dict(), f'model/{epoch + 1:03d}_acc.pth')
        if early_stopper(val):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    log_file.close()