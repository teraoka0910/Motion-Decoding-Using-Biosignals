import glob
import os
import random
from collections import defaultdict

from pymatreader import read_mat
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
from torch.utils.data import Dataset


short_progress_bar="{l_bar}{bar:20}{r_bar}{bar:-10b}"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CROP_LEN_ = 250


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(data_dir):
    """
    クロスバリデーションのfold分け
    ラベルの取得
    差分時系列の設定
    """
    data_dict = defaultdict(list)
    label_dict = defaultdict(list)
    rate_dict = defaultdict(list)
    crop_len = CROP_LEN_

    for subject in range(5):
        for mat_data in glob.glob(f'{data_dir}/subject{subject}/*'):
            data = read_mat(mat_data)
            start_indexes = (data['event']['init_time'] + 0.2)*1000 // 2
            end_indexes = (data['event']['init_time'] + 0.7)*1000 // 2
            labels = data['event']['type']

            for start_index, end_index, label in zip(start_indexes, end_indexes, labels):
                start_index_ = int(start_index.item() - crop_len)
                end_index_ = int(end_index.item()) + 2
                for item in range(start_index_, end_index_, 2):
                    count = item + crop_len - start_index.item()
                    rate = count / crop_len
                    if count > 250:
                        count -= 250
                        rate = 1 - count / crop_len
                    
                    if (subject == 0) or (subject == 3):
                        if 'train1' in mat_data:
                            data_dict[0].append(f'{mat_data}_{item}')
                            label_dict[0].append(int(str(int(label))[-1])-1)
                            rate_dict[0].append(rate)
                        elif 'train2' in mat_data:
                            data_dict[1].append(f'{mat_data}_{item}')
                            label_dict[1].append(int(str(int(label))[-1])-1)
                            rate_dict[1].append(rate)
                        elif 'train3' in mat_data:
                            data_dict[2].append(f'{mat_data}_{item}')
                            label_dict[2].append(int(str(int(label))[-1])-1)
                            rate_dict[2].append(rate)
                    elif (subject == 1) or (subject == 4):
                        if 'train2' in mat_data:
                            data_dict[0].append(f'{mat_data}_{item}')
                            label_dict[0].append(int(str(int(label))[-1])-1)
                            rate_dict[0].append(rate)
                        elif 'train3' in mat_data:
                            data_dict[1].append(f'{mat_data}_{item}')
                            label_dict[1].append(int(str(int(label))[-1])-1)
                            rate_dict[1].append(rate)
                        elif 'train1' in mat_data:
                            data_dict[2].append(f'{mat_data}_{item}')
                            label_dict[2].append(int(str(int(label))[-1])-1)
                            rate_dict[2].append(rate)
                    elif subject == 2:
                        if 'train3' in mat_data:
                            data_dict[0].append(f'{mat_data}_{item}')
                            label_dict[0].append(int(str(int(label))[-1])-1)
                            rate_dict[0].append(rate)
                        elif 'train1' in mat_data:
                            data_dict[1].append(f'{mat_data}_{item}')
                            label_dict[1].append(int(str(int(label))[-1])-1)
                            rate_dict[1].append(rate)
                        elif 'train2' in mat_data:
                            data_dict[2].append(f'{mat_data}_{item}')
                            label_dict[2].append(int(str(int(label))[-1])-1)
                            rate_dict[2].append(rate)

    ch_names = [c.replace(' ', '') for c in data['ch_labels']]
    diff_list = [
        #横方向
        'F3_F4',
        'FCz_FC1', 'FCz_FC2', 'FCz_FC3', 'FCz_FC4', 'FCz_FC5', 'FCz_FC6', 'FC1_FC2', 'FC3_FC4', 'FC5_FC6',
        'Cz_C1', 'Cz_C2', 'Cz_C3', 'Cz_C4', 'Cz_C5', 'Cz_C6', 'C1_C2', 'C3_C4', 'C5_C6',
        'CPz_CP1', 'CPz_CP2', 'CPz_CP3', 'CPz_CP4', 'CPz_CP5', 'CPz_CP6', 'CP1_CP2', 'CP3_CP4', 'CP5_CP6',
        'P3_P4',
        #縦方向
        'Cz_FCz', 'C1_FC1', 'C2_FC2', 'C3_FC3', 'C4_FC4', 'C5_FC5', 'C6_FC6',
        'Cz_CPz', 'C1_CP1', 'C2_CP2', 'C3_CP3', 'C4_CP4', 'C5_CP5', 'C6_CP6',
        'FCz_CPz', 'FC1_CP1', 'FC2_CP2', 'FC3_CP3', 'FC4_CP4', 'FC5_CP5', 'FC6_CP6',
    ]

    use_ch = []
    for item in diff_list:
        ch1 = item.split('_')[0]
        ch2 = item.split('_')[1]
        use_ch.append(ch1)
        use_ch.append(ch2)

    use_ch = list(set(use_ch))
    use_ch_dict = {ch_names[idx]:idx for idx in range(len(ch_names)) if ch_names[idx] in use_ch}
    
    return data_dict, label_dict, rate_dict, diff_list, use_ch_dict


class SkateDataset(Dataset):
    """
    前処理部分
    """
    def __init__(self, fold, data_list, label_list, rate_list, diff_list, use_ch_dict, phase='train'):
        self.phase = phase
        self.label_list = label_list
        self.rate_list = rate_list
        self.diff_list = diff_list
        self.use_ch_dict = use_ch_dict
        self.crop_len = CROP_LEN_

        if phase == 'train':
            target_index = np.where(np.array(rate_list)>0.1)[0].tolist()
            data_list = np.array(data_list)[target_index].tolist()
            self.label_list = np.array(label_list)[target_index].tolist()
            self.rate_list = np.array(rate_list)[target_index].tolist()
        elif phase == 'valid':
            target_index = np.where(np.array(rate_list)==1)[0].tolist()
            data_list = np.array(data_list)[target_index].tolist()
            self.label_list = np.array(label_list)[target_index].tolist()
            self.rate_list = np.array(rate_list)[target_index].tolist()

        self.file_list = [item.split('_')[0] for item in data_list]
        self.index_list = [item.split('_')[1] for item in data_list]
        file_name_list = list(set(self.file_list))

        self.data_dict = {}
        for file_name in tqdm(file_name_list):
            data = read_mat(file_name)['data']
            self.data_dict[file_name] = data

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        idx = int(self.index_list[index])
        label = self.label_list[index]
        rate = self.rate_list[index]
        eeg_signal = self.data_dict[file_name][:, idx:idx+self.crop_len]
        
        median = np.median(eeg_signal, axis=1).reshape(72, 1)
        q1 = np.percentile(eeg_signal, 25, axis=1)
        q3 = np.percentile(eeg_signal, 75, axis=1)
        iqr = (q3 - q1).reshape(72, 1)
        iqr = np.where(iqr==0, 1, iqr)
        eeg_signal = (eeg_signal - median) / iqr

        if self.phase == 'train':
            eeg_signal = self.transform(eeg_signal)
        
        return eeg_signal, label, rate

    def transform(self, signal):
        if np.random.uniform() < 0.25:
            signal = self.time_warp(signal)
        if np.random.uniform() < 0.25:
            if np.random.uniform() < 0.5:
                signal = self.white_noise(signal)
            else:
                signal = self.gaussian_noise(signal)
        
        return signal
        
    def white_noise(self, signal):
        noise = np.random.normal(0, 0.1, signal.shape)

        return signal + noise
    
    def gaussian_noise(self, signal):
        noise = np.random.normal(0, 0.1, signal.shape)

        return signal + gaussian_filter(noise, sigma=1)
    
    def time_warp(self, signal, sigma=0.2):
        orig_steps = np.arange(signal.shape[1], dtype=np.uint8)
        new_steps = np.linspace(0, signal.shape[1] - 1, signal.shape[1])
        warping = np.random.normal(loc=1.0, scale=sigma, size=signal.shape[1])
        warping = np.cumsum(warping) / np.sum(warping) * (signal.shape[1] - 1)
        warping = np.clip(warping, 0, signal.shape[1] - 1)
        f = interp1d(orig_steps, signal, kind='linear', axis=1)
        
        return f(warping)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn_residual = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        residual = self.bn_residual(residual)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.gelu(out)
        out = self.pool(out)
        out = self.dropout(out)
        
        return out
    

class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ConvEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 1024)
        )

    def forward(self, x):
        return self.conv_layers(x)


class EEGNet(nn.Module):
    def __init__(self, num_channels=50, num_samples=250):
        super(EEGNet, self).__init__()
        depth = 5
        temporal_filters = 16
        
        self.temporal_conv = nn.Conv2d(1, temporal_filters, (1, num_samples//2), bias=False, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(temporal_filters)
        
        self.depthwise_conv = nn.Conv2d(temporal_filters, depth*temporal_filters, (num_channels, 1), groups=temporal_filters, bias=False, padding='valid')
        self.batch_norm2 = nn.BatchNorm2d(depth*temporal_filters)
        
        self.avg_pool = nn.AvgPool2d((1, 4))

        self.sep_conv = nn.Conv2d(depth*temporal_filters, depth*temporal_filters*2, (1, num_samples//4), groups=depth*temporal_filters, bias=False, padding='same')
        self.batch_norm3 = nn.BatchNorm2d(depth*temporal_filters*2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.25)
        self.activate = nn.ELU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.batch_norm1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.activate(x)
        x = self.avg_pool(x)
        x = self.dropout(x)

        x = self.sep_conv(x)
        x = self.batch_norm3(x)
        x = self.activate(x) 
        x = self.avg_pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        
        return x
    

class SkateModel(nn.Module):
    def __init__(self, encoder=None, in_channels=72):
        super(SkateModel, self).__init__()
        self.encoder1 = ConvEncoder(in_channels)
        self.first_dropout = nn.Dropout(0.4)
        self.encoder2 = EEGNet(num_channels=72)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024+1120, 512, bias=False),
            nn.Dropout(0.4),
            nn.Linear(512, 3, bias=False),
        )

    def forward(self, x):
        x = self.first_dropout(x)
        x1 = self.encoder1(x)
        x1 = self.pool(x1)
        bs, channels, _ = x1.shape
        x1 = x1.reshape(bs, channels)

        x2 = self.encoder2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)

        return x


def get_model():
    model = SkateModel()

    return model
    

def train(epoch, epochs, loader, batch_size, model, optimizer, criterion, scaler):
    """
    学習部分
    """
    train_loss_temp = 0
    accuracy_temp = 0
    correct = 0
    print_cycle = 1000
    label_smooth = 0.1
    model.train()
    optimizer.zero_grad()

    with tqdm(loader, desc=f'Epoch {epoch+1} / {epochs} ', bar_format=short_progress_bar) as t:
        for i, (X, label, rate) in enumerate(t):
            X = X.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            label_ = label.clone()
            label = nn.functional.one_hot(label, num_classes=3).float()
            rate = rate.to(device, non_blocking=True).float().unsqueeze(1)
            label = label * rate

            pos_weight = torch.zeros_like(label)
            pos_weight[torch.where(label>=0.1)] = 5.0
            pos_weight *= torch.tensor([2.0, 2.0, 1.0]).to(device)
            pos_weight = torch.where(pos_weight==0, torch.tensor(1.).float().to(device), pos_weight)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')

            target_index = torch.where(label>=0.3)[0]
            label[target_index] = torch.where(label[target_index]==0, label_smooth, label[target_index]-label_smooth)

            y_pred = model(X)
            loss = criterion(y_pred, label)

            y_pred = nn.functional.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_temp = y_pred.eq(label_.view_as(y_pred)).sum().item()
            correct += correct_temp
            accuracy = correct_temp / X.shape[0]
            accuracy_temp += accuracy

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mes = f'loss: {loss.item():.3f}, accuracy: {accuracy:.3f}'
            train_loss_temp += loss.item()
            t.postfix = mes

            if (i+1)%print_cycle == 0:
                train_loss = train_loss_temp / print_cycle
                accuracy = accuracy_temp / print_cycle
                print(f'Epoch: {epoch+1}/{epochs} {(i+1)*batch_size}/{batch_size*len(loader)} loss: {train_loss:.3f} accuracy: {accuracy:.3f}')
                train_loss_temp = 0
                accuracy_temp = 0
                with open('log.txt', 'a', encoding='shift_jis') as f:
                    f.write(f'iter: {i+1} -> train_loss: {train_loss:.3f} accuracy: {accuracy:.3f} \n')

    torch.cuda.empty_cache()


def valid(epoch, loader, model, criterion):
    """
    検証部分
    """
    val_loss = 0
    correct = 0
    model.eval()

    with torch.no_grad():
        with tqdm(loader, bar_format=short_progress_bar) as t:
            for i, (X, label, rate) in enumerate(t):
                X = X.to(device, non_blocking=True).float()
                label = label.to(device, non_blocking=True).long()
                rate = rate.to(device, non_blocking=True).float()

                y_pred = model(X)
                loss = criterion(y_pred, label)
                val_loss += loss.item()

                y_pred = nn.functional.softmax(y_pred, dim=1)
                y_pred = torch.argmax(y_pred, dim=1)
                correct_temp = y_pred.eq(label.view_as(y_pred)).sum().item()
                correct += correct_temp
                accuracy = correct_temp / X.shape[0]

                mes = f'loss: {loss.item():.3f}, accuracy: {accuracy:.3f}'
                t.postfix = mes

    accuracy = correct / len(loader.dataset)
    valid_loss = val_loss/len(loader)
    with open('log.txt', 'a', encoding='shift_jis') as f:
        f.write(f'Epoch: {epoch+1}  ')

    with open('log.txt', 'a', encoding='shift_jis') as f:
        f.write(f'Loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}\n')

    print(f'Loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}')

    return 1 - accuracy


class Earlystopping:
    def __init__(self, patience=10, verbose=False, model_name='model.pth'):
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.model_name = model_name
        self.save_flag = False

    def __call__(self, val_loss, model):
        self.save_flag = False
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.save_flag = True
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.save_flag = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_name)
        self.val_loss_min = val_loss


