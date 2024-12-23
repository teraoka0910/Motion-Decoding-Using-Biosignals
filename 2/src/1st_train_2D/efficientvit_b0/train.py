import os
import warnings


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler


from my_utils import get_data, SkateDataset, train, valid, Earlystopping, seed_everything, get_model, MyPipeline


warnings.simplefilter('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = '../../../data/train'


def main():
    seed_everything(42)
    batch_size = 32
    epochs = 100
    lr = 1.e-3
    
    data_dict, label_dict, rate_dict, diff_list, use_ch_dict = get_data(data_dir)
    pipeline = MyPipeline(samplerate=500, lowcut=1, highcut=180, wavelet_width=1, n_scales=16, stride=1).to(device)

    for fold in range(3):
        with open('log.txt', 'a', encoding='shift_jis') as f:
            f.write(f'k-fold: {fold} -> 学習開始\n')
            print(f'k-fold: {fold}')
        
        valid_data_list = data_dict[fold]
        valid_label_list = label_dict[fold]
        valid_rate_list = rate_dict[fold]
        if fold == 0:
            train_data_list = data_dict[1] + data_dict[2]
            train_label_list = label_dict[1] + label_dict[2]
            train_rate_list = rate_dict[1] + rate_dict[2]
        elif fold == 1:
            train_data_list = data_dict[0] + data_dict[2]
            train_label_list = label_dict[0] + label_dict[2]
            train_rate_list = rate_dict[0] + rate_dict[2]
        elif fold == 2:
            train_data_list = data_dict[0] + data_dict[1]
            train_label_list = label_dict[0] + label_dict[1]
            train_rate_list = rate_dict[0] + rate_dict[1]
        
        train_dataset = SkateDataset(fold, train_data_list, train_label_list, train_rate_list, diff_list, use_ch_dict, phase='train')
        valid_dataset = SkateDataset(fold, valid_data_list, valid_label_list, valid_rate_list, diff_list, use_ch_dict, phase='valid')

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True, pin_memory=True)#, collate_fn=collate_fn
        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

        model = get_model()
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        early_stopping = Earlystopping(patience=10, verbose=True, model_name=f'model/model{fold}.pth')
        criterion =  nn.BCEWithLogitsLoss()
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1.e-5, warmup_t=2, warmup_lr_init=1.e-5, warmup_prefix=True)

        for epoch in range(epochs):
            phase = 'train'
            print(f'===================== {phase} =====================')                        
            train(epoch, epochs, train_loader, batch_size, model, pipeline, optimizer, criterion)
            
            phase = 'valid'
            print(f'===================== {phase} =====================')
            val_loss = valid(epoch, valid_loader, model, pipeline, nn.CrossEntropyLoss())
            print(f'val_loss: {val_loss:.3f}')
            
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                torch.cuda.empty_cache()
                break

            scheduler.step(epoch+1)


if __name__ == '__main__':
    main()