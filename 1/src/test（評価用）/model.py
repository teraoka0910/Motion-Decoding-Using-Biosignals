import torch
import torch.nn as nn
import timm

CROP_LEN_ = 250


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
    def __init__(self, in_channels=50, num_samples=250):
        super(EEGNet, self).__init__()
        depth = 5
        temporal_filters = 16
        
        self.temporal_conv = nn.Conv2d(1, temporal_filters, (1, num_samples//2), bias=False, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(temporal_filters)
        
        self.depthwise_conv = nn.Conv2d(temporal_filters, depth*temporal_filters, (in_channels, 1), groups=temporal_filters, bias=False, padding='valid')
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
    

class EEG1DTemporal(nn.Module):
    def __init__(self, in_channels=50, num_samples=250):
        super(EEG1DTemporal, self).__init__()
        self.pad = nn.ZeroPad2d((3, 3, 3, 3))
        self.conv2d_freq = nn.Conv2d(1, 125, (1, num_samples//2), padding='same', bias=False)
        self.bn_freq = nn.BatchNorm2d(125)
        self.conv2d_depth = nn.Conv2d(125, 250, (in_channels, 1), groups=125, bias=False, padding='valid')
        self.bn_depth = nn.BatchNorm2d(250)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv2d_freq(x)
        x = self.bn_freq(x)
        x = self.conv2d_depth(x)
        x = self.bn_depth(x)
        x = self.elu(x)
        x = x[:, :, 0, :]
        x = x.unsqueeze(1)
        x = self.pad(x)

        return x
        
    
class EEG2DCNN(nn.Module):
    def __init__(self, model_name):
        super(EEG2DCNN, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=3)

    def forward(self, x):
        self.features = []
        if 'tf_efficientnet' in self.model_name:
            hook = self.model.global_pool.register_forward_hook(self.hook_fn)
        elif 'efficientvit' in self.model_name:
            hook = self.model.head.global_pool.register_forward_hook(self.hook_fn)
        x = self.model(x)
        
        return x, self.features[0]
    
    def hook_fn(self, module, input, output):
        self.features.append(output)
    
    
class OneEncoderNet(nn.Module):
    def __init__(self, encoder=None, in_channels=50):
        super(OneEncoderNet, self).__init__()
        self.first_dropout = nn.Dropout(0.4)
        self.encoder = ConvEncoder(in_channels) #encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.first_dropout(x)
        x = self.encoder(x)
        x = self.pool(x)
        bs, channels, _ = x.shape
        x_ = x.reshape(bs, channels)
        x = self.classifier(x_)

        return x, x_
    

class TwoEncoderNet(nn.Module):
    def __init__(self, encoder=None, in_channels=50):
        super(TwoEncoderNet, self).__init__()
        self.encoder1 = ConvEncoder(in_channels)
        self.first_dropout = nn.Dropout(0.4)
        self.encoder2 = EEGNet(in_channels=in_channels)
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
        x_ = torch.cat([x1, x2], dim=1)
        x = self.classifier(x_)

        return x, x_
    

class TemporalNet(nn.Module):
    def __init__(self, model_name, in_channels):
        super(TemporalNet, self).__init__()
        self.oned_encoder = EEG1DTemporal(in_channels=in_channels)
        self.twod_encoder = EEG2DCNN(model_name)
        self.first_dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.first_dropout(x)
        x = self.oned_encoder(x)
        x = self.twod_encoder(x)

        return x