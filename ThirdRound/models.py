import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Conv2D with ReLU for the ToF
"""
class Conv2DReLUBN(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ToF_2D_Block(nn.Module):
    def __init__(self, output_channels, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2DReLUBN(
                in_channels=1,
                output_channels=output_channels,
                kernel_size=kernel_size
            ),
            nn.MaxPool2d(kernel_size=2),

            Conv2DReLUBN(
                in_channels=output_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            ),
            nn.MaxPool2d(kernel_size=2),

            Conv2DReLUBN(
                in_channels=output_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            ),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.layers(x)


"""
The Conv1D with LSTM for The IMU, the Tof, The thm
""" 

class Conv1DReLUBN_LSTM(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size,
                 stride=1, groups=1, lstm_hidden=128):
        super().__init__()

        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

        self.lstm_block = nn.LSTM(
            input_size=output_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Linear(2 * lstm_hidden, output_channels)

    def forward(self, x):
        """
        x : (batch, channels, seq_len)
        """
        #CNN forward
        x = self.conv_block(x) 
        
        #Prepare the CNN -> LSTM
        x = x.permute(0, 2, 1)#(batch, channels, lenght)
        
        #LSTM forward
        x, _ = self.lstm_block(x)
        
        #Apply Projection
        x = self.projection(x)
        
        # Return in time-major or batch-major depending on your next block
        x = x.permute(0, 2, 1) #(batch, channels, lenght)
        return x




class MLP(nn.Module):
    def __init__(self, number_channels, mlp_dropout, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(number_channels, number_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),

            nn.Linear(number_channels // 2, number_channels // 4),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),

            nn.Linear(number_channels // 4, out_channels)
        )

    def forward(self, x):
        return self.layers(x)



class UNet1D(nn.Module):
    
    def __init__(self, in_channels=6, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU()
        )

        
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m  = self.mid(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)  



class CNN1D_LSTM_Branch(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 initial_channels_per_feature,
                 CNN1D_channels_size,
                 CNN1D_kernel_size,
                 mlp_dropout,
                 lstm_hidden=128):
        super().__init__()

        self.inital_layers = Conv1DReLUBN_LSTM(
            in_channels=input_channels,
            output_channels=input_channels * initial_channels_per_feature,
            kernel_size=CNN1D_kernel_size,
            stride=1,
            groups=input_channels,
            lstm_hidden=lstm_hidden
        )

        CNN1D_layers = []
        for i in range(len(CNN1D_channels_size)):
            in_ch = (
                input_channels * initial_channels_per_feature
                if i == 0 else CNN1D_channels_size[i - 1]
            )
            out_ch = CNN1D_channels_size[i]

            CNN1D_layers.append(
                Conv1DReLUBN_LSTM(
                    in_channels=in_ch,
                    output_channels=out_ch,
                    kernel_size=CNN1D_kernel_size,
                    stride=1,
                    groups=1,
                    lstm_hidden=lstm_hidden
                )
            )

            if i < len(CNN1D_channels_size) - 1:
                CNN1D_layers.append(nn.MaxPool1d(kernel_size=3, stride=3))

        self.CNN1D_layers = nn.Sequential(*CNN1D_layers)

        self.mlp = MLP(
            number_channels=CNN1D_channels_size[-1] * 2,
            mlp_dropout=mlp_dropout,
            out_channels=output_channels  # 128
        )

    def forward(self, x, gesture_mask=None):
        """
        x: (B, C, T)
        gesture_mask: (B, 1, T) o None
        """
        x = self.inital_layers(x)   # (B, C', T')
        x = self.CNN1D_layers(x)    # (B, C_last, T')

        
        if gesture_mask is not None:
            if gesture_mask.dim() == 2:
                gesture_mask = gesture_mask.unsqueeze(1)

        
            if gesture_mask.size(2) != x.size(2):
                T = min(gesture_mask.size(2), x.size(2))
                x = x[:, :, :T]
                gesture_mask = gesture_mask[:, :, :T]

            x = x * gesture_mask  

        
        x_mean = x.mean(dim=2)
        x_max = x.max(dim=2).values
        x = torch.cat([x_mean, x_max], dim=1)  

        out = self.mlp(x)  # (B, 128)
        return x, out

#gesture_mask = torch.sigmoid(self.unet_1d(x))
NUM_CLASSES = 18
BRANCH_FEATURES = 128


class MultiBranchClassifier(nn.Module):
    def __init__(
        self,
        number_imu_blocks,
        in_channels,
        out_channels,                 
        initial_channels_per_feature,
        cnn1d_channels,
        cnn1d_kernel_size,
        ToF_out_channels,
        ToF_kernel_size,
        THM_out_channels,
        THM_kernel_size,
        mlp_dropout,
        lstm_hidden=256
    ):
        super().__init__()

        num_classes = out_channels       
        branch_features = BRANCH_FEATURES 


        self.unet_1d = UNet1D(
            in_channels=sum(in_channels),  
            base_channels=16
        )

        self.number_imu_blocks = number_imu_blocks


        self.block_indexes = [0] + [sum(in_channels[:i + 1]) for i in range(len(in_channels))]

        
        self.thm_embed = nn.Linear(5, THM_out_channels * 5)

        
        self.cnn_branches = nn.ModuleList(
            [
                # IMU branches
                CNN1D_LSTM_Branch(
                    input_channels=in_channels[i],
                    output_channels=branch_features,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden
                )
                for i in range(len(in_channels))
            ]
            +
            [
                # TOF branch
                CNN1D_LSTM_Branch(
                    input_channels=ToF_out_channels * 5,
                    output_channels=branch_features,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden
                )
            ]
            +
            [
                # THM branch
                CNN1D_LSTM_Branch(
                    input_channels=THM_out_channels * 5,
                    output_channels=branch_features,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden
                )
            ]
        )

        # Bloques 2D ToF/THM
        self.tof_block = nn.ModuleList([
            ToF_2D_Block(output_channels=ToF_out_channels, kernel_size=ToF_kernel_size)
            for _ in range(5)
        ])

        self.thm_block = nn.ModuleList([
            ToF_2D_Block(output_channels=THM_out_channels, kernel_size=THM_kernel_size)
            for _ in range(5)
        ])

        # ====== Heads ======
        # mlp_all: in_features = 1024, out_features = 18
        n_channels_all = cnn1d_channels[-1] * (len(in_channels) + 2) * 2
        self.mlp_all = MLP(n_channels_all, mlp_dropout, num_classes)

        # ensemble_all: 18 + 4*128 → 18
        self.ensemble_all = nn.Linear(
            num_classes + (len(in_channels) + 2) * branch_features,
            num_classes
        )

        # mlp_imu: 256 → 18
        n_channels_imu = cnn1d_channels[-1] * number_imu_blocks * 2
        self.mlp_imu = MLP(n_channels_imu, mlp_dropout, num_classes)

        # ensemble_imu: 18 + 1*128 → 18
        self.ensemble_imu = nn.Linear(
            num_classes + number_imu_blocks * branch_features,
            num_classes
        )

    def forward(self, x, x_tof, x_thm):

        list_of_x = []
        list_outs = []

        
        
        gesture_mask = F.avg_pool1d(x[:, :1], kernel_size=5, stride=1, padding=2)
        gesture_mask = torch.tanh(gesture_mask)
# (B,1,T)

        if gesture_mask.dim() == 2:
            gesture_mask = gesture_mask.unsqueeze(1)

        
        
        T = x.shape[2]
        if gesture_mask.shape[2] > T:
            gesture_mask = gesture_mask[:, :, :T]
        elif gesture_mask.shape[2] < T:
            pad = T - gesture_mask.shape[2]
            gesture_mask = F.pad(gesture_mask, (0, pad))

        

        
        for i in range(len(self.block_indexes) - 1):
            x_block = x[:, self.block_indexes[i]:self.block_indexes[i + 1], :]  # (B,3,T)
            feat, out = self.cnn_branches[i](x_block, gesture_mask)
            list_of_x.append(feat)
            list_outs.append(out)

    
        B = x.shape[0]
        tof_feats = []
        for i in range(5):
            t = x_tof[:, :, i * 64:(i + 1) * 64]   # (B, T, 64)
            t = t.reshape(-1, 1, 8, 8)
            t = self.tof_block[i](t)
            t = t.reshape(B, -1, t.shape[1]).transpose(1, 2)
            tof_feats.append(t)

        tof_all = torch.cat(tof_feats, dim=1)
        feat, out = self.cnn_branches[len(self.block_indexes) - 1](tof_all, gesture_mask)
        list_of_x.append(feat)
        list_outs.append(out)

        thm = self.thm_embed(x_thm).transpose(1, 2)
        feat, out = self.cnn_branches[len(self.block_indexes)](thm, gesture_mask)
        list_of_x.append(feat)
        list_outs.append(out)

 
        x_all = torch.cat(list_of_x, dim=1)
        out_all = self.mlp_all(x_all)

        out_all = self.ensemble_all(
            torch.cat([out_all] + list_outs, dim=1)
        )

        return out_all
