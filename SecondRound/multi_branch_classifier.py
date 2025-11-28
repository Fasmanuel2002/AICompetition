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
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=output_channels,
                      kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        def forward(self, x):
            return self.layers(x)
        

#Combine Everything for the ToF block   
class ToF_2D_Block(nn.Module):
    def __init__(self, output_channels, kernel_size):
        super().__init__()
        
        self.layers = nn.Sequential(
            Conv2DReLUBN(
               in_channels=1, 
               output_channels=output_channels, 
               kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=2),
            
            Conv2DReLUBN(
                in_channels=output_channels, 
                output_channels=output_channels, 
                kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=2),
            
            Conv2DReLUBN(
                in_channels=output_channels, 
                output_channels=output_channels,
                kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=2),
        )
        
    def forward(self, x):
        return self.layers(x)
    

"""
The Conv1D with LSTM for The IMU, the Tof, The thm
""" 
class Conv1DReLUBN_LSTM(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride = 1, groups = 1, lstm_hidden=128):
        super().__init__()
                        
        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2
        
        #CNN block
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU())
        #LSTM block
        self.lstm_block = nn.LSTM(input_size=output_channels, 
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
    

"""
Multi-Layer Perceptron: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
"""
#Fusion the information of different sensor
#Concatenate
#Introduction no linearity
#Prepare the last signal

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
    

class U_NET1D(nn.Module):
    def __init__(self, in_channles, base_channels=32):
        super().__init__()
        
        #Encoder, convolutional layers with the Conv1d
        self.encoder1 = self.conv_block(in_channels=in_channles, 
                                        out_channels=base_channels)
        self.pool1 = nn.MaxPool1d(2)
        
        self.encoder2 = self.conv_block(in_channels=base_channels,
                                        out_channels=base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.encoder3 = self.conv_block(in_channels=base_channels*2,
                                        out_channels=base_channels*4)
        
        #Decoder
        self.upcoder2 = nn.ConvTranspose1d(in_channels=base_channels*4,
                                           out_channels=base_channels*2,
                                           kernel_size=2,
                                           stride=2)
        self.decoder2 = self.conv_block(in_channels=base_channels*4,
                                        out_channels=base_channels*2)
        
        self.upcoder1 = nn.ConvTranspose1d(in_channels=base_channels*2,
                                           out_channels=base_channels,
                                           kernel_size=2,
                                           stride=2)
        self.decoder1 = self.conv_block(in_channels=base_channels*2,
                                        out_channels=base_channels)
        
        #Segmentation Mask
        self.out = nn.Conv1d(base_channels, out_channels=1, kernel_size=1)
        
        
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1
            ),
            nn.ReLU()
        )
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))

        d2 = self.upcoder2(x3)
        # Match dimensions before concatenation
        if d2.shape[2] != x2.shape[2]:
            d2 = F.interpolate(d2, size=x2.shape[2], mode='linear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x2], dim=1))

        d1 = self.upcoder1(d2)
        # Match dimensions before concatenation
        if d1.shape[2] != x1.shape[2]:
            d1 = F.interpolate(d1, size=x1.shape[2], mode='linear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x1], dim=1))

        return self.out(d1)
    

"""
Fusion everypart of the layers that are the THM and the IMU
"""
        
class CNN1D_LSTM_Branch(nn.Module):
    def __init__(self,
                input_channels, 
                output_channels,
                initial_channels_per_feature,
                CNN1D_channels_size,
                CNN1D_kernel_size,
                mlp_dropout,
                lstm_hidden = 128):
        super().__init__()
        
        self.mlp_dropout = mlp_dropout
        
        self.inital_layers = Conv1DReLUBN_LSTM(
            in_channels=input_channels,
            output_channels = input_channels * initial_channels_per_feature,
            kernel_size=CNN1D_kernel_size,
            stride=1,
            groups=input_channels,
            lstm_hidden=lstm_hidden
        )
        CNN1D_layers = nn.ModuleList()
        
        for i in range(len(CNN1D_channels_size)):
            if i == 0:
                in_channels = input_channels * initial_channels_per_feature
            else:
                in_channels = CNN1D_channels_size[i - 1]
            output_channels = CNN1D_channels_size[i]
            
            CNN1D_layers.append(
                Conv1DReLUBN_LSTM(
                    in_channels=in_channels,
                    output_channels=output_channels,
                    kernel_size=CNN1D_kernel_size,
                    stride=1,
                    groups=1,
                    lstm_hidden=lstm_hidden
                )
            ) 
            
            if i < len(CNN1D_channels_size) - 1:
                CNN1D_layers.append(
                    nn.MaxPool1d(kernel_size=3, stride=3)
                )
        self.CNN1D_layers = nn.Sequential(*CNN1D_layers)
        
        #original final_channels = CNN1D_channels_size[-1] * 2
        final_channels = CNN1D_channels_size[-1] * 2
        
        self.mlp = MLP(
            number_channels=final_channels,
            mlp_dropout=mlp_dropout,
            out_channels=output_channels
        )
        
    def forward(self, x, gesture_segment):
        x = self.inital_layers(x)
        x = self.CNN1D_layers(x)

        # Resize gesture_segment to match x's temporal dimension
        if gesture_segment.shape[2] != x.shape[2]:
            gesture_segment = F.interpolate(gesture_segment, size=x.shape[2], mode='linear', align_corners=False)

        #Gestures that are part of the target gestures
        x1_gestures = (x * (gesture_segment > 0)).sum(dim=2) / (gesture_segment > 0).sum(dim=2).clamp(min=1)
        #Non gestures that are part of the non target gestures
        x2_non_gestures = (x * (gesture_segment < 0)).sum(dim=2) / (gesture_segment < 0).sum(dim=2).clamp(min=1)

        x = torch.cat([x1_gestures, x2_non_gestures], dim=1)
        out = self.mlp(x)

        return x, out
    

"""
Fusion Everything in the model, is the One you need to call
"""  
        
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
        lstm_hidden=128
    ):
        super().__init__()

        # U-Net for gesture segmentation
        self.unet_1d = U_NET1D(in_channles=sum(in_channels))

        self.number_imu_blocks = number_imu_blocks

        # in_channels=[6,5] → [0,6,11]
        self.block_indexes = [0] + [sum(in_channels[:i+1]) for i in range(len(in_channels))]

        
        
        self.thm_embed = nn.Linear(5, THM_out_channels * 5)

        
        self.cnn_branches = nn.ModuleList(
            [
                # IMU branches
                CNN1D_LSTM_Branch(
                    input_channels=in_channels[i],
                    output_channels=out_channels,
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
                    output_channels=out_channels,
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
                    output_channels=out_channels,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden
                )
            ]
        )

        
        self.tof_block = nn.ModuleList([
            ToF_2D_Block(output_channels=ToF_out_channels, kernel_size=ToF_kernel_size)
            for _ in range(5)
        ])

        
        self.thm_block = nn.ModuleList([
            ToF_2D_Block(output_channels=THM_out_channels, kernel_size=THM_kernel_size)
            for _ in range(5)
        ])

        
        n_channels = cnn1d_channels[-1] * (len(in_channels) + 2) * 2
        self.mlp_all = MLP(n_channels, mlp_dropout, out_channels)
        self.ensemble_all = nn.Linear(
            out_channels + 4 * 128,
            out_channels
        )

        n_channels = cnn1d_channels[-1] * number_imu_blocks * 2
        self.mlp_imu = MLP(n_channels, mlp_dropout, out_channels)
        self.ensemble_imu = nn.Linear(
            out_channels + number_imu_blocks * 128,
            out_channels
        )

    def forward(self, x, x_tof, x_thm):

            list_of_x = []
            list_out_puts = []
    
            
            gesture_segment = torch.sigmoid(self.unet_1d(x))
    
            
            for i in range(len(self.block_indexes) - 1):
                x_block = x[:, self.block_indexes[i] : self.block_indexes[i + 1], :]
                x_block, out = self.cnn_branches[i](x_block, gesture_segment)
                list_of_x.append(x_block)
                list_out_puts.append(out)
    
            
            list_x_tof = []
            for i in range(5):
                x_block = x_tof[:, :, i*64:(i+1)*64].reshape(-1,1,8,8)
                out = self.tof_block[i](x_block)
                out = out.reshape(x.shape[0], -1, out.shape[1]).transpose(1, 2)
                list_x_tof.append(out)
    
            x_tof_processed = torch.cat(list_x_tof, dim=1)
            x_tof_processed, out_tof = self.cnn_branches[len(self.block_indexes)-1](
                x_tof_processed, gesture_segment
            )
            list_of_x.append(x_tof_processed)
            list_out_puts.append(out_tof)
    
            
            x_thm_embed = self.thm_embed(x_thm)   # [batch, time, THM_out*5]
            x_thm_transposed = x_thm_embed.transpose(1, 2)
    
            x_thm_processed, out_thm = self.cnn_branches[len(self.block_indexes)](
                x_thm_transposed, gesture_segment
            )
            list_of_x.append(x_thm_processed)
            list_out_puts.append(out_thm)
    
            # --- FINAL FUSION ---
            x_all = torch.cat(list_of_x, dim=1)
            out_all = self.mlp_all(x_all)
            out_all = self.ensemble_all(torch.cat([out_all] + list_out_puts, dim=1))
    
            
    
            return out_all