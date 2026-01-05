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
Simple Conv1D block without LSTM (for conv stack)
"""
class Conv1DReLUBN(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride=1, groups=1):
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

    def forward(self, x):
        """
        x : (batch, channels, seq_len)
        """
        return self.conv_block(x)


"""
The Conv1D with LSTM for The IMU, the Tof, The thm (DEPRECATED - kept for backward compatibility)
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

        # Upsample and match dimensions
        up2_out = self.up2(m)
        if up2_out.size(2) != e2.size(2):
            up2_out = F.interpolate(up2_out, size=e2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([up2_out, e2], dim=1))

        up1_out = self.up1(d2)
        if up1_out.size(2) != e1.size(2):
            up1_out = F.interpolate(up1_out, size=e1.size(2), mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([up1_out, e1], dim=1))

        return self.out(d1)  



class CNN1D_LSTM_Branch(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 initial_channels_per_feature,
                 CNN1D_channels_size,
                 CNN1D_kernel_size,
                 mlp_dropout,
                 lstm_hidden=128,
                 num_classes=18,
                 use_gru=False):
        super().__init__()

        # Depthwise-Separable Conv:
        # Depthwise temporal conv (groups=in_ch, no channel expansion)
        # Pointwise 1x1 conv to mix channels and expand modestly
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,  # No expansion
                kernel_size=CNN1D_kernel_size,
                stride=1,
                padding='same',
                groups=input_channels  # Depthwise
            ),
            nn.BatchNorm1d(input_channels),
            nn.ReLU()
        )

        # Pointwise expansion (1x1 conv to mix and expand channels modestly)
        first_expansion = CNN1D_channels_size[0] if len(CNN1D_channels_size) > 0 else 64
        self.pointwise = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=first_expansion,  # Modest expansion
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(first_expansion),
            nn.ReLU()
        )

        # First MaxPool after depthwise+pointwise (replaces pool after first conv block)
        self.first_pool = nn.MaxPool1d(kernel_size=3, stride=3) if len(CNN1D_channels_size) > 1 else nn.Identity()

        # Stack of remaining conv blocks (no LSTM inside)
        CNN1D_layers = []
        for i in range(len(CNN1D_channels_size)):
            if i == 0:
                # Skip first block - already handled by depthwise+pointwise
                continue

            in_ch = CNN1D_channels_size[i - 1]
            out_ch = CNN1D_channels_size[i]

            CNN1D_layers.append(
                Conv1DReLUBN(
                    in_channels=in_ch,
                    output_channels=out_ch,
                    kernel_size=CNN1D_kernel_size,
                    stride=1,
                    groups=1
                )
            )

            # Add pool after each block except the last
            if i < len(CNN1D_channels_size) - 1:
                CNN1D_layers.append(nn.MaxPool1d(kernel_size=3, stride=3))

        self.CNN1D_layers = nn.Sequential(*CNN1D_layers)

        # One shared BiLSTM/GRU after the conv stack
        self.use_gru = use_gru
        if use_gru:
            self.rnn = nn.GRU(
                input_size=CNN1D_channels_size[-1],
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=CNN1D_channels_size[-1],
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        # Project RNN output back to conv channel dimension
        self.rnn_projection = nn.Linear(2 * lstm_hidden, CNN1D_channels_size[-1])

        # Feature MLP (outputs feature vector)
        self.mlp = MLP(
            number_channels=CNN1D_channels_size[-1] * 2,
            mlp_dropout=mlp_dropout,
            out_channels=output_channels  # 128
        )

        # Classification head (outputs class predictions)
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x, gesture_mask_logits=None):
        """
        x: (B, C, T)
        gesture_mask_logits: (B, 1, T) raw logits from UNet, or None
        Returns: features (B, 128), class_predictions (B, num_classes)
        """
        # Depthwise-separable conv: depthwise → pointwise
        x = self.depthwise(x)   # (B, C, T) - no channel expansion
        x = self.pointwise(x)   # (B, first_expansion, T) - modest expansion
        x = self.first_pool(x)  # (B, first_expansion, T/3) - downsample

        # Remaining conv stack (no LSTM inside)
        x = self.CNN1D_layers(x)    # (B, C_last, T')

        # Apply one shared RNN after conv stack
        # x: (B, C, T) → permute to (B, T, C) for RNN
        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.rnn(x)  # (B, T, 2*lstm_hidden)
        x = self.rnn_projection(x)  # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)

        # Soft pooling with gesture mask
        if gesture_mask_logits is not None:
            if gesture_mask_logits.dim() == 2:
                gesture_mask_logits = gesture_mask_logits.unsqueeze(1)

            if gesture_mask_logits.size(2) != x.size(2):
                T = min(gesture_mask_logits.size(2), x.size(2))
                x = x[:, :, :T]
                gesture_mask_logits = gesture_mask_logits[:, :, :T]

            # Soft pooling with sigmoid weights
            # w_t = sigmoid(mask_logit_t) gives gesture probability in [0,1]
            w_gesture = torch.sigmoid(gesture_mask_logits)  # (B, 1, T)
            w_non_gesture = 1 - w_gesture  # (B, 1, T)

            # gesture_pool = sum(w_t * f_t) / sum(w_t)
            x_gesture = (x * w_gesture).sum(dim=2) / w_gesture.sum(dim=2).clamp(min=1e-6)
            # non_gesture_pool = sum((1 - w_t) * f_t) / sum(1 - w_t)
            x_non_gesture = (x * w_non_gesture).sum(dim=2) / w_non_gesture.sum(dim=2).clamp(min=1e-6)

            x = torch.cat([x_gesture, x_non_gesture], dim=1)
        else:
            # Fallback to standard pooling if no mask
            x_mean = x.mean(dim=2)
            x_max = x.max(dim=2).values
            x = torch.cat([x_mean, x_max], dim=1)

        features = self.mlp(x)  # (B, 128)
        class_pred = self.classifier(features)  # (B, num_classes)
        return features, class_pred


NUM_CLASSES = 18
BRANCH_FEATURES = 128


class MultiBranchClassifier(nn.Module):
    def __init__(
        self,
        number_imu_blocks,
        in_channels,
        out_channels,                 # num_classes (18)
        initial_channels_per_feature,
        cnn1d_channels,
        cnn1d_kernel_size,
        ToF_out_channels,
        ToF_kernel_size,
        THM_out_channels,
        THM_kernel_size,
        mlp_dropout,
        unet_in_channels,
        lstm_hidden=256,
        use_gru=False  # True for GRU, False for LSTM
    ):
        super().__init__()

        num_classes = out_channels        # 18
        branch_features = BRANCH_FEATURES # 128


        self.unet_1d = UNet1D(
            in_channels=sum(in_channels),  # 6
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
                    lstm_hidden=lstm_hidden,
                    num_classes=num_classes,
                    use_gru=use_gru
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
                    lstm_hidden=lstm_hidden,
                    num_classes=num_classes,
                    use_gru=use_gru
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
                    lstm_hidden=lstm_hidden,
                    num_classes=num_classes,
                    use_gru=use_gru
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

        # Ensemble Heads 
        # For multi-output ensemble, we directly map concatenated features to classes
        # Total branches = len(in_channels) IMU + 1 ToF + 1 THM
        num_total_branches = len(in_channels) + 2

        # ensemble_all: concat all branch features (each 128-dim) → num_classes
        self.ensemble_all = nn.Linear(
            num_total_branches * branch_features,  # 6 * 128 = 768
            num_classes
        )

        # ensemble_imu: concat IMU branch features only → num_classes
        self.ensemble_imu = nn.Linear(
            number_imu_blocks * branch_features,  # 4 * 128 = 512
            num_classes
        )

    def forward(self, x, x_tof, x_thm, gesture_segment=None):
        """
        Args:
            x: Input features (B, C, T)
            x_tof: ToF features
            x_thm: Thermal features
            gesture_segment: Ground truth gesture segmentation (B, 1, T) or (B, T)
                            Values: 1=gesture, 0=non-gesture (changed from [-1, 1])
                            Only provided during training for U-Net supervision

        Returns:
            all_outputs: Stacked predictions from all heads
            gesture_mask_logits: Raw logits for gesture mask (for BCEWithLogitsLoss)
        """
        list_of_x = []
        list_outs = []

        # Use U-Net to predict gesture mask logits (no activation)
        # Output is raw logits: sigmoid(logits) -> [0,1] where 1=gesture, 0=non-gesture
        gesture_mask_logits = self.unet_1d(x)  # (B, 1, T)

        if gesture_mask_logits.dim() == 2:
            gesture_mask_logits = gesture_mask_logits.unsqueeze(1)



        T = x.shape[2]
        if gesture_mask_logits.shape[2] > T:
            gesture_mask_logits = gesture_mask_logits[:, :, :T]
        elif gesture_mask_logits.shape[2] < T:
            pad = T - gesture_mask_logits.shape[2]
            gesture_mask_logits = F.pad(gesture_mask_logits, (0, pad))




        for i in range(len(self.block_indexes) - 1):
            x_block = x[:, self.block_indexes[i]:self.block_indexes[i + 1], :]  # (B,3,T)
            feat, out = self.cnn_branches[i](x_block, gesture_mask_logits)
            list_of_x.append(feat)
            list_outs.append(out)

        # ===== TOF =====
        B = x.shape[0]
        tof_feats = []
        for i in range(5):
            t = x_tof[:, :, i * 64:(i + 1) * 64]   # (B, T, 64)
            t = t.reshape(-1, 1, 8, 8)
            t = self.tof_block[i](t)
            t = t.reshape(B, -1, t.shape[1]).transpose(1, 2)
            tof_feats.append(t)

        tof_all = torch.cat(tof_feats, dim=1)
        feat, out = self.cnn_branches[len(self.block_indexes) - 1](tof_all, gesture_mask_logits)
        list_of_x.append(feat)
        list_outs.append(out)

        # THM
        thm = self.thm_embed(x_thm).transpose(1, 2)
        feat, out = self.cnn_branches[len(self.block_indexes)](thm, gesture_mask_logits)
        list_of_x.append(feat)
        list_outs.append(out)

        # MULTI-OUTPUT ENSEMBLE
        # return multiple prediction heads

        # list_of_x contains features (128-dim each)
        # list_outs contains class predictions (18-dim each)

        # All-branches ensemble (use concatenated features from all branches)
        x_all_features = torch.cat(list_of_x, dim=1)  # Concatenate MLP features
        out_all = self.ensemble_all(
            torch.cat([x_all_features], dim=1)  # Just use features
        )

        # IMU-only ensemble (use concatenated features from IMU branches only)
        x_imu_features = torch.cat(list_of_x[:self.number_imu_blocks], dim=1)
        out_imu = self.ensemble_imu(
            torch.cat([x_imu_features], dim=1)
        )

        # Individual branch outputs (already computed in list_outs)
        # Stack all outputs: [out_all, out_imu, branch_0, branch_1, ..., branch_N]
        all_outputs = [out_all, out_imu] + list_outs

        # Return stacked tensor: (batch_size, num_heads, num_classes)
        # num_heads = 2 (ensembles) + len(list_outs) (individual branches)
        predictions = torch.stack(all_outputs, dim=1)

        # Return predictions and raw logits (for BCEWithLogitsLoss)
        return predictions, gesture_mask_logits

    def compute_unet_loss(self, gesture_mask_logits, gesture_segment_true, mask_valid=None):
        """
        Compute supervision loss for the U-Net gesture segmentation using BCEWithLogitsLoss.

        Args:
            gesture_mask_logits: Predicted gesture mask logits (B, 1, T), raw values (no sigmoid)
            gesture_segment_true: Ground truth gesture segment (B, 1, T) or (B, T)
                                 Values: 1=gesture, 0=non-gesture (changed from [-1,1])
            mask_valid: Optional mask indicating valid (non-padding) positions (B, T)

        Returns:
            loss: BCEWithLogitsLoss between predicted logits and true gesture segmentation
        """
        # Ensure same shape
        if gesture_segment_true.dim() == 2:
            gesture_segment_true = gesture_segment_true.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # Match sequence length
        T = min(gesture_mask_logits.size(2), gesture_segment_true.size(2))
        gesture_mask_logits = gesture_mask_logits[:, :, :T]
        gesture_segment_true = gesture_segment_true[:, :, :T]

        # Convert to float for BCE loss
        gesture_segment_true = gesture_segment_true.float()

        if mask_valid is not None:
            # Only compute loss on valid (non-padding) positions
            if mask_valid.dim() == 2:
                mask_valid = mask_valid.unsqueeze(1)  # (B, T) -> (B, 1, T)
            mask_valid = mask_valid[:, :, :T]

            # BCEWithLogitsLoss with reduction='none', then manually mask
            bce_loss = F.binary_cross_entropy_with_logits(
                gesture_mask_logits,
                gesture_segment_true,
                reduction='none'
            )
            loss = (bce_loss * mask_valid).sum() / mask_valid.sum().clamp(min=1)
        else:
            # Standard BCEWithLogitsLoss
            loss = F.binary_cross_entropy_with_logits(
                gesture_mask_logits,
                gesture_segment_true
            )

        return loss