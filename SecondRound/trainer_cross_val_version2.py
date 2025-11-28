import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Conv2D with ReLU + BatchNorm for the ToF / THM 2D blocks
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


"""
2D CNN block for ToF / THM (stacked Conv2D+BN+ReLU + MaxPool)
"""
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
Conv1D + LSTM block used inside the 1D branches (IMU / ToF / THM)
"""
class Conv1DReLUBN_LSTM(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_size,
        stride=1,
        groups=1,
        lstm_hidden=128
    ):
        super().__init__()

        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2

        # CNN block
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
        )

        # LSTM block (bidirectional)
        self.lstm_block = nn.LSTM(
            input_size=output_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Projection back to output_channels
        self.projection = nn.Linear(2 * lstm_hidden, output_channels)

    def forward(self, x):
        """
        x: (batch, channels, seq_len)
        """
        # CNN forward
        x = self.conv_block(x)

        # Prepare for LSTM: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # LSTM forward
        x, _ = self.lstm_block(x)

        # Linear projection
        x = self.projection(x)

        # Back to (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        return x


"""
Multi-Layer Perceptron: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
"""
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

            nn.Linear(number_channels // 4, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


"""
1D U-Net for gesture segmentation over the concatenated sensor channels
"""
class U_NET1D(nn.Module):
    def __init__(self, in_channles, base_channels=32):
        super().__init__()

        # Encoder
        self.encoder1 = self.conv_block(
            in_channels=in_channles,
            out_channels=base_channels
        )
        self.pool1 = nn.MaxPool1d(2)

        self.encoder2 = self.conv_block(
            in_channels=base_channels,
            out_channels=base_channels * 2
        )
        self.pool2 = nn.MaxPool1d(2)

        self.encoder3 = self.conv_block(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4
        )

        # Decoder
        self.upcoder2 = nn.ConvTranspose1d(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = self.conv_block(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2
        )

        self.upcoder1 = nn.ConvTranspose1d(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = self.conv_block(
            in_channels=base_channels * 2,
            out_channels=base_channels
        )

        # Output segmentation mask (1 channel)
        self.out = nn.Conv1d(
            in_channels=base_channels,
            out_channels=1,
            kernel_size=1
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))

        # Decoder
        d2 = self.upcoder2(x3)
        if d2.shape[2] != x2.shape[2]:
            d2 = F.interpolate(d2, size=x2.shape[2], mode="linear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x2], dim=1))

        d1 = self.upcoder1(d2)
        if d1.shape[2] != x1.shape[2]:
            d1 = F.interpolate(d1, size=x1.shape[2], mode="linear", align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x1], dim=1))

        return self.out(d1)


"""
Branch used for IMU / ToF / THM 1D signals:
  - Initial Conv1D+BN+ReLU+LSTM
  - Stack of Conv1D+BN+ReLU+LSTM (+ MaxPool)
  - Pooling over gesture / non-gesture segments
  - Final MLP to project to out_channels
"""
class CNN1D_LSTM_Branch(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        initial_channels_per_feature,
        CNN1D_channels_size,
        CNN1D_kernel_size,
        mlp_dropout,
        lstm_hidden=128,
    ):
        super().__init__()

        self.mlp_dropout = mlp_dropout

        # First depthwise Conv1D+LSTM layer
        self.inital_layers = Conv1DReLUBN_LSTM(
            in_channels=input_channels,
            output_channels=input_channels * initial_channels_per_feature,
            kernel_size=CNN1D_kernel_size,
            stride=1,
            groups=input_channels,
            lstm_hidden=lstm_hidden,
        )

        CNN1D_layers = nn.ModuleList()
        for i in range(len(CNN1D_channels_size)):
            if i == 0:
                in_channels = input_channels * initial_channels_per_feature
            else:
                in_channels = CNN1D_channels_size[i - 1]

            out_ch = CNN1D_channels_size[i]

            CNN1D_layers.append(
                Conv1DReLUBN_LSTM(
                    in_channels=in_channels,
                    output_channels=out_ch,
                    kernel_size=CNN1D_kernel_size,
                    stride=1,
                    groups=1,
                    lstm_hidden=lstm_hidden,
                )
            )

            if i < len(CNN1D_channels_size) - 1:
                CNN1D_layers.append(
                    nn.MaxPool1d(kernel_size=3, stride=3)
                )

        self.CNN1D_layers = nn.Sequential(*CNN1D_layers)

        # final feature dim after segment pooling:
        # x1_gestures:  [B, last_channels]
        # x2_non_gests: [B, last_channels]
        # cat → [B, last_channels * 2]
        final_channels = CNN1D_channels_size[-1] * 2

        self.mlp = MLP(
            number_channels=final_channels,
            mlp_dropout=mlp_dropout,
            out_channels=output_channels,
        )

    def forward(self, x, gesture_segment):
        # x: [B, C, T]
        x = self.inital_layers(x)
        x = self.CNN1D_layers(x)  # [B, C_last, T']

        # Resize gesture_segment to match temporal dimension
        if gesture_segment.shape[2] != x.shape[2]:
            gesture_segment = F.interpolate(
                gesture_segment,
                size=x.shape[2],
                mode="linear",
                align_corners=False,
            )

        # gesture_segment > 0 for "gesture", < 0 for "non-gesture"
        mask_gesture = (gesture_segment > 0)
        mask_nongesture = (gesture_segment < 0)

        x1_gestures = (x * mask_gesture).sum(dim=2) / mask_gesture.sum(dim=2).clamp(min=1)
        x2_non_gestures = (x * mask_nongesture).sum(dim=2) / mask_nongesture.sum(dim=2).clamp(min=1)

        # [B, 2 * C_last]
        x_cat = torch.cat([x1_gestures, x2_non_gestures], dim=1)

        # Project to out_channels
        out = self.mlp(x_cat)

        return x_cat, out


"""
Multi-branch classifier that fuses:
  - IMU branches
  - ToF 2D processed + 1D branch
  - THM 2D processed + 1D branch
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
        lstm_hidden=128,
    ):
        super().__init__()

        # U-Net for gesture segmentation (over all IMU channels concatenated)
        self.unet_1d = U_NET1D(in_channles=sum(in_channels))

        self.number_imu_blocks = number_imu_blocks
        self.in_channels = in_channels

        # Example: in_channels = [6, 5] → block_indexes = [0, 6, 11]
        self.block_indexes = [0] + [sum(in_channels[: i + 1]) for i in range(len(in_channels))]

        # THM embedding from 5 features -> THM_out_channels * 5
        self.thm_embed = nn.Linear(5, THM_out_channels * 5)

        # 1D branches (IMU + ToF + THM)
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
                    lstm_hidden=lstm_hidden,
                )
                for i in range(len(in_channels))
            ]
            + [
                # ToF branch
                CNN1D_LSTM_Branch(
                    input_channels=ToF_out_channels * 5,
                    output_channels=out_channels,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden,
                )
            ]
            + [
                # THM branch
                CNN1D_LSTM_Branch(
                    input_channels=THM_out_channels * 5,
                    output_channels=out_channels,
                    initial_channels_per_feature=initial_channels_per_feature,
                    CNN1D_channels_size=cnn1d_channels,
                    CNN1D_kernel_size=cnn1d_kernel_size,
                    mlp_dropout=mlp_dropout,
                    lstm_hidden=lstm_hidden,
                )
            ]
        )

        # Número total de ramas (IMU + ToF + THM)
        self.num_branches = len(self.cnn_branches)

        # 2D ToF blocks (one per "view")
        self.tof_block = nn.ModuleList(
            [
                ToF_2D_Block(
                    output_channels=ToF_out_channels,
                    kernel_size=ToF_kernel_size
                )
                for _ in range(5)
            ]
        )

        # 2D THM blocks (one per "view")
        self.thm_block = nn.ModuleList(
            [
                ToF_2D_Block(
                    output_channels=THM_out_channels,
                    kernel_size=THM_kernel_size
                )
                for _ in range(5)
            ]
        )

        # Fusion MLP over concatenated x from all branches
        # Cada rama devuelve x_cat de tamaño (2 * cnn1d_channels[-1])
        # → x_all: [B, num_branches * 2 * cnn1d_channels[-1]]
        n_channels_all = cnn1d_channels[-1] * self.num_branches * 2
        self.mlp_all = MLP(
            number_channels=n_channels_all,
            mlp_dropout=mlp_dropout,
            out_channels=out_channels,
        )

        # Ensemble final:
        # fusion = cat([out_all] + list_out_puts, dim=1)
        #   - out_all: [B, out_channels]
        #   - list_out_puts: num_branches elementos, cada uno [B, out_channels]
        # → fusion_dim = out_channels * (num_branches + 1)
        self.ensemble_all = nn.Linear(
            out_channels * (self.num_branches + 1),
            out_channels,
        )

        # Cabeza IMU (no usada en forward actual, la dejamos definida por si se usa más adelante)
        n_channels_imu = cnn1d_channels[-1] * number_imu_blocks * 2
        self.mlp_imu = MLP(
            number_channels=n_channels_imu,
            mlp_dropout=mlp_dropout,
            out_channels=out_channels,
        )
        # Si algún día se usa, habrá que definir bien la concatenación aquí.
        self.ensemble_imu = nn.Linear(
            out_channels + number_imu_blocks * 128,
            out_channels,
        )

    def forward(self, x, x_tof, x_thm):
        """
        x:      [B, sum(in_channels), T]       (IMU concatenado)
        x_tof:  [B, 1, 64*5] o similar (para 5 vistas de 8x8)
        x_thm:  [B, T, 5]
        """
        list_of_x = []
        list_out_puts = []

        # Gesture segmentation mask
        gesture_segment = torch.sigmoid(self.unet_1d(x))  # [B, 1, T_seg]

        # --- IMU branches ---
        for i in range(len(self.block_indexes) - 1):
            start = self.block_indexes[i]
            end = self.block_indexes[i + 1]
            x_block = x[:, start:end, :]  # [B, C_block, T]

            x_block_feat, out_branch = self.cnn_branches[i](x_block, gesture_segment)
            list_of_x.append(x_block_feat)
            list_out_puts.append(out_branch)

        # --- ToF 2D + 1D branch ---
        list_x_tof = []
        for i in range(5):
            # Extraer cada "vista" 8x8 a partir de x_tof
            x_block = x_tof[:, :, i * 64:(i + 1) * 64].reshape(-1, 1, 8, 8)
            out = self.tof_block[i](x_block)  # [B*?, ToF_out_channels, H', W']

            # Convertir a [B, C, T] para la rama 1D
            out = out.reshape(x.shape[0], -1, out.shape[1]).transpose(1, 2)
            list_x_tof.append(out)

        x_tof_processed = torch.cat(list_x_tof, dim=1)  # [B, ToF_out_channels * 5, T_toF]
        tof_branch_idx = len(self.block_indexes) - 1

        x_tof_feat, out_tof = self.cnn_branches[tof_branch_idx](
            x_tof_processed,
            gesture_segment
        )
        list_of_x.append(x_tof_feat)
        list_out_puts.append(out_tof)

        # --- THM 2D + 1D branch ---
        x_thm_embed = self.thm_embed(x_thm)  # [B, T, THM_out_channels * 5]
        x_thm_transposed = x_thm_embed.transpose(1, 2)  # [B, THM_out_channels * 5, T]

        thm_branch_idx = len(self.block_indexes)
        x_thm_feat, out_thm = self.cnn_branches[thm_branch_idx](
            x_thm_transposed,
            gesture_segment
        )
        list_of_x.append(x_thm_feat)
        list_out_puts.append(out_thm)

        # --- FINAL FUSION ---
        # Concatenar todas las features x_cat de cada rama
        x_all = torch.cat(list_of_x, dim=1)     # [B, num_branches * 2 * cnn1d_channels[-1]]
        out_all = self.mlp_all(x_all)           # [B, out_channels]

        # Concatenar out_all + outputs de cada rama
        fusion = torch.cat([out_all] + list_out_puts, dim=1)
        # fusion.shape[1] = out_channels * (num_branches + 1)

        out_all = self.ensemble_all(fusion)     # [B, out_channels]

        return out_all
