import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mask(lengths , max_len: int) -> float:
    #Create a mask from the sequence
    batch_size = lengths.size(0)
    
    # Generate position indices [0, 1, 2, ..., max_len-1]
    # Expand to (batch_size, max_len) so each row corresponds to one sequence
    positions = torch.arange(max_len, device = lengths.device).expand(batch_size, max_len)
    
    mask =  positions < lengths.unsqueeze(1)
    
    return mask.float() #(batch_size, max_len)



"""
The Squeeze-and-Excitation (SE) block 
applies channel-wise attention to help a 
CNN focus on the most relevant features 
for each input. It does this by learning
importance weights for each channel and 
reweighting them, amplifying informative 
features while suppressing less useful ones.
"""
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        #The part of the attention layers that learns about the weights of the channel
        self.excitation = torch.nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features= channels // reduction, out_features=channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x , mask):
            #Compute the Avarage pooling
            mask = mask.unsqueeze(1) ## (batch, 1, seq_len)
            masked_x = x * mask # Zero Out padding positions
            seq_lenghts = mask.sum(dim=-1, keepdim=True) # # (batch, 1, 1)

            y = masked_x.sum(dim=-1, keepdim= True) / (seq_lenghts + 1e-8) #(batch, channels, 1)
            y = self.excitation(y.squeeze(-1)).unsqueeze(-1)  # (batch, channels, 1)
            
            return x * y.expand_as(x)
        
        
class Residual_SqueezeExcitation_CNN_LSTM_block(nn.Module):
    """
    Hybrid block combining:
    - 3× Conv1D layers with BatchNorm and ReLU
    - Squeeze-and-Excitation channel attention
    - Residual connection with shortcut
    - BiLSTM for temporal modeling
    """
    def __init__(self, in_channels: int, out_channels: int , kernel_size : int, dropout = 0.3, weight_decay = 1e-4):
        super().__init__()
        
        #First ConvBlock
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        
        #Second ConvBlock
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,  padding="same")
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        
        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.bn3 = nn.BatchNorm1d(num_features=out_channels)
        
        
        #Squeeze and Excitation Block
        self.squeeze_excitation = SqueezeExcitationBlock(out_channels)
        
        ##Shortcut Connection Fighting Vanish Gradient
        self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.shortcut_batchNormalization = nn.BatchNorm1d(out_channels)
        
        #Adding Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=out_channels,   # depende del tamaño de tus features en D2
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm_proj = nn.Linear(256, out_channels)
        
        
    def forward(self, x, mask):
        #Shortcut 
        shortcut = self.shortcut(x) * mask.unsqueeze(1)
        shortcut = self.shortcut_batchNormalization(shortcut)
        
        #First Convolutional Neuronal Network
        x = self.conv1(x) * mask.unsqueeze(1)
        x = F.relu(self.bn1(x)) 
        
        #Second Convolutional Neuronal Network
        x = self.conv2(x) * mask.unsqueeze(1)
        x = F.relu(self.bn2(x))
        
        #Squeeze exitation block
        x = self.squeeze_excitation(x, mask)
        
        #Add Shortcut
        x += shortcut
        x = F.relu(x)
        
        
        #Droput
        x = self.dropout(x) * mask.unsqueeze(1)
        
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.bilstm(x)
        x = self.lstm_proj(x[:, -1, :])
        return x
        
        
        
        