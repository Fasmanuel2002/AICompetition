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
class SqueezeExcitationBlock(torch.nn.Module):
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
        
        
        