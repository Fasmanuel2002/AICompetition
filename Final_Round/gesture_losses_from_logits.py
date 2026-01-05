import torch.nn as nn
import torch

target_gestures = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
]
non_target_gestures = [
    "Write name on leg",
    "Wave hello",
    "Glasses on/off",
    "Text on phone",
    "Write name in air",
    "Feel around in tray and pull out an object",
    "Scratch knee/leg skin",
    "Pull air toward your face",
    "Drink from bottle/cup",
    "Pinch knee/leg skin",
]
all_gestures = target_gestures + non_target_gestures
dict_gestures = {v: i for i, v in enumerate(all_gestures)}
N_TARGET = len(target_gestures)  # 8
N_GESTURES = len(all_gestures) 
bce_bin = nn.BCEWithLogitsLoss()
def gesture_losses_from_logits(logits_g18: torch.Tensor, y_g18: torch.Tensor, label_smoothing: float = 0.0, lambda_bin: float = 1.0):
    """
    logits_g18: (B, 18) over all gestures (targets first, then non-targets)
    y_g18:      (B,)   in [0..17]
    """
    # --- Collapse non-target into ONE bucket (index = 8) for macro part ---
    y9 = torch.clamp(y_g18, max=N_TARGET)  

    logits_target = logits_g18[:, :N_TARGET]                      
    logits_nont   = torch.logsumexp(logits_g18[:, N_TARGET:], dim=1, keepdim=True) 
    logits9 = torch.cat([logits_target, logits_nont], dim=1)       

    ce9 = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_macro_proxy = ce9(logits9, y9)

    # --- Binary head derived from logits: target vs non-target ---
    logit_t = torch.logsumexp(logits_target, dim=1)               
    logit_n = torch.logsumexp(logits_g18[:, N_TARGET:], dim=1)    
    bin_logit = logit_t - logit_n                                 
    y_bin = (y_g18 < N_TARGET).float()                            
    loss_bin = bce_bin(bin_logit, y_bin)

    return loss_macro_proxy + lambda_bin * loss_bin, loss_macro_proxy.detach(), loss_bin.detach(), bin_logit.detach()