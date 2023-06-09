# REF: https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
# good tutorial contains how to save the best model based on validation loss =(apply early stopping)
# very stuctured scripts

# BACKGROUND ------
# pass

# CORE --------
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
