# BACKGROUND ------
def build_model(pretrained=True, fine_tune=True, num_classes=1):
    """
    Function to build the neural network model. Returns the final model.
    Parameters
    :param pretrained (bool): Whether to load the pre-trained weights or not.
    :param fine_tune (bool): Whether to train the hidden layers or not.
    :param num_classes (int): Number of classes in the dataset. 
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
    model = models.resnet18(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # change the final classification head, it is trainable
    model.fc = nn.Linear(512, num_classes)
    return model
  
# build the model
model = build_model(
    pretrained=False, fine_tune=True, num_classes=10
).to(device)
print(model)

# CORE --------
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
