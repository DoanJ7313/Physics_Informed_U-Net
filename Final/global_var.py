# --- Dataset parameters ---
dataroot = r'/content/drive/MyDrive/se232/Final/dataset.mat' # colab
# dataroot = r'/home/jhdoan/Final/dataset.mat' # jupyterhub
main_dir = r'/content/drive/MyDrive/se232/Final/' # colab
# main_dir = r'/home/jhdoan/Final' # jupyterhub
unet_name = ''
workers = 0
batch_size = 256

# --- Network parameters ---

# Number of GPUs available, 0 for CPU mode
ngpu = 1

# --- Training parameters ---
n_epochs = 200
start_epoch = 0

# Learning rate for optimizers 
lr = 5e-3
lr_min = 1e-4

# Beta1 hyperparameter for Adam optimizers, 0.9 is default
beta1 = 0.9

# Weights for losses
weight_mse = 1.0
weight_en = 1.0e-3

config = {
    "ngpu": ngpu,
    "lr": lr,
    "lr_min": lr_min,
    "weight_mse": weight_mse,
    "weight_en": weight_en,
    "n_epochs": n_epochs,
    "start_epoch": start_epoch,
    "beta1": beta1,
    "workers": workers,
    "batch_size": batch_size,
    "dataroot": dataroot,
    "main_dir": main_dir,
    "unet_name": unet_name
}