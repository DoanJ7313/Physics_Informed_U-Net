# Libraries
import time, torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from unet import UNet
from loss_func import energy_loss
from global_var import config
from dataset import Dataset
import os

def validate_UNET(model, dataloader, mse, eng, D, device, weight_en):
    model.eval()
    val_losses, val_mse, val_eng = [], [], []

    with torch.no_grad():
        for data in dataloader:
            node_inp, t, u = data
            node_inp, t, u = node_inp.to(device), t.to(device), u.to(device)

            inp = torch.cat([node_inp, t], dim=1)
            pred = model(inp).float()
            mse_loss = mse(pred, u.float())
            eng_loss = eng(pred, u.float(), D, t.float())
            loss_tot = config["weight_mse"] * mse_loss + weight_en * eng_loss

            val_losses.append(loss_tot.item())
            val_mse.append(mse_loss.item())
            val_eng.append(eng_loss.item())

    avg_total = sum(val_losses) / len(val_losses)
    avg_mse = sum(val_mse) / len(val_mse)
    avg_eng = sum(val_eng) / len(val_eng)

    return avg_total, avg_mse, avg_eng

# Training Loop
def train_UNET(model, n_epoch, train_loader, mse, eng, optimizer, scheduler, 
               test_loader, D, device, rank, start_epoch):

    # Loss tracking
    train_losses, train_mse, train_eng = [], [], []
    val_losses, val_mse_list, val_eng_list = [], [], []

    # Early stopping setup
    patience = 30
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = None
    best_epoch = -1

    # Timer start
    start_time = time.time()

    for epoch in range(start_epoch, n_epoch):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_losses, epoch_mse, epoch_eng = [], [], []

        # decay_rate = 0.95
        # weight_en = config["weight_en"] * (decay_rate ** epoch)
        weight_en = config["weight_en"]

        for i, data in enumerate(train_loader):
            node_inp, t, u = data
            node_inp, t, u = node_inp.to(device), t.to(device), u.to(device)

            optimizer.zero_grad()

            inp = torch.cat([node_inp, t], dim=1)
            pred = model(inp).float()
            mse_loss = mse(pred, u.float())
            eng_loss = eng(pred, u.float(), D, t.float())
            loss_tot = config["weight_mse"] * mse_loss + weight_en * eng_loss

            loss_tot.backward()
            optimizer.step()
            
            epoch_losses.append(loss_tot.item())
            epoch_mse.append(mse_loss.item())
            epoch_eng.append(eng_loss.item())

        # Epoch averages
        train_losses.append(sum(epoch_losses) / len(epoch_losses))
        train_mse.append(sum(epoch_mse) / len(epoch_mse))
        train_eng.append(sum(epoch_eng) / len(epoch_eng))

        # Validation
        val_loss, val_mse, val_eng = validate_UNET(
            model, test_loader, mse, eng, D, device, weight_en)
        
        scheduler.step(val_loss)
        
        val_losses.append(val_loss)
        val_mse_list.append(val_mse)
        val_eng_list.append(val_eng)

        # Early stopping (just tracking best, no break)
        stop_training = torch.tensor(0, device=device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            best_model_wts = (
                model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            )

            if rank == 0:
                save_dir = config["main_dir"] + "Train_Results/"
            #     torch.save(best_model_wts, save_dir + "unet_best.pth")
        
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 and rank == 0:
            print(
                f'[{epoch}/{n_epoch}] [{i}/{len(train_loader)}] '
                f'UNet Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss} '
                f'LR = {current_lr:.2e} '
            )
            torch.cuda.empty_cache()
        
        if rank == 0 and (epoch + 1) % 50 == 0:
            model_eval = model.module if isinstance(model, DDP) else model
        
            # Save full training state
            checkpoint = {
                "epoch": epoch + 1,
                "model": model_eval.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_losses": train_losses,
                "train_mse": train_mse,
                "train_eng": train_eng,
                "val_losses": val_losses,
                "val_mse": val_mse,
                "val_eng": val_eng
            }
        
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at epoch {epoch+1} to {save_path}")
            torch.cuda.empty_cache()

    # Timer end
    elapsed_time = time.time() - start_time
    if rank == 0:
        print(f"\nTraining Complete! \nElapsed time: {elapsed_time:.2f} seconds")


    if rank == 0:
        save_dir = config["main_dir"] + "Train_Results/"
        torch.save(
            {
                "train_losses": train_losses, 
                "train_mse": train_mse,
                "train_eng": train_eng,
                "val_losses": val_losses,
                "val_mse": val_mse_list,
                "val_eng": val_eng_list
            },
            save_dir + "unet_outputs.pth",
        )

        # Save final model (last weights)
        torch.save(
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            save_dir + "unet_last.pth",
        )
        
        # Save best weights
        def save_best(state_dict, name_suffix):
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k if k.startswith("module.") else f"module.{k}"
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict, strict=False)

        # Run both best + last
        if best_model_wts is not None:
            # reload a fresh model with best weights
            # best_model = UNet().to(device)  
            # best_model.load_state_dict(best_model_wts)
            # save_best(best_model.state_dict(), "best")
            print(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}")

        # now save last model
        last_model_wts = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        save_best(last_model_wts, "last")


    return train_losses, val_losses

# Training Call
def train(rank, world_size):
    torch.cuda.empty_cache()
    
    print(f"Rank {rank} out of {world_size} GPUs")
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size
        )

    # Checkpoint / Resume
    resume = False
    start_epoch = config["start_epoch"]

    # Initialize model
    model = UNet().to(device)

    checkpoint = None
    if resume:
        checkpoint_path = os.path.join(config["main_dir"], f"Train_Results/checkpoint_epoch_{start_epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        start_epoch = checkpoint['epoch']

    # Multiple GPU wrap
    if world_size > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=True)

    # Optimizer & Scheduler (created AFTER DDP)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], 0.9))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10,
                                  threshold=1e-4, threshold_mode="rel", cooldown=0,
                                  min_lr=config["lr_min"])

    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # Force LR to match checkpoint
        for pg, cpg in zip(optimizer.param_groups, checkpoint['optimizer']['param_groups']):
            pg['lr'] = cpg['lr']

    print("LR after resume:", optimizer.param_groups[0]['lr'])

    # Dataset / Dataloader
    train_dset, test_dset = Dataset.get_dataset(config["dataroot"])
    _, D, _ = Dataset.get_constants(config["dataroot"])

    train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=test_sampler, shuffle=True, drop_last=True, pin_memory=True
    )

    print(f"Training set size: {len(train_dset)}")
    print(f"REMINDER: STARTING FROM EPOCH {start_epoch}")

    # Call training loop
    try:
        losses, val_losses = train_UNET(
            model=model, n_epoch=config["n_epochs"], train_loader=train_loader,
            mse=nn.MSELoss(), eng=energy_loss, optimizer=optimizer, scheduler=scheduler,
            test_loader=test_loader, D=D, device=device, rank=rank, start_epoch=start_epoch
        )
    finally:
        # Cleanup distributed
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()