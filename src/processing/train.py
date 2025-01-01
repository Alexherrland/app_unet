import time
import os
import json
import torch
import matplotlib.pyplot as plt
from torcheval.metrics.functional import peak_signal_noise_ratio
from processing.utils import mean_absolute_error, crps_gaussian, generate_images

def train_epoch(model, optimizer, criterion, train_dataloader, device, scaler, epoch=0, log_interval=1500):  # Agregar scaler
    model.train()
    total_psnr, total_count = 0, 0
    losses = []
    start_time = time.time()
    
    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()

        total_psnr += peak_signal_noise_ratio(predictions, labels)
        total_count += 1
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| psnr {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_psnr / total_count
                )
            )
            total_psnr, total_count = 0, 0
            start_time = time.time()

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss

def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []
    mae_scores = []
    crps_scores = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            # Calcular MAE
            mae = mean_absolute_error(predictions, labels)
            mae_scores.append(mae)

            # Calcular CRPS
            crps = crps_gaussian(predictions, labels)
            crps_scores.append(crps)

            total_psnr +=  peak_signal_noise_ratio(predictions, labels)
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    epoch_mae = sum(mae_scores) / len(mae_scores)
    epoch_crps = sum(crps_scores) / len(crps_scores)
    
    return epoch_psnr, epoch_loss, epoch_mae, epoch_crps

def train_model(model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device, scheduler=None):
    train_psnrs, train_losses = [], []
    eval_psnrs, eval_losses = [], []
    mae_scores, crps_scores = [], []
    best_psnr_eval = -1000
    times = []
    scaler = torch.amp.GradScaler(device)
    for epoch in range(35, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_psnr, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device,scaler, epoch)
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)

        # Evaluation
        eval_psnr, eval_loss, eval_mae, eval_crps = evaluate_epoch(model, criterion, valid_dataloader, device)
        eval_psnrs.append(eval_psnr.cpu())
        eval_losses.append(eval_loss)
        mae_scores.append(eval_mae)
        crps_scores.append(eval_crps)

        # Learning Rate Scheduler
        if scheduler is not None:
            scheduler.step(eval_loss)

        # Save best model
        if best_psnr_eval < eval_psnr :
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
            inputs_t, targets_t = next(iter(valid_dataloader))
            generate_images(model, inputs_t, targets_t,epoch)
            best_psnr_eval = eval_psnr
        times.append(time.time() - epoch_start_time)
        # Print loss, psnr end epoch
        print("-" * 89)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train psnr {:8.3f} | Train Loss {:8.3f} "
            "| Valid psnr {:8.3f} | Valid Loss {:8.3f} | MAE {:8.3f} | CRPS {:8.3f}".format(
                epoch, time.time() - epoch_start_time, train_psnr, train_loss, 
                eval_psnr, eval_loss, eval_mae, eval_crps
            )
        )
        print("-" * 89)

    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_psnr': train_psnrs,
        'train_loss': train_losses,
        'valid_psnr': eval_psnrs,
        'valid_loss': eval_losses,
        'mae_scores': mae_scores,
        'crps_scores': crps_scores,
        'time': times
    }
    return model, metrics