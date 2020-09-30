import os
import time
import numpy as np
import torch
import torch.nn as nn
import nsml
from evaluation import *

def train(args, trn_cfg):
    
    train_loader = trn_cfg['train_loader']
    valid_loader = trn_cfg['valid_loader']
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']

    best_epoch = 0
    best_val_score = 0.0

    # Train the model
    for epoch in range(args.epochs):
        
        start_time = time.time()
    
        trn_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device)
        val_loss, val_score = validation(args, trn_cfg, model, criterion, valid_loader, device)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - trn_loss: {:.4f}  val_loss: {:.4f}  val_score: {:.4f} lr: {:.5f}  time: {:.0f}s\n".format(
                epoch+1, epoch_train_loss, epoch_val_loss, val_score, lr[0], elapsed))

        nsml.report(summary=True, step=epoch, epoch_total=args.epochs, trn_loss=trn_loss, val_loss=val_loss, val_score=val_score)
        
        # save model weight
        if val_score > best_val_score:
            best_val_score = val_score            
            file_save_name = 'best_score' + '_fold' + str(args.fold_num)
            nsml.save(file_save_name)

        if args.scheduler == 'Plateau':
            scheduler.step(val_score)
        else:
            scheduler.step()
    

def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device):

    model.train()
    trn_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(train_loader):

        if device:
            images = images.to(device)
            labels = labels.reshape(-1, 1).to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    epoch_train_loss = trn_loss / len(train_loader)

    return epoch_train_loss


def validation(args, trn_cfg, model, criterion, valid_loader, device):
    
    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
            
            total_labels.append(labels)

            if device:
                images = images.to(device)
                labels = labels.reshape(-1, 1).to(device)

            outputs = torch.round(torch.sigmoid(model(images)))

            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            total_outputs.append(outputs.cpu().detach().numpy())

    epoch_val_loss = val_loss / len(valid_loader)

    total_labels = np.concatenate(total_labels).tolist()
    total_outputs = np.concatenate(total_outputs).tolist()

    metrics = evaluation_metrics(total_labels, total_outputs)
    val_score = np.round(np.mean(list(metrics.values())), 4)
    
    return epoch_val_loss, val_score

