#!/usr/bin/env python3
"""
Experiment 1: Performance Comparison with Different Missing Rates
Using FULL DGSDNet model with all components (SP, TP, DSD, GSC)
Supported missing rates: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import argparse
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
from datetime import datetime

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

# Add project path
sys.path.append('.')

# Import full DGSDNet model
from models import DGSDNet

def read_data(label_path, feature_root, modality='audio'):
    """Load feature data"""
    with open(label_path, 'rb') as f:
        data = pickle.load(f)
    
    # Parse data structure: [videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid]
    if isinstance(data, list) and len(data) == 6:
        videoIDs = data[0]
        names = []
        for vid, uids in videoIDs.items():
            names.extend(uids)
    else:
        names = []
        if isinstance(data, list) and len(data) > 0:
            for i, fold_data in enumerate(data):
                if isinstance(fold_data, dict):
                    for vid, uids in fold_data.items():
                        for uid in uids:
                            if isinstance(uid, (int, float)):
                                uid = str(uid)
                            names.append(uid)
                elif isinstance(fold_data, set):
                    for uid in fold_data:
                        if isinstance(uid, (int, float)):
                            uid = str(uid)
                        names.append(uid)
    
    features = []
    feature_dim = -1
    
    for name in names:
        if modality == 'video':
            feature_dir = os.path.join(feature_root, name)
            if os.path.exists(feature_dir):
                for suffix in ['compress_F.npy', 'compress_M.npy']:
                    feature_path = os.path.join(feature_dir, suffix)
                    if os.path.exists(feature_path):
                        single_feature = np.load(feature_path)
                        single_feature = single_feature.squeeze()
                        features.append(single_feature)
                        feature_dim = max(feature_dim, single_feature.shape[-1])
                        break
                else:
                    features.append(np.zeros((1024,)))
                    feature_dim = max(feature_dim, 1024)
            else:
                features.append(np.zeros((1024,)))
                feature_dim = max(feature_dim, 1024)
        else:
            feature_path = os.path.join(feature_root, name + '.npy')
            if os.path.exists(feature_path):
                single_feature = np.load(feature_path)
                single_feature = single_feature.squeeze()
                features.append(single_feature)
                feature_dim = max(feature_dim, single_feature.shape[-1])
            else:
                if modality == 'audio':
                    features.append(np.zeros((512,)))
                    feature_dim = max(feature_dim, 512)
                else:
                    features.append(np.zeros((1024,)))
                    feature_dim = max(feature_dim, 1024)
    
    name2feats = {}
    for i, name in enumerate(names):
        name2feats[name] = features[i]
    
    return name2feats, feature_dim

class FullDGSDNetDataset(Dataset):
    """Dataset for full DGSDNet model with all components"""
    
    def __init__(self, label_path, audio_root, text_root, video_root, mode='train', cv_fold=0, missing_rate=0.0):
        self.label_path = label_path
        self.mode = mode
        self.cv_fold = cv_fold
        self.missing_rate = missing_rate
        
        # Load features
        self.name2audio, self.adim = read_data(label_path, audio_root, 'audio')
        self.name2text, self.tdim = read_data(label_path, text_root, 'text')
        self.name2video, self.vdim = read_data(label_path, video_root, 'video')
        
        # Load label data
        with open(label_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Process data
        self._process_data()
    
    def _process_data(self):
        """Process data with REAL labels"""
        # Parse data structure: [videoIDs, videoLabels, videoSpeakers, videoSentences, trainVid, testVid]
        if isinstance(self.data, list) and len(self.data) == 6:
            videoIDs = self.data[0]
            videoLabels = self.data[1]
            videoSpeakers = self.data[2]
            videoSentences = self.data[3]
            trainVid = self.data[4]
            testVid = self.data[5]
        else:
            raise ValueError(f"Unexpected data format: {type(self.data)}, length: {len(self.data) if hasattr(self.data, '__len__') else 'N/A'}")
        
        # Organize data by video
        self.video_audio = {}
        self.video_text = {}
        self.video_visual = {}
        self.video_labels = {}
        self.video_speakers = {}
        
        # Process each video
        for vid in videoIDs.keys():
            uids = videoIDs[vid]
            labels = videoLabels[vid] if vid in videoLabels else []
            speakers = videoSpeakers[vid] if vid in videoSpeakers else []
            
            if not uids:
                continue
            
            # Collect all features for this video
            audio_feats = []
            text_feats = []
            visual_feats = []
            
            for i, uid in enumerate(uids):
                # Features
                if uid in self.name2audio:
                    audio_feats.append(self.name2audio[uid])
                else:
                    audio_feats.append(np.zeros((self.adim,)))
                
                if uid in self.name2text:
                    text_feats.append(self.name2text[uid])
                else:
                    text_feats.append(np.zeros((self.tdim,)))
                
                if uid in self.name2video:
                    visual_feats.append(self.name2video[uid])
                else:
                    visual_feats.append(np.zeros((self.vdim,)))
            
            # Store data
            self.video_audio[vid] = np.array(audio_feats)
            self.video_text[vid] = np.array(text_feats)
            self.video_visual[vid] = np.array(visual_feats)
            self.video_labels[vid] = np.array(labels)
            
            # Convert speaker strings to indices (M=0, F=1)
            speaker_indices = []
            for s in speakers:
                if isinstance(s, str):
                    speaker_indices.append(0 if s == 'M' else 1)
                else:
                    speaker_indices.append(int(s))
            self.video_speakers[vid] = np.array(speaker_indices)
        
        # Use provided train/test split
        self.train_vids = [vid for vid in trainVid if vid in self.video_audio]
        self.test_vids = [vid for vid in testVid if vid in self.video_audio]
        
        if self.mode == 'train':
            self.current_vids = self.train_vids
        else:
            self.current_vids = self.test_vids
    
    def __len__(self):
        return len(self.current_vids)
    
    def __getitem__(self, idx):
        vid = self.current_vids[idx]
        
        # Get features
        audio = torch.FloatTensor(self.video_audio[vid])
        text = torch.FloatTensor(self.video_text[vid])
        visual = torch.FloatTensor(self.video_visual[vid])
        labels = torch.LongTensor(self.video_labels[vid])
        speakers = torch.LongTensor(self.video_speakers[vid])
        
        # Apply missing rate
        missing_mask = self._generate_missing_mask(audio.shape[0])
        
        # Create qmask (speaker mask for graph construction)
        qmask = speakers.clone()
        
        return {
            'audio': audio,
            'text': text,
            'visual': visual,
            'labels': labels,
            'speakers': speakers,
            'qmask': qmask,
            'vid': vid,
            'missing_mask': missing_mask
        }
    
    def _generate_missing_mask(self, seq_len):
        """
        Generate missing mask following paper definition:
        M = 1 - sum(s_i) / (n × M)
        where s_i is number of available modalities for i-th sample
        """
        num_modalities = 3
        missing_mask = torch.zeros(seq_len, num_modalities, dtype=torch.bool)
        
        # For each utterance, randomly decide which modalities are missing
        # to achieve the target missing rate
        for i in range(seq_len):
            # Randomly determine number of missing modalities for this utterance
            # Ensure at least 1 modality is available
            num_missing = 0
            if self.missing_rate > 0:
                # Sample from binomial to match expected missing rate
                target_available = num_modalities * (1 - self.missing_rate)
                num_available = max(1, int(np.random.normal(target_available, 0.5)))
                num_available = min(num_modalities, max(1, num_available))
                num_missing = num_modalities - num_available
            
            if num_missing > 0:
                # Randomly select which modalities to mask
                missing_indices = random.sample(range(num_modalities), num_missing)
                for idx in missing_indices:
                    missing_mask[i, idx] = True
        
        return missing_mask
    
    def collate_fn(self, batch):
        """Batch processing function"""
        audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
        text = pad_sequence([item['text'] for item in batch], batch_first=True)
        visual = pad_sequence([item['visual'] for item in batch], batch_first=True)
        labels = pad_sequence([item['labels'] for item in batch], batch_first=True)
        speakers = pad_sequence([item['speakers'] for item in batch], batch_first=True)
        qmask = pad_sequence([item['qmask'] for item in batch], batch_first=True)
        missing_mask = pad_sequence([item['missing_mask'] for item in batch], batch_first=True)
        vids = [item['vid'] for item in batch]
        
        return {
            'audio': audio,
            'text': text,
            'visual': visual,
            'labels': labels,
            'speakers': speakers,
            'qmask': qmask,
            'missing_mask': missing_mask,
            'vids': vids
        }

class ModelArgs:
    """Arguments for DGSDNet model"""
    def __init__(self, text_dim=1024, audio_dim=512, video_dim=1024, hidden_dim=128, num_classes=4,
                 ablation_type='full', alpha=0.5):
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Ablation settings
        self.disable_speaker_path = (ablation_type == 'w/o_SP')
        self.disable_temporal_path = (ablation_type == 'w/o_TP')
        self.disable_diffusion = (ablation_type == 'w/o_DSD')
        self.disable_gated_classifier = (ablation_type == 'w/o_GSC')
        
        # Model parameters
        self.kernel_size_l = 1
        self.kernel_size_a = 1
        self.kernel_size_v = 1
        self.num_diffusion_steps = 100
        self.sigma = 25.0
        self.graph_hidden_size = hidden_dim
        
        # Hyperparameters for loss function (Equation 7 in paper)
        self.alpha = alpha  # Graph fusion weight

def calculate_waf1(predictions, labels, num_classes):
    """Calculate Weighted Average F1 score"""
    f1_scores = []
    for i in range(num_classes):
        if i in labels:
            f1 = f1_score(labels == i, predictions == i, average='binary', zero_division=0)
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    # Calculate weighted average
    weights = np.bincount(labels, minlength=num_classes)
    weights = weights / weights.sum()
    waf1 = np.average(f1_scores, weights=weights)
    return waf1

def train_missing_rate_experiment(dataset_name='iemocap', missing_rate=0.0, cv_fold=0, epochs=50, 
                                  ablation_type='full', hidden_dim=32, beta=0.5, lambda_reg=0.1, alpha=0.5):
    """Missing rate experiment with FULL DGSDNet model"""
    print(f"\n{'='*70}")
    print(f"Missing Rate Experiment [FULL DGSDNet Model]")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Missing Rate: {missing_rate*100:.0f}%")
    print(f"Ablation Type: {ablation_type}")
    print(f"CV Fold: {cv_fold}")
    print(f"Epochs: {epochs}")
    print(f"Hyperparameters: β={beta}, λ={lambda_reg}, α={alpha}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set paths
    if dataset_name == 'iemocap':
        label_path = './dataset/IEMOCAPFour/IEMOCAP_features_raw_4way.pkl'
        audio_root = './dataset/IEMOCAPFour/features/wav2vec-large-c-UTT'
        text_root = './dataset/IEMOCAPFour/features/deberta-large-4-UTT'
        video_root = './dataset/IEMOCAPFour/features/manet_UTT'
        num_classes = 4
    else:
        label_path = './dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl'
        audio_root = './dataset/CMUMOSI/features/wav2vec-large-c-UTT'
        text_root = './dataset/CMUMOSI/features/deberta-large-4-UTT'
        video_root = './dataset/CMUMOSI/features/manet_UTT'
        num_classes = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = FullDGSDNetDataset(
        label_path=label_path,
        audio_root=audio_root,
        text_root=text_root,
        video_root=video_root,
        mode='train',
        cv_fold=cv_fold,
        missing_rate=missing_rate
    )
    
    test_dataset = FullDGSDNetDataset(
        label_path=label_path,
        audio_root=audio_root,
        text_root=text_root,
        video_root=video_root,
        mode='test',
        cv_fold=cv_fold,
        missing_rate=missing_rate
    )
    
    print(f"Train videos: {len(train_dataset.train_vids)}")
    print(f"Test videos: {len(test_dataset.test_vids)}")
    print(f"Feature dims - Text: {train_dataset.tdim}, Audio: {train_dataset.adim}, Video: {train_dataset.vdim}")
    print(f"{'='*70}\n")
    
    # Create data loaders (GPU-optimized)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Optimized for GPU
        shuffle=True, 
        collate_fn=train_dataset.collate_fn,
        num_workers=4,  # Parallel data loading
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  # Optimized for GPU
        shuffle=False, 
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create FULL DGSDNet model
    model_args = ModelArgs(
        text_dim=train_dataset.tdim,
        audio_dim=train_dataset.adim,
        video_dim=train_dataset.vdim,
        hidden_dim=hidden_dim,  # Adjustable (default: 32 for 4K samples, paper: 128 for larger datasets)
        num_classes=num_classes,
        ablation_type=ablation_type,
        alpha=alpha  # Graph fusion weight
    )
    
    print(f"hidden_dim: {hidden_dim} ({'paper setting' if hidden_dim == 128 else 'adjusted for dataset size'})")
    
    # Set diffusion steps according to paper
    model_args.num_diffusion_steps = 100  # Paper setting for full performance
    
    model = DGSDNet(model_args).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # GPU memory info
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Compute class weights for imbalanced dataset
    train_labels = []
    for vid in train_dataset.train_vids:
        train_labels.extend(train_dataset.video_labels[vid].tolist())
    train_labels = np.array(train_labels)
    
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Optimizer with balanced regularization
    base_lr = 2e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    # Learning rate warmup + ReduceLROnPlateau
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )
    
    # Weighted cross-entropy (no label smoothing for imbalanced data)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    best_waf1 = 0.0
    best_epoch = 0
    
    # Training loop
    print("\nStarting training...\n")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                         bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                labels = batch['labels'].to(device)
                qmask = batch['qmask'].to(device)
                missing_mask = batch['missing_mask'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with full model (get auxiliary losses)
                outputs, aux_losses = model(text, audio, visual, missing_mask=missing_mask, qmask=qmask, return_auxiliary_losses=True)
                
                # Compute total loss (Paper Equation 7)
                # L_total = β·L_miss + CE_loss + λ·Σ||Z^k||_F^2
                ce_loss = criterion(outputs.reshape(-1, num_classes), labels.reshape(-1))
                recon_loss = aux_losses['recon_loss']
                
                # Gate regularization (Frobenius norm)
                gate_reg_loss = 0.0
                if aux_losses['gates'] is not None:
                    # ||Z^k||_F^2 for each graph k
                    gate_reg_loss = torch.norm(aux_losses['gates'], p='fro') ** 2
                
                # Total loss
                loss = beta * recon_loss + ce_loss + lambda_reg * gate_reg_loss
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Gradient monitoring (every 50 batches in first 5 epochs)
                if epoch < 5 and batch_idx % 50 == 0:
                    grad_norms = [p.grad.norm().item() for p in model.parameters() 
                                 if p.grad is not None and p.requires_grad]
                    if grad_norms:
                        print(f"\n  Gradient: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, "
                              f"mean={sum(grad_norms)/len(grad_norms):.6f}")
                
                # Update progress bar with loss breakdown
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce': f'{ce_loss.item():.3f}',
                    'recon': f'{recon_loss.item():.3f}' if isinstance(recon_loss, torch.Tensor) else f'{recon_loss:.3f}',
                    'grad': f'{grad_norm:.3f}'
                })
                
            except Exception as e:
                print(f"\nWarning: Training batch {batch_idx} error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation phase
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Validation progress bar
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]', 
                        bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        with torch.no_grad():
            for batch in test_pbar:
                try:
                    text = batch['text'].to(device)
                    audio = batch['audio'].to(device)
                    visual = batch['visual'].to(device)
                    labels = batch['labels'].to(device)
                    qmask = batch['qmask'].to(device)
                    missing_mask = batch['missing_mask'].to(device)
                    
                    outputs = model(text, audio, visual, missing_mask=missing_mask, qmask=qmask)
                    predictions = torch.argmax(outputs, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    
                except Exception as e:
                    print(f"\nWarning: Test batch error: {e}")
                    continue
        
        # Calculate metrics
        if all_predictions and all_labels:
            predictions_array = np.array(all_predictions)
            labels_array = np.array(all_labels)
            
            waf1 = calculate_waf1(predictions_array, labels_array, num_classes)
            accuracy = np.mean(predictions_array == labels_array)
            
            # Diagnostic: Check prediction distribution
            if epoch < 5 or (epoch + 1) % 20 == 0:
                unique_preds, counts_preds = np.unique(predictions_array, return_counts=True)
                pred_dist = {int(p): int(c) for p, c in zip(unique_preds, counts_preds)}
                print(f"   Prediction dist: {pred_dist}")
            
            # Update best results
            is_best = waf1 > best_waf1
            if is_best:
                best_waf1 = waf1
                best_epoch = epoch + 1
            
            # Print epoch summary with learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {accuracy:.4f}, "
                  f"WAF1: {waf1:.4f}, LR: {current_lr:.6f} {'[BEST]' if is_best else ''}")
            
            # Update learning rate schedulers
            if epoch < warmup_epochs:
                warmup_scheduler.step()  # Linear warmup
            else:
                plateau_scheduler.step(waf1)  # Plateau-based after warmup
        else:
            print(f"Warning: Epoch {epoch+1} test problem")
    
    # Training completed
    end_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"Training Completed!")
    print(f"Best WAF1: {best_waf1:.4f} ({best_waf1*100:.2f}%) at Epoch {best_epoch}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Save results to results/ directory
    result_data = {
        'dataset': dataset_name,
        'missing_rate': float(missing_rate),
        'cv_fold': cv_fold,
        'ablation_type': ablation_type,
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'best_waf1': float(best_waf1),
        'best_epoch': best_epoch,
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    # Save individual result
    result_filename = f"results/{dataset_name}_mr{missing_rate:.1f}_fold{cv_fold}_{ablation_type}_h{hidden_dim}.json"
    with open(result_filename, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Results saved to: {result_filename}")
    
    # Save model checkpoint if best performance
    if best_waf1 > 0.5:  # Only save if reasonable performance
        checkpoint_filename = f"results/{dataset_name}_mr{missing_rate:.1f}_fold{cv_fold}_{ablation_type}_h{hidden_dim}_best.pth"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'waf1': best_waf1,
            'model_args': vars(model_args) if hasattr(model_args, '__dict__') else model_args.__dict__
        }, checkpoint_filename)
        print(f"Best model saved to: {checkpoint_filename}\n")
    
    return best_waf1

def run_missing_rate_experiments(hidden_dim=32, beta=0.5, lambda_reg=0.1, alpha=0.5):
    """Run all missing rate experiments"""
    datasets = ['iemocap', 'cmumosi']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = {}
    
    print("\n" + "="*80)
    print("Batch Missing Rate Experiments [FULL DGSDNet Model]")
    print("="*80)
    print(f"Datasets: {', '.join([d.upper() for d in datasets])}")
    print(f"Missing Rates: {', '.join([f'{mr*100:.0f}%' for mr in missing_rates])}")
    print(f"Hyperparameters: hidden_dim={hidden_dim}, β={beta}, λ={lambda_reg}, α={alpha}")
    print(f"Total: {len(datasets)} x {len(missing_rates)} = {len(datasets)*len(missing_rates)} experiments")
    print("="*80 + "\n")
    
    experiment_count = 0
    total_experiments = len(datasets) * len(missing_rates)
    
    for dataset in datasets:
        results[dataset] = {}
        
        for missing_rate in missing_rates:
            experiment_count += 1
            print(f"\nExperiment Progress: {experiment_count}/{total_experiments}")
            
            waf1 = train_missing_rate_experiment(
                dataset_name=dataset,
                missing_rate=missing_rate,
                cv_fold=0,
                epochs=100,
                ablation_type='full',
                hidden_dim=hidden_dim,
                beta=beta,
                lambda_reg=lambda_reg,
                alpha=alpha
            )
            results[dataset][missing_rate] = waf1
    
    # Save results to results/ directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/missing_rate_h{hidden_dim}_{timestamp}.json'
    
    save_data = {
        'experiment_type': 'missing_rate',
        'hidden_dim': hidden_dim,
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'total_experiments': total_experiments,
            'datasets': datasets,
            'missing_rates': missing_rates
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print results table (matching paper format)
    print(f"\n{'='*80}")
    print("Experiment Results Summary (WAF1 scores in %)")
    print(f"{'='*80}\n")
    
    for dataset in datasets:
        print(f"\n{dataset.upper()} Results:")
        print(f"{'Missing Rate':<15}", end='')
        for mr in missing_rates:
            print(f"{mr*100:>6.0f}%", end='')
        print(f"{'Avg.':>8}")
        print("-" * 80)
        
        print(f"{'Our Method':<15}", end='')
        for mr in missing_rates:
            waf1 = results[dataset][mr]
            print(f"{waf1*100:>6.2f}", end='')
        
        # Calculate average
        avg_waf1 = np.mean([results[dataset][mr] for mr in missing_rates])
        print(f"{avg_waf1*100:>8.2f}")
        print()
    
    print(f"{'='*80}")
    print(f"All Experiments Completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Missing Rate Experiments with Full DGSDNet')
    parser.add_argument('--dataset', type=str, default='iemocap', choices=['iemocap', 'cmumosi'],
                       help='Dataset selection')
    parser.add_argument('--missing_rate', type=float, default=0.0,
                       help='Missing rate (0.0-0.7)')
    parser.add_argument('--cv_fold', type=int, default=0,
                       help='Cross-validation fold')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100 for paper results)')
    parser.add_argument('--ablation_type', type=str, default='full',
                       choices=['full', 'w/o_SP', 'w/o_TP', 'w/o_DSD', 'w/o_GSC'],
                       help='Ablation type (default: full model)')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all missing rate experiments')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension (default: 32 for 4K samples, paper: 128 for larger datasets)')
    
    # Hyperparameters (Paper Equation 7)
    parser.add_argument('--beta', type=float, default=0.5,
                       help='β: Weight for reconstruction loss (default: 0.5)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                       help='λ: Weight for gate regularization (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='α: Graph fusion weight, A=α·E^s+(1-α)·E^q (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_missing_rate_experiments(hidden_dim=args.hidden_dim, beta=args.beta, 
                                     lambda_reg=args.lambda_reg, alpha=args.alpha)
    else:
        waf1 = train_missing_rate_experiment(
            dataset_name=args.dataset,
            missing_rate=args.missing_rate,
            cv_fold=args.cv_fold,
            epochs=args.epochs,
            ablation_type=args.ablation_type,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            alpha=args.alpha
        )
        print(f"\nFinal WAF1: {waf1*100:.2f}%")