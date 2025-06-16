#!/usr/bin/env python3
"""
Evaluation module for ILWGAN and LWGAN models.
This module contains evaluation metrics and functions used by main.py.
"""

import torch
import torch.nn as nn
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import os

def evaluate_model_quality(model, eval_noise, fid_metric, is_metric, device='cpu'):
    """
    Evaluate model sample quality using FID and Inception Score.
    
    Args:
        model: Model to evaluate (ILWGAN or LWGAN)
        eval_noise: Fixed noise for evaluation
        fid_metric: FID metric instance
        is_metric: Inception Score metric instance
        device: Device to run evaluation on
    
    Returns:
        Tuple of (FID score, Inception Score)
    """
    model.gen.eval()
    
    with torch.no_grad():
        samples = model.generate(eval_noise)
        
        if samples.shape[1] == 1:
            samples = samples.repeat(1, 3, 1, 1)
        
        samples = (samples + 1) / 2.0
        samples = torch.clamp(samples, 0, 1)
    
    fid_metric.update(samples)
    is_metric.update(samples)
    
    fid_score = fid_metric.compute()
    is_score = is_metric.compute()
    
    fid_metric.reset()
    is_metric.reset()
    
    return fid_score.item(), is_score.item()

def compute_latent_interpolation_quality(model, z1, z2, num_steps=10):
    """
    Evaluate the quality of latent space interpolation.
    
    Args:
        model: Model to evaluate
        z1, z2: Start and end latent codes
        num_steps: Number of interpolation steps
    
    Returns:
        List of interpolated images
    """
    model.gen.eval()
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = model.generate(z_interp.unsqueeze(0))
            img = (img + 1) / 2.0  # Scale to [0,1]
            interpolated_images.append(img.squeeze().cpu().numpy())
    
    return interpolated_images

def analyze_latent_space_structure(model, data_samples, threshold=1e-2):
    """
    Analyze the structure of the learned latent space.
    
    Args:
        model: Model with encoder (netQ)
        data_samples: Sample data to encode
        threshold: Threshold for effective dimension computation
    
    Returns:
        Dictionary with analysis results
    """
    model.netQ.eval()
    
    with torch.no_grad():
        data_flat = data_samples.view(data_samples.size(0), -1)
        latent_codes = model.netQ(data_flat)
        
        latent_centered = latent_codes - torch.mean(latent_codes, dim=0, keepdim=True)
        
        cov = torch.matmul(latent_centered.t(), latent_centered) / (latent_centered.shape[0] - 1)
        
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals_np = eigvals.cpu().numpy()
        
        sorted_eigvals = np.sort(eigvals_np)[::-1]
        
        effective_dim = np.sum(sorted_eigvals > threshold)
        
        total_var = np.sum(sorted_eigvals)
        explained_var_ratio = sorted_eigvals / total_var
        
        return {
            'eigenvalues': sorted_eigvals,
            'effective_dimension': effective_dim,
            'explained_variance_ratio': explained_var_ratio,
            'total_variance': total_var
        }

def plot_training_curves(ilwgan_losses, lwgan_losses, output_dir):
    """
    Plot and save training loss curves.
    
    Args:
        ilwgan_losses: List of ILWGAN training losses
        lwgan_losses: List of LWGAN training losses
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(ilwgan_losses) + 1)
    
    plt.plot(epochs, ilwgan_losses, 'o-', label='ILWGAN', linewidth=2)
    plt.plot(epochs, lwgan_losses, 'x-', label='LWGAN', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join(output_dir, 'training_curves.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def plot_quality_metrics(epochs, ilwgan_fid, lwgan_fid, ilwgan_is, lwgan_is, output_dir):
    """
    Plot and save quality metric curves.
    
    Args:
        epochs: List of epoch numbers
        ilwgan_fid, lwgan_fid: FID scores for both models
        ilwgan_is, lwgan_is: Inception scores for both models
        output_dir: Directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, ilwgan_fid, 'o-', label='ILWGAN', linewidth=2)
    ax1.plot(epochs, lwgan_fid, 'x-', label='LWGAN', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID Score (lower is better)')
    ax1.set_title('FID Score Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, ilwgan_is, 'o-', label='ILWGAN', linewidth=2)
    ax2.plot(epochs, lwgan_is, 'x-', label='LWGAN', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Inception Score (higher is better)')
    ax2.set_title('Inception Score Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'quality_metrics.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def compute_reconstruction_error(model, real_images):
    """
    Compute reconstruction error for autoencoder-like evaluation.
    
    Args:
        model: Model with encoder and generator
        real_images: Real images to reconstruct
    
    Returns:
        Mean reconstruction error
    """
    model.gen.eval()
    model.netQ.eval()
    
    with torch.no_grad():
        real_flat = real_images.view(real_images.size(0), -1)
        latent_codes = model.netQ(real_flat)
        
        reconstructed = model.generate(latent_codes)
        
        mse = torch.mean((real_images - reconstructed) ** 2)
        
    return mse.item()
