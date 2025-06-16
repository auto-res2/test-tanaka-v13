#!/usr/bin/env python3
"""
Preprocessing module for ILWGAN and LWGAN experiments.
This module contains data preprocessing and synthetic data generation functions.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SyntheticManifoldDataset(Dataset):
    """
    Dataset for synthetic manifold data used in intrinsic dimension experiments.
    """
    def __init__(self, n_samples=1000, intrinsic_dim=2, ambient_dim=10, noise_level=0.01):
        """
        Initialize synthetic manifold dataset.
        
        Args:
            n_samples: Number of samples to generate
            intrinsic_dim: True intrinsic dimension of the manifold
            ambient_dim: Ambient dimension of the embedding space
            noise_level: Amount of noise to add
        """
        self.n_samples = n_samples
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.noise_level = noise_level
        
        self.manifold_coords = np.random.uniform(-1, 1, (n_samples, intrinsic_dim))
        
        self.data = np.zeros((n_samples, ambient_dim))
        self.data[:, :intrinsic_dim] = self.manifold_coords
        
        noise = np.random.normal(0, noise_level, (n_samples, ambient_dim))
        self.data += noise
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_synthetic_dataset(n_samples=1000, intrinsic_dim=2, ambient_dim=10, batch_size=64):
    """
    Create a DataLoader for synthetic manifold data.
    
    Args:
        n_samples: Number of samples
        intrinsic_dim: Intrinsic dimension
        ambient_dim: Ambient dimension
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = SyntheticManifoldDataset(n_samples, intrinsic_dim, ambient_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def generate_evaluation_noise(batch_size, latent_dim, device='cpu'):
    """
    Generate fixed noise for consistent evaluation.
    
    Args:
        batch_size: Number of noise vectors
        latent_dim: Dimension of latent space
        device: Device to create tensors on
    
    Returns:
        Fixed noise tensor
    """
    torch.manual_seed(42)  # For reproducibility
    noise = torch.randn(batch_size, latent_dim, device=device)
    return noise

def preprocess_images(images, target_size=(28, 28), normalize=True):
    """
    Preprocess images for model input.
    
    Args:
        images: Input images tensor
        target_size: Target image size
        normalize: Whether to normalize to [-1, 1]
    
    Returns:
        Preprocessed images
    """
    if images.shape[-2:] != target_size:
        transform = transforms.Resize(target_size)
        images = transform(images)
    
    if normalize:
        images = (images - 0.5) / 0.5
    
    return images

def create_feature_extractor_transform():
    """
    Create transform for feature extractor (VGG16) preprocessing.
    
    Returns:
        Transform function
    """
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def generate_interpolation_pairs(latent_dim, n_pairs=5, device='cpu'):
    """
    Generate pairs of latent codes for interpolation experiments.
    
    Args:
        latent_dim: Dimension of latent space
        n_pairs: Number of interpolation pairs
        device: Device to create tensors on
    
    Returns:
        List of (z1, z2) tuples
    """
    pairs = []
    torch.manual_seed(123)  # For reproducibility
    
    for i in range(n_pairs):
        z1 = torch.randn(latent_dim, device=device)
        z2 = torch.randn(latent_dim, device=device)
        pairs.append((z1, z2))
    
    return pairs

def compute_pairwise_distances(data):
    """
    Compute pairwise Euclidean distances for a batch of data.
    
    Args:
        data: Tensor of shape (n, d)
    
    Returns:
        Distance matrix of shape (n, n)
    """
    n = data.size(0)
    dist_sq = torch.sum(data ** 2, dim=1).unsqueeze(1) + \
              torch.sum(data ** 2, dim=1).unsqueeze(0) - \
              2 * torch.mm(data, data.t())
    
    distances = torch.sqrt(torch.clamp(dist_sq, min=0))
    
    return distances

def prepare_vgg_input(images):
    """
    Prepare images for VGG16 feature extraction.
    
    Args:
        images: Input images tensor (B, C, H, W)
    
    Returns:
        VGG-ready images tensor
    """
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    
    if images.min() < 0:
        images = (images + 1) / 2.0
    
    transform = create_feature_extractor_transform()
    images = transform(images)
    
    return images

class DataAugmentation:
    """
    Data augmentation utilities for training.
    """
    def __init__(self, rotation_range=10, noise_std=0.01):
        self.rotation_range = rotation_range
        self.noise_std = noise_std
    
    def add_noise(self, data):
        """Add Gaussian noise to data."""
        noise = torch.randn_like(data) * self.noise_std
        return data + noise
    
    def rotate_data(self, data):
        """Apply small random rotations (for 2D manifold data)."""
        if data.shape[-1] >= 2:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            angle_rad = np.radians(angle)
            
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                                         dtype=data.dtype, device=data.device)
            
            data_2d = data[..., :2]
            rotated_2d = torch.matmul(data_2d, rotation_matrix.t())
            
            result = data.clone()
            result[..., :2] = rotated_2d
            
            return result
        
        return data

def create_evaluation_metrics_setup(device='cpu'):
    """
    Set up evaluation metrics (FID and Inception Score).
    
    Args:
        device: Device to run metrics on
    
    Returns:
        Tuple of (FID metric, Inception Score metric)
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    fid_metric = FrechetInceptionDistance(feature=64).to(device)
    is_metric = InceptionScore().to(device)
    
    return fid_metric, is_metric
