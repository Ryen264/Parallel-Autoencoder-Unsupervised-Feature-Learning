"""
Data Loading and Preprocessing:
+ Create a CIFAR10 Dataset class to handle data loading 
+ Read CIFAR-10 binary files (5 training batches + 1 test batch)
+ Parse the binary format: 1 byte label + 3,072 bytes image per record 
+ Convert uint8 pixel values [0, 255] to float [0, 1] for normalization
+ Implement batch generation for training
+ Add data shuffling capability 
+ Organize train images (50,000), test images (10,000), and their labels in memory
"""

import numpy as np
import os
from typing import Tuple


class CIFAR10Dataset:
    """Dataset class for loading and preprocessing CIFAR-10 binary data."""
    
    def __init__(self, data_dir: str = r".\data\cifar-10-batches-bin"):
        """
        Initialize CIFAR-10 dataset loader.
        
        Args:
            data_dir: Path to directory containing CIFAR-10 binary files
        """
        self.data_dir = data_dir
        self.num_classes = 10
        self.image_shape = (32, 32, 3)  # Height, Width, Channels

        self.image_size = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        self.record_size = 1 + self.image_size  # 1 byte label + image bytes
        
        # Load all data into memory
        self.train_images, self.train_labels = self._load_training_data()
        self.test_images, self.test_labels = self._load_test_data()

        self.num_train_samples = self.train_images.shape[0]
        self.num_test_samples = self.test_images.shape[0]
        
        # Initialize batch generation state
        self.train_indices = np.arange(self.num_train_samples)
        self.current_index = 0
        
    def _read_binary_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a single CIFAR-10 binary file.
        
        Args:
            filepath: Path to binary file
            
        Returns:
            Tuple of (images, labels) as numpy arrays
        """
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        num_records = len(data) // self.record_size
        
        # Reshape and separate labels from images
        data = data.reshape(num_records, self.record_size)
        labels = data[:, 0].astype(np.int32)
        images = data[:, 1:].astype(np.float32)
        
        # Normalize pixel values from [0, 255] to [0, 1]
        images = images / 255.0
        
        # Reshape images to (N, 32, 32, 3) - RGB format
        images = images.reshape(num_records, self.image_shape[2], self.image_shape[0], self.image_shape[1])
        images = images.transpose(0, 2, 3, 1)  # Convert to (N, H, W, C)
        
        return images, labels
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all training batch files.
        
        Returns:
            Tuple of (train_images, train_labels)
        """
        all_images = []
        all_labels = []
        num_batches = 5
        
        # Load training batches
        for i in range(1, num_batches + 1):
            batch_file = os.path.join(self.data_dir, f"data_batch_{i}.bin")
            images, labels = self._read_binary_file(batch_file)
            all_images.append(images)
            all_labels.append(labels)
        
        # Concatenate all batches
        train_images = np.concatenate(all_images, axis=0)
        train_labels = np.concatenate(all_labels, axis=0)
        
        return train_images, train_labels
    
    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test batch file.
        
        Returns:
            Tuple of (test_images, test_labels)
        """
        test_file = os.path.join(self.data_dir, "test_batch.bin")
        test_images, test_labels = self._read_binary_file(test_file)
        
        return test_images, test_labels
    
    def shuffle_training_data(self):
        """Shuffle the training data indices for random batch generation."""
        np.random.shuffle(self.train_indices)
        self.current_index = 0
    
    def get_batch(self, batch_size: int, shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of training data.
        
        Args:
            batch_size: Number of samples in the batch
            shuffle: Whether to shuffle data when starting a new epoch
            
        Returns:
            Tuple of (batch_images, batch_labels)
        """
        # Check if we need to start a new epoch
        if self.current_index + batch_size > self.num_train_samples:
            if shuffle:
                self.shuffle_training_data()
            else:
                self.current_index = 0
        
        # Get batch indices
        batch_indices = self.train_indices[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size
        
        # Return batch
        batch_images = self.train_images[batch_indices]
        batch_labels = self.train_labels[batch_indices]
        
        return batch_images, batch_labels
    
    def get_all_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all training data.
        
        Returns:
            Tuple of (train_images, train_labels)
        """
        return self.train_images, self.train_labels
    
    def get_all_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all test data.
        
        Returns:
            Tuple of (test_images, test_labels)
        """
        return self.test_images, self.test_labels
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_train_samples': self.num_train_samples,
            'num_test_samples': self.num_test_samples,
            'image_shape': self.image_shape,
            'num_classes': self.num_classes,
            'train_images_shape': self.train_images.shape,
            'train_labels_shape': self.train_labels.shape,
            'test_images_shape': self.test_images.shape,
            'test_labels_shape': self.test_labels.shape
        }


# Example usage
if __name__ == "__main__":   
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    dataset = CIFAR10Dataset()
    print("✓ Dataset loaded successfully!\n")
    
    # Verification 1: Check number of training images
    train_images, train_labels = dataset.get_all_training_data()
    print(f"✓ Training images: {train_images.shape[0]:,} samples")
    assert train_images.shape[0] == 50000, f"Expected 50,000 training images, got {train_images.shape[0]}"
    print(f"  - Shape: {train_images.shape}")
    print(f"  - Labels shape: {train_labels.shape}")
    
    # Verification 2: Check number of test images
    test_images, test_labels = dataset.get_all_test_data()
    print(f"\n✓ Test images: {test_images.shape[0]:,} samples")
    assert test_images.shape[0] == 10000, f"Expected 10,000 test images, got {test_images.shape[0]}"
    print(f"  - Shape: {test_images.shape}")
    print(f"  - Labels shape: {test_labels.shape}")
    
    # Verification 3: Check normalization to [0, 1]
    train_min, train_max = train_images.min(), train_images.max()
    test_min, test_max = test_images.min(), test_images.max()
    print(f"\n✓ Preprocessing - Normalized to [0, 1]:")
    print(f"  - Training data range: [{train_min:.2f}, {train_max:.2f}]")
    print(f"  - Test data range: [{test_min:.2f}, {test_max:.2f}]")
    assert 0.0 <= train_min and train_max <= 1.0, "Training data not properly normalized"
    assert 0.0 <= test_min and test_max <= 1.0, "Test data not properly normalized"
    
    # Additional information
    print(f"\n" + "="*60)
    print("Additional Dataset Information")
    print("="*60)
    info = dataset.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example: Test batch generation
    print(f"\n" + "="*60)
    print("Batch Generation Test")
    print("="*60)
    batch_images, batch_labels = dataset.get_batch(batch_size=128, shuffle=True)
    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Batch pixel range: [{batch_images.min():.2f}, {batch_images.max():.2f}]")
    print(f"  Sample labels: {batch_labels[:10]}")
