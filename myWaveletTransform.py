# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 09:58:47 2026

@author: Dayoub
"""

"""
Haar Wavelet Transform for IMU Sensor Data
===========================================
Input shape: (samples, time, 6, 6)
- Dimension 0: Number of samples/recordings
- Dimension 1: Time series length  
- Dimension 2: 6 features [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
- Dimension 3: 6 IMU sensors
"""

import numpy as np


def wavelet_transform(dataset, levels=2):
    """
    Apply multi-level Haar wavelet transform on IMU data.
    
    Parameters
    ----------
    dataset : np.ndarray
        Shape: (samples, time, features, sensors)
        - samples: number of data samples
        - time: signal length (must be even for each level)
        - features: 6 (acc_xyz + gyro_xyz)
        - sensors: 6 IMU sensors
        
    levels : int, default=2
        Number of decomposition levels
        
    Returns
    -------
    tuple
        (A1, D1, A2, D2, ...) approximation and detail coefficients
        Each output has shape: (samples, time//(2**level), features, sensors)
    """
    # Validate input
    if dataset.ndim != 4:
        raise ValueError(f"Expected 4D input, got {dataset.ndim}D")
    
    if dataset.shape[2] != 6 or dataset.shape[3] != 6:
        print(f"Warning: Expected (..., 6, 6) shape, got (..., {dataset.shape[2]}, {dataset.shape[3]})")
    
    results = []
    current = dataset
    
    for level in range(1, levels + 1):
        # Ensure even length
        length = (current.shape[1] // 2) * 2
        
        if length == 0:
            raise ValueError(f"Signal too short for level {level} decomposition")
        
        # Trim to even length
        current = current[:, :length, :, :]
        
        # Haar wavelet decomposition
        A = (current[:, 0::2, :, :] + current[:, 1::2, :, :]) / np.sqrt(2)
        D = (current[:, 0::2, :, :] - current[:, 1::2, :, :]) / np.sqrt(2)
        
        results.append(D)  # Store detail coefficients
        current = A       # Continue with approximation
        
        results.append(A)  # This will be reordered
    
    # Reorder to: A1, D1, A2, D2, ...
    output = []
    for level in range(levels):
        output.append(results[2*level + 1])  # Approximation A(level+1)
        output.append(results[2*level])      # Detail D(level+1)
    
    return tuple(output)


def wavelet_transform_v2(dataset):
    """
    Two-level wavelet transform (optimized for your specific use case).
    
    Parameters
    ----------
    dataset : np.ndarray
        Shape: (samples, time, 6, 6)
        
    Returns
    -------
    tuple
        (A1, D1, A2, D2)
        A1: (samples, time/2, 6, 6) - first level approximation
        D1: (samples, time/2, 6, 6) - first level detail
        A2: (samples, time/4, 6, 6) - second level approximation
        D2: (samples, time/4, 6, 6) - second level detail
    """
    # Level 1
    length1 = (dataset.shape[1] // 2) * 2
    data_even = dataset[:, :length1, :, :]
    
    A1 = (data_even[:, 0::2, :, :] + data_even[:, 1::2, :, :]) / np.sqrt(2)
    D1 = (data_even[:, 0::2, :, :] - data_even[:, 1::2, :, :]) / np.sqrt(2)
    
    # Level 2
    length2 = (A1.shape[1] // 2) * 2
    A1_even = A1[:, :length2, :, :]
    
    A2 = (A1_even[:, 0::2, :, :] + A1_even[:, 1::2, :, :]) / np.sqrt(2)
    D2 = (A1_even[:, 0::2, :, :] - A1_even[:, 1::2, :, :]) / np.sqrt(2)
    
    return A1, D1, A2, D2






