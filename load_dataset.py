# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:26:24 2026

@author: Dayoub
"""

"""
Data loader for IMU sign language digit recognition dataset.
"""

import numpy as np

# Indices for 6-digit dataset that have valid start/end indices
six_digits_valid_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    45, 46, 63, 64, 99, 100, 224, 225, 226, 243, 244,
    261, 262, 279, 280, 297, 298, 441, 442,
    513, 514, 657, 658, 675, 676, 693, 694
]


def load_dataset_files():
    """
    Load all dataset files.
    
    Returns
    -------
    tuple
        (   
            # IMU data
            imu_2digits, imu_3digits, imu_4digits, imu_6digits,
            # Labels
            lbl_2digits, lbl_3digits, lbl_4digits, lbl_6digits,
            # Segments (start/end indices)
            seg_2digits, seg_3digits, seg_4digits, seg_6digits,
            # Special indices for 6-digit
            six_digits_valid_indices
        )
    """
    
    # Load IMU data
    imu_2digits = np.load("./imu_2digits.npy")
    imu_3digits = np.load("./imu_3digits.npy")
    imu_4digits = np.load("./imu_4digits.npy")
    imu_6digits = np.load("./imu_6digits.npy")
    
    # Load labels
    lbl_2digits = np.load("./labels_2digits.npy").astype(int)
    lbl_3digits = np.load("./labels_3digits.npy").astype(int)
    lbl_4digits = np.load("./labels_4digits.npy").astype(int)
    lbl_6digits = np.load("./labels_6digits.npy").astype(int)
    
    # Load start/end indices for each digit in sequence
    seg_2digits = np.load("./start_end_2digits.npy").astype(int)
    seg_3digits = np.load("./start_end_3digits.npy").astype(int)
    seg_4digits = np.load("./start_end_4digits.npy").astype(int)
    seg_6digits = np.load("./start_end_6digits.npy").astype(int)
    
    return (imu_2digits, imu_3digits, imu_4digits, imu_6digits, 
            lbl_2digits, lbl_3digits, lbl_4digits, lbl_6digits, 
            seg_2digits, seg_3digits, seg_4digits, seg_6digits, 
            six_digits_valid_indices)













