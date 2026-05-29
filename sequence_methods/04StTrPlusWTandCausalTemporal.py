# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:12:51 2026

@author: Dayoub
"""

"""
Streaming Transformer with Causal Temporal Reduction for IMU Digit Recognition

This model uses wavelet transform 
and adds a causal temporal convolution layer to reduce sequence length
before the Transformer encoder, improving efficiency for long sequences.

This model uses a causal Transformer encoder with CTC loss for end-to-end
recognition of multi-digit sign language sequences from IMU sensor data.

Input shape: (batch, time, 6 IMU sensors, 6 features)
Output: Sequence of digit class indices (0-38)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import editdistance
from sklearn.model_selection import train_test_split
from data_loader import load_dataset_files
from myWaveletTransform import wavelet_transform_v2

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model and training configuration."""
    # Data dimensions
    MAX_LEN = 138          # Maximum sequence length (after padding)
    NUM_IMU = 6            # Number of IMU sensors
    IMU_DIM = 6            # Features per sensor (acc_xyz + gyro_xyz)
    NUM_CLASSES = 39       # Number of digit classes
    BLANK_INDEX = 39       # CTC blank token index
    
    # Model architecture
    D_MODEL = 128          # Transformer embedding dimension
    NUM_HEADS = 4          # Number of attention heads
    FFN_DIM = 256          # Feed-forward network dimension
    KERNEL_SIZE = 3        # causal temporal kernel size
    STRIDE = 3             # causal temporal stride
    DROPOUT_RATE = 0.1
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    TRAIN_SPLIT = 0.8
    BEAM_WIDTH = 10        # For beam search decoding
    
    # Unique digit classes (mapped to indices 0-38)
    UNIQUE_CLASSES = [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 30, 40, 50, 60, 70,
        100, 200, 300, 400, 500, 600, 700,
        1000, 2000, 3000, 4000, 5000, 6000, 7000
    ]
    
    REDUCED_MAX_LEN = MAX_LEN // STRIDE + 1


# ============================================================================
# Model Definition
# ============================================================================

class StreamingTransformer(tf.keras.Model):
    """
    Streaming Transformer with causal masking for sequence recognition.
    
    Processes IMU data in a streaming fashion where each time step can only
    attend to previous and current positions.
    """
    
    def __init__(self, config=Config):
        super().__init__()
        
        # Flatten and project IMU features: (6, 6) -> 36 -> D_MODEL
        self.spatial = tf.keras.Sequential([
            layers.Dense(config.D_MODEL),
            layers.LayerNormalization()
        ])
        
        # Positional encoding for sequence order
        self.pos_emb = layers.Embedding(
            input_dim=config.REDUCED_MAX_LEN, 
            output_dim=config.D_MODEL
        )
        
        # Transformer self-attention with causal mask
        self.attn = layers.MultiHeadAttention(
            num_heads=config.NUM_HEADS, 
            key_dim=config.D_MODEL
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(config.FFN_DIM, activation='relu'),
            layers.Dense(config.D_MODEL)
        ])
        
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        # Causal temporal reduction (reduces sequence length by factor of STRIDE)
        self.temporal_reduction = tf.keras.Sequential([
            layers.Conv1D(
                filters=config.D_MODEL, 
                kernel_size=config.KERNEL_SIZE,
                strides=config.STRIDE,
                padding='causal',
                activation='relu'
            ),
            layers.LayerNormalization(),
            layers.Dropout(config.DROPOUT_RATE)
        ])
        
        # Output layer (NUM_CLASSES + 1 for CTC blank)
        self.classifier = layers.Dense(
            config.NUM_CLASSES + 1, 
            activation='softmax'
        )
    
    def call(self, x, training=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, NUM_IMU, IMU_DIM)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch, time, NUM_CLASSES + 1)
        """
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        # Flatten sensor data: (B, T, 6, 6) -> (B, T, 36)
        x = tf.reshape(x, (batch_size, time_steps, Config.NUM_IMU * Config.IMU_DIM))
        x = self.spatial(x)
        
        x = self.temporal_reduction(x)
        
        reduced_time  = tf.shape(x)[1]
        
        # Add positional encoding
        positions = tf.range(reduced_time)
        x = x + self.pos_emb(positions)
        
        # Causal mask (prevents attending to future positions)
        causal_mask = tf.linalg.band_part(tf.ones((reduced_time , reduced_time )), -1, 0)
        
        # Self-attention with residual connection
        attn_out = self.attn(x, x, attention_mask=causal_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return self.classifier(x)
    
    def get_reduced_length(self, original_length):
        """Calculate reduced sequence length after temporal reduction."""
        return (original_length - self.config.KERNEL_SIZE) // self.config.STRIDE + 1


# ============================================================================
# Data Processing Utilities
# ============================================================================

def load_and_prepare_data():
    """
    Load dataset and convert labels to class indices.
    
    Returns:
        X: List of IMU sequences
        Y: List of encoded label sequences
    """
    # Load raw data
    (imu_2d, imu_3d, imu_4d, imu_6d,
     lbl_2d, lbl_3d, lbl_4d, lbl_6d,
     *_) = load_dataset_files()
    
    #wavelet transform
    imu_2d,*_ = wavelet_transform_v2(imu_2d)
    imu_3d,*_ = wavelet_transform_v2(imu_3d)
    imu_4d,*_ = wavelet_transform_v2(imu_4d)
    imu_6d,*_ = wavelet_transform_v2(imu_6d)
    
    # Combine all sequences
    X = list(imu_2d) + list(imu_3d) + list(imu_4d) + list(imu_6d)
    
    # Encode labels to indices
    Y = []
    for labels in [lbl_2d, lbl_3d, lbl_4d, lbl_6d]:
        for seq in labels:
            encoded = np.zeros(len(seq), dtype=int)
            for i, lbl in enumerate(seq):
                encoded[i] = np.where(lbl == Config.UNIQUE_CLASSES)[0][0]
            Y.append(encoded)
    
    return X, Y


def pad_sequences(X, y):
    """
    Pad sequences to fixed length for batch processing.
    
    Args:
        X: List of IMU sequences
        y: List of label sequences
        
    Returns:
        X_padded: Padded IMU data
        y_padded: Padded labels (-1 for padding)
        input_len: Original lengths of X sequences
        label_len: Original lengths of y sequences
    """
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, maxlen=Config.MAX_LEN, padding='post', dtype='float32'
    )
    
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, padding='post', value=-1
    )
    
    input_len = np.array([len(seq) for seq in X])
    label_len = np.array([len(seq) for seq in y])
    
    # Add dimension for CTC compatibility
    input_len = np.expand_dims(input_len, axis=-1)
    label_len = np.expand_dims(label_len, axis=-1)
    
    return X_padded, y_padded, input_len, label_len


def create_dataset(X, y, training=True):
    """
    Create tf.data.Dataset for training or evaluation.
    
    Args:
        X: List of IMU sequences
        y: List of label sequences
        training: If True, enables shuffling
        
    Returns:
        tf.data.Dataset yielding (X, y, input_len, label_len) batches
    """
    X_pad, y_pad, input_len, label_len = pad_sequences(X, y)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_pad, y_pad, input_len, label_len)
    )
    
    if training:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.batch(Config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# CTC Loss and Training
# ============================================================================

def ctc_loss(y_true, y_pred, input_len, label_len):
    """Compute CTC batch loss."""
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

@tf.function
def train_step(model, optimizer, x, y, input_len, label_len):
    """Single training step with gradient computation."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = ctc_loss(y, logits, input_len, label_len)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return tf.reduce_mean(loss)


def train_model(model, optimizer, train_dataset, epochs=None):
    """Train the model for specified epochs."""
    epochs = epochs or Config.EPOCHS
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = []
        
        for step, (x, y, input_len, label_len) in enumerate(train_dataset):
            reduced_len = tf.cast(tf.math.ceil(input_len / Config.STRIDE), tf.int32)
            loss = train_step(model, optimizer, x, y, reduced_len, label_len)
            epoch_losses.append(loss.numpy())
            
            if step % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"  Step {step}, Loss: {loss.numpy():.4f} (avg: {avg_loss:.4f})")
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")


# ============================================================================
# Decoding and Evaluation
# ============================================================================

def greedy_decode(logits):
    """Greedy CTC decoding."""
    input_len = np.ones(logits.shape[0]) * logits.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(
        logits, input_length=input_len, greedy=True
    )
    return decoded[0].numpy()


def beam_search_decode(logits, beam_width=None):
    """Beam search CTC decoding."""
    beam_width = beam_width or Config.BEAM_WIDTH
    input_len = np.ones(logits.shape[0]) * logits.shape[1]
    
    decoded, _ = tf.keras.backend.ctc_decode(
        logits, input_length=input_len, greedy=False,
        beam_width=beam_width, top_paths=1
    )
    return decoded[0].numpy()


def clean_sequence(seq):
    """Remove padding values (-1) from decoded sequence."""
    return [s for s in seq if s != -1]


def collapse_repeats(seq):
    """Remove consecutive duplicate tokens (CTC collapse)."""
    if not seq:
        return []
    result = [seq[0]]
    for s in seq[1:]:
        if s != result[-1]:
            result.append(s)
    return result


def evaluate_wer(model, dataset):
    """
    Evaluate Word Error Rate on test dataset.
    
    Args:
        model: Trained model
        dataset: Test dataset
        
    Returns:
        wer: Word Error Rate (0-1, lower is better)
    """
    total_distance = 0
    total_length = 0
    dataset_size = len(dataset)
    
    for count, (x, y_true, input_len, label_len) in enumerate(dataset):
        logits = model(x, training=False)
        y_pred = beam_search_decode(logits)
        
        for i in range(len(y_pred)):
            # Decode prediction
            pred_seq = collapse_repeats(clean_sequence(y_pred[i]))
            
            # Get ground truth
            true_seq = y_true[i].numpy()
            true_seq = true_seq[:label_len[i][0]]
            
            # Compute edit distance
            total_distance += editdistance.eval(pred_seq, true_seq)
            total_length += len(true_seq)
        
        # Progress indicator
        progress = (count + 1) * 100 / dataset_size
        print(f"\rEvaluating: {progress:.1f}%", end="")
    
    print()
    wer = total_distance / total_length if total_length > 0 else 1.0
    print(f"Word Error Rate (WER): {wer:.4f}")
    
    return wer


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the complete training and evaluation pipeline."""
    print("=" * 60)
    print("Streaming Transformer with Causal Temporal Reduction")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    X, Y = load_and_prepare_data()
    print(f"      Loaded {len(X)} sequences")
    
    # Split data
    print("\n[2/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, 
        train_size=Config.TRAIN_SPLIT,
        random_state=42,
        shuffle=True
    )
    print(f"      Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"      Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create datasets
    print("\n[3/5] Creating TensorFlow datasets...")
    train_dataset = create_dataset(X_train, y_train, training=True)
    test_dataset = create_dataset(X_test, y_test, training=False)
    
    # Build model
    print("\n[4/5] Building model...")
    model = StreamingTransformer()
    optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE)
    
    
    # Train
    print("\n[5/5] Training...")
    train_model(model, optimizer, train_dataset)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    wer = evaluate_wer(model, test_dataset)
    
    return model, wer


if __name__ == "__main__":
    main()


