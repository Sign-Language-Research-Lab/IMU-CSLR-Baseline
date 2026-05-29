# IMU-CSLR Baselines

Reproducible benchmark implementations for inertial sensor–based continous sign language recognition using IMU data (6 sensors × 6 features).

## Methods

### Sliding Window Methods (with transition indices)
- **ResNet + Sliding Window** (`sliding_window_methods/resnet_sliding_window.py`)
- **Transformer Core + Sliding Window** (`sliding_window_methods/transformer_sliding.py`)

### Sequence Methods (without transition indices)
- **Streaming Transformer + CTC** (`sequence_methods/transformer_ctc_baseline.py`)
- **Streaming Transformer + wavelet transform + CTC** (`sequence_methods/bilstm_ctc.py`)
- *More methods coming soon...*

## Data Format
numpy array

## Citation

If you use this code in your research, please cite:

```bibtex
@software{imu_slr_baselines,
  author = {ayoub_parizi},
  title = {IMU-SLR Baselines: Benchmark Implementations for IMU-based Sign Language Recognition},
  year = {2026},
  url = {https://github.com/ayoubprz/IMU-CSLR}
}
