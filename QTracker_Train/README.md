# QTracker Training System

This directory contains the scripts and utilities for creating and training the QTracker system. The training system is designed to work both on the Rivanna HPC cluster and in local environments.

## Directory Structure

- `Python_Files/`: Core training scripts and utilities
- `Training_Jobscripts/`: Job submission scripts for Rivanna
- `Models/`: Trained model outputs and checkpoints
- `Configs/`: Configuration files for different training scenarios

## Quick Start

### On Rivanna HPC

1. Connect to Rivanna:
```bash
ssh your-username@rivanna.hpc.virginia.edu
```

2. Navigate to the QTracker_Train directory:
```bash
cd path/to/QTracker_Train
```

3. Submit the training job:
```bash
source job_submission.sh
```

### Local Environment

1. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run individual training scripts:
```bash
python Python_Files/script.py [options]
```

## Training Scripts

### Main Components

- `train_model.py`: Main training script
- `data_loader.py`: Data loading and preprocessing
- `model_architecture.py`: Neural network architecture definitions
- `evaluation.py`: Model evaluation utilities

### Usage Examples

1. Basic training:
```bash
python Python_Files/train_model.py --config configs/basic_config.yaml
```

2. Custom training:
```bash
python Python_Files/train_model.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

## Job Submission (Rivanna)

### Batch Jobs

Each script in `Training_Jobscripts/` can be run separately using:
```bash
sbatch Training_Jobscripts/script_name.sh
```

### Job Monitoring

- Check job status:
```bash
squeue -u your-username
```

- View job output:
```bash
cat slurm-<job_id>.out
```

## Configuration

### Environment Variables

- `QTRAIN_DATA_DIR`: Path to training data
- `QTRAIN_MODEL_DIR`: Path to save models
- `QTRAIN_LOG_DIR`: Path for training logs

### Configuration Files

Configuration files in the `Configs/` directory control:
- Model architecture
- Training parameters
- Data preprocessing
- Evaluation metrics

## Output

Trained models and checkpoints are saved in the `Models/` directory with the following structure:
```
Models/
├── checkpoints/
│   └── epoch_*.pt
├── final_models/
│   └── model_*.pt
└── logs/
    └── training_*.log
```

## Troubleshooting

Common issues and solutions:
1. Memory errors: Reduce batch size
2. GPU errors: Check CUDA compatibility
3. Data loading errors: Verify data paths

## Contributing

When adding new training scripts:
1. Follow the existing code structure
2. Add appropriate documentation
3. Include configuration templates
4. Update this README if necessary

## Contact

For issues and questions:
- Open an issue in the repository
- Contact the development team
- Check the troubleshooting guide
