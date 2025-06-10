# Monte Carlo Generation and Resampling

This directory contains tools for generating, processing, and resampling Monte Carlo (MC) data for E906 and E1039 experiments. The code handles data splitting, resampling, and preparation for training and validation datasets.

## ⚠️ Important Note

This code requires a valid installation of PyRoot, which is not available on Rivanna. You must run this code on a machine with PyRoot installed.

## Directory Structure

- `MC/`: Directory for input MC data
- `Merged_MC/`: Directory for processed and merged MC data
- `Data_Prep.py`: Main script for data preparation
- `resampling/`: Resampling utilities
- `validation/`: Validation tools and scripts

## Prerequisites

1. PyRoot installation:
```bash
# Check PyRoot installation
python -c "import ROOT; print(ROOT.__version__)"
```

2. Required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your generated MC data in the `MC/` directory following the naming convention:
```
MC/
├── e906_data_*.root
├── e1039_data_*.root
└── metadata.json
```

2. Run the data preparation script:
```bash
python Data_Prep.py
```

### Advanced Usage

#### Custom Resampling
```bash
python Data_Prep.py --mass-range 2.0 4.0 --bin-size 0.1
```

#### Specific Experiments
```bash
python Data_Prep.py --experiment e906
```

#### Output Options
```bash
python Data_Prep.py --output-dir custom_output/ --format hdf5
```

## Data Processing Pipeline

1. **Input Validation**
   - Check file format
   - Verify metadata
   - Validate mass distributions

2. **Resampling**
   - Flatten mass distribution
   - Apply selection cuts
   - Generate balanced datasets

3. **Output Generation**
   - Split into training/validation
   - Apply final formatting
   - Generate metadata

## Output Structure

```
Merged_MC/
├── training/
│   ├── e906_training.root
│   └── e1039_training.root
├── validation/
│   ├── e906_validation.root
│   └── e1039_validation.root
└── metadata/
    ├── mass_distributions.pdf
    └── processing_log.txt
```

## Configuration

### Mass Range Settings
- E906: 2.0 - 4.0 GeV/c²
- E1039: 2.0 - 4.0 GeV/c²

### Resampling Parameters
- Bin size: 0.1 GeV/c²
- Target distribution: Flat
- Minimum events per bin: 1000

## Troubleshooting

Common issues and solutions:

1. **PyRoot Errors**
   - Verify PyRoot installation
   - Check ROOT version compatibility
   - Ensure proper environment setup

2. **Memory Issues**
   - Process data in chunks
   - Use appropriate batch sizes
   - Monitor system resources

3. **Data Quality**
   - Validate input data
   - Check mass distributions
   - Verify output statistics

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate documentation
3. Include validation tests
4. Update this README

## Contact

For issues and questions:
- Open an issue in the repository
- Contact the MC generation team
- Check the troubleshooting guide
