# Data Converters

This directory contains utilities for converting legacy data formats to the universal RUS (ROOT Universal Standard) format. These converters are maintained for backward compatibility and will eventually be deprecated as all data migrates to the RUS format.

## ⚠️ Deprecation Notice

These converters are temporary solutions for data migration. New data should be written directly in the RUS format. The converters will be removed once all legacy data has been migrated.

## Available Converters

- `TrackQAconvert.py`: Converts TrackQA format to RUS format
- `LegacyConvert.py`: Converts legacy ROOT formats to RUS format
- `FormatValidator.py`: Validates converted data against RUS specifications

## Installation

1. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. Verify ROOT installation:
```bash
root-config --version
```

## Usage

### Command Line Interface

#### Basic Conversion
```bash
python TrackQAconvert.py input_trackQA.root output_file.root
```

#### Advanced Options
```bash
# Custom compression level (0-9)
python TrackQAconvert.py input_trackQA.root output_file.root --compression 7

# Batch conversion
python TrackQAconvert.py --batch input_directory/ output_directory/

# Verbose output
python TrackQAconvert.py input_trackQA.root output_file.root --verbose

# Get help
python TrackQAconvert.py --help
```

### Programmatic Usage

#### Using TrackQAConverter
```python
from TrackQAconvert import TrackQAConverter

# Initialize converter
converter = TrackQAConverter(
    input_file="input.root",
    output_file="output.root",
    compression_level=7
)

# Convert data
converter.convert()

# Validate conversion
converter.validate()
```

#### Using LegacyConverter
```python
from LegacyConvert import LegacyConverter

converter = LegacyConverter(
    input_file="legacy_data.root",
    output_file="rus_data.root",
    format_version="1.0"
)
converter.convert()
```

## Data Format Specifications

### RUS Format
- Tree structure: `RUSData`
- Required branches:
  - `EventID`
  - `TrackData`
  - `Metadata`
- Compression: ZLIB (default level 7)

### Legacy Formats
- TrackQA format (deprecated)
- Old ROOT format (deprecated)
- Custom binary format (deprecated)

## Validation

### Automatic Validation
```bash
python FormatValidator.py converted_file.root
```

### Manual Validation
```python
from FormatValidator import RUSValidator

validator = RUSValidator("converted_file.root")
validation_report = validator.validate()
print(validation_report)
```

## Troubleshooting

Common issues and solutions:

1. **Memory Errors**
   - Use batch processing for large files
   - Increase system swap space
   - Process in chunks

2. **Format Errors**
   - Verify input file format
   - Check ROOT version compatibility
   - Use verbose mode for detailed error messages

3. **Performance Issues**
   - Adjust compression level
   - Use batch processing
   - Optimize system resources

## Contributing

When adding new converters:
1. Follow the existing code structure
2. Include validation tests
3. Document the legacy format
4. Update this README

## Contact

For issues and questions:
- Open an issue in the repository
- Contact the data management team
- Check the troubleshooting guide
