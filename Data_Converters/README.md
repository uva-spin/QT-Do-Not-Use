## Converters

This directory contains converters for converting older data formats that previous data was written in to the universal RUS format. These converters will eventually become deprecated. The sole reason for this directory is to have an on-the-go converter.  

### Example Usage 

```
# Basic usage
python TrackQAconvert.py input_trackQA.root output_file.root

# With custom compression level
python TrackQAconvert.py input_trackQA.root output_file.root --compression 7

# Get help
python TrackQAconvert.py --help
```

### Additional Usage

If you want to call either of the classes which convert the data in a script of your own, you can do  
```
from TrackQAconvert import TrackQAConverter

converter = TrackQAConverter("input.root", "output.root")
converter.convert()
```