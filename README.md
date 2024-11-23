# Arma 3 Satellite Mask/Shape Blob Generator

## Features

- **Random Blob Generation**: Creates organic natural looking blob shapes with customizable properties.
- **Color Detection**: Blobs are placed within areas of a target color in the input image.
- **Multiple Blob Types**: Supports multiple blob types with individual settings.
- **Shape Customization**: Control blob size, shape smoothness, elongation, and "chaotic-ness".

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - `numpy`
  - `scipy`
  - `Pillow`
  - `opencv-python`

## Usage

The script can be run directly from the command line. It will prompt you for input and output image paths and configuration settings. A config.json file is created to store configuration settings, make sure to back it up if it's a setting you like and would want to use in the future. The configuration is divided into general settings and blob type configurations. You must have a black background to the image if you are working with an image that is not entirely the target color.

Command Line Arguments
- "-n" or "--no_prompt" will skip the blob configuration step of the script.
- "-c" or "--config" will allow you to set the path to a saved config file so you don't have to re-input values for another blob variation.

General Settings
- input_path: Path to the input image, no quotation marks.
- output_path: Path where the output image will be saved, no quotation marks.
- target_color: RGB values of the color you want the blobs to be contained in. Should be the same color as the one in your layers.cfg you exported with in Terrain Builder.
- color_tolerance: Tolerance for color detection (0-255).
- buffer_size: Size of the buffer zone around target areas where blobs can be placed (in pixels). 

Blob Types
Each blob type has its own set of configurable parameters:

- blob_color: RGB values of the blob color. Based on your layers.cfg if using for Satellite Mask.
- average_blob_size: Average size of the blobs (in pixels).
- size_variance: Variance in blob size.
- max_blob_size: Maximum blob size.
- min_blob_size: Minimum blob size.
- max_num_blobs: Maximum number of blobs to generate.
- min_num_blobs: Minimum number of blobs to generate.
- randomness: Controls the irregularity of blob shapes, lower numbers resemble circles while higher numbers resemble... blobs! (0.0 - 1.0).
- min_elongation: Minimum elongation factor (>0.0).
- max_elongation: Maximum elongation factor (>0.0).
- curved_shapes: True for smooth blobs, False for polygonal blobs.
- avoid_overlap: True to avoid overlapping existing blobs, False to allow overlap.
- num_points_per_blob: Number of points defining each blob shape.

## Known Issues

- Large amount of blobs to generate with multiple blobs will take a LONG time. Reducing randomness, disabling smooth lines, and the number of points will speed up the process.
- Probably won't work very well with users with 8GB memory or less.


If you discover any issues with the script or have some suggestions, please put them in the issues portion of this repository and when I find the time I'll add them, or someone cool will add it themselves!
