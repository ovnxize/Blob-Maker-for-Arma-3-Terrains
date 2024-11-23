import math
import os
import json
import random
import numpy as np
from scipy.interpolate import splprep, splev
from PIL import Image, ImageDraw
import cv2
import argparse  # Import argparse for command-line arguments

# Disable decompression bomb protection (allow large images)
Image.MAX_IMAGE_PIXELS = None  # Disable limit

# Use argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Random Blob Generator')
parser.add_argument('--config', '-c', type=str, default='config.json', help='Path to the configuration file')
parser.add_argument('--no-prompt', '-n', action='store_true', help='Skip configuration prompts and use existing config')
args = parser.parse_args()

# Default configuration (used if keys are missing in the provided config file)
default_config = {
    "buffer_size": 150,  # Buffer zone size in pixels
    "num_points_per_blob": 100,  # Number of points defining each blob
    "input_path": "path_to_input_image.png",  # Default input path
    "output_path": "final_blob_image_no_blur.png",  # Default output path
    "target_color": [0, 255, 0],  # Default color to detect (green)
    "color_tolerance": 50,  # Tolerance for color detection
    "blob_types": [
        {
            "name": "Type 1",
            "blob_color": [120, 120, 120],  # Grey color [R, G, B]
            "average_blob_size": 30,  # Average radius of blobs
            "size_variance": 10,      # Variance in blob size
            "max_blob_size": 50,  # Maximum radius of blobs
            "min_blob_size": 10,  # Minimum radius of blobs
            "max_num_blobs": 200,  # Maximum number of blobs
            "min_num_blobs": 50,  # Minimum number of blobs
            "randomness": 0.3,  # Randomness factor for blob shape (0.0 - 1.0)
            "min_elongation": 0.9,  # Minimum elongation factor
            "max_elongation": 1.1,   # Maximum elongation factor
            "curved_shapes": True,    # Whether to use curved shapes
            "avoid_overlap": False    # Whether to avoid overlapping other blobs
        }
    ]
}

# Configuration file path from command-line argument
config_file = args.config

# Load or create config file
def load_config():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            try:
                user_config = json.load(f)
            except json.JSONDecodeError:
                print("Error decoding the configuration file. Using default configuration.")
                user_config = {}
        # Update user_config with any missing keys from default_config
        updated = False
        for key, value in default_config.items():
            if key not in user_config:
                user_config[key] = value
                updated = True
                print(f"Missing key '{key}' found. Adding default value: {value}")
        if 'blob_types' in user_config:
            # Update blob_types
            for i, blob_type in enumerate(user_config['blob_types']):
                default_blob_type = default_config['blob_types'][0]
                for bt_key, bt_value in default_blob_type.items():
                    if bt_key not in blob_type:
                        blob_type[bt_key] = bt_value
                        updated = True
                        print(f"Missing key '{bt_key}' in blob type {i+1}. Adding default value: {bt_value}")
                    # Remove 'shape_smoothness' if present
                    if 'shape_smoothness' in blob_type:
                        del blob_type['shape_smoothness']
                        updated = True
                        print(f"Removed 'shape_smoothness' from blob type {i+1}.")
        if updated:
            print("Updating configuration with missing default values...")
            save_config(user_config)
        return user_config
    else:
        print(f"Configuration file '{config_file}' not found. Creating with default configuration.")
        save_config(default_config)
        return default_config

def save_config(config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file}.\n")

def display_configuration(config):
    print("\nCurrent Configuration:")
    for key, default_value in default_config.items():
        user_value = config.get(key, default_value)
        if key == "blob_types":
            print("Blob Types:")
            for i, blob_type in enumerate(user_value):
                print(f"  Type {i + 1}:")
                for bt_key, bt_value in blob_type.items():
                    if bt_key == "blob_color":
                        bt_value_str = ",".join(map(str, bt_value))
                        print(f"    {bt_key.replace('_', ' ').title()}: {bt_value_str}")
                    else:
                        print(f"    {bt_key.replace('_', ' ').title()}: {bt_value}")
        elif key in ["input_path", "output_path"]:
            print(f"{key.replace('_', ' ').title()}: {user_value}")
        elif key == "target_color":
            user_value_str = ",".join(map(str, user_value))
            print(f"{key.replace('_', ' ').title()}: {user_value_str}")
        else:
            print(f"{key.replace('_', ' ').title()}: {user_value}")
    print()

def get_user_input(config):
    # Display current configuration
    display_configuration(config)

    # Get image paths and configuration settings from user input
    new_input = input(f"Enter input image path (default '{config['input_path']}'): ").strip()
    if new_input:
        config['input_path'] = new_input

    new_output = input(f"Enter output image path (default '{config['output_path']}'): ").strip()
    if new_output:
        config['output_path'] = new_output

    # Get target color and tolerance
    new_target_color = input(
        f"Enter target color to detect as R,G,B (default {','.join(map(str, config['target_color']))}): ").strip()
    if new_target_color:
        try:
            target_color = list(map(int, new_target_color.split(",")))
            if len(target_color) != 3 or not all(0 <= val <= 255 for val in target_color):
                raise ValueError
            config['target_color'] = target_color
        except:
            print("Invalid target color input. Keeping default.")

    new_color_tolerance = input(f"Enter color tolerance (default {config['color_tolerance']}): ").strip()
    if new_color_tolerance:
        try:
            config['color_tolerance'] = int(new_color_tolerance)
        except ValueError:
            print("Invalid input. Keeping default color_tolerance.")

    # Get buffer size
    new_buffer_size = input(f"Enter buffer size in pixels (default {config['buffer_size']}): ").strip()
    if new_buffer_size:
        try:
            config['buffer_size'] = int(new_buffer_size)
        except ValueError:
            print("Invalid input. Keeping default buffer_size.")

    # Get number of blob types
    num_blob_types = input(f"Enter number of blob types (default {len(config['blob_types'])}): ").strip()
    if num_blob_types:
        try:
            num_blob_types = int(num_blob_types)
            if num_blob_types <= 0:
                raise ValueError
        except ValueError:
            print("Invalid input. Keeping default number of blob types.")
            num_blob_types = len(config['blob_types'])
    else:
        num_blob_types = len(config['blob_types'])

    # Collect blob type configurations
    blob_types = []
    for i in range(num_blob_types):
        print(f"\nConfiguring Blob Type {i + 1}:")
        if i < len(config['blob_types']):
            blob_type = config['blob_types'][i]
        else:
            blob_type = default_config['blob_types'][0].copy()
            blob_type['name'] = f"Type {i + 1}"
        # Blob color
        new_blob_color = input(
            f"Enter blob color as R,G,B (default {','.join(map(str, blob_type['blob_color']))}): ").strip()
        if new_blob_color:
            try:
                blob_color = list(map(int, new_blob_color.split(",")))
                if len(blob_color) != 3 or not all(0 <= val <= 255 for val in blob_color):
                    raise ValueError
                blob_type['blob_color'] = blob_color
            except:
                print("Invalid blob color input. Keeping default.")

        # Average blob size
        new_average_blob_size = input(f"Enter average blob size (default {blob_type['average_blob_size']}): ").strip()
        if new_average_blob_size:
            try:
                blob_type['average_blob_size'] = float(new_average_blob_size)
            except ValueError:
                print("Invalid input. Keeping default average_blob_size.")

        # Size variance
        new_size_variance = input(f"Enter size variance (default {blob_type['size_variance']}): ").strip()
        if new_size_variance:
            try:
                blob_type['size_variance'] = float(new_size_variance)
            except ValueError:
                print("Invalid input. Keeping default size_variance.")

        # Max and Min blob size
        new_max_blob_size = input(f"Enter max blob size (default {blob_type['max_blob_size']}): ").strip()
        if new_max_blob_size:
            try:
                blob_type['max_blob_size'] = float(new_max_blob_size)
            except ValueError:
                print("Invalid input. Keeping default max_blob_size.")

        new_min_blob_size = input(f"Enter min blob size (default {blob_type['min_blob_size']}): ").strip()
        if new_min_blob_size:
            try:
                blob_type['min_blob_size'] = float(new_min_blob_size)
            except ValueError:
                print("Invalid input. Keeping default min_blob_size.")

        # Max and Min number of blobs
        new_max_num_blobs = input(f"Enter max number of blobs (default {blob_type['max_num_blobs']}): ").strip()
        if new_max_num_blobs:
            try:
                blob_type['max_num_blobs'] = int(new_max_num_blobs)
            except ValueError:
                print("Invalid input. Keeping default max_num_blobs.")

        new_min_num_blobs = input(f"Enter min number of blobs (default {blob_type['min_num_blobs']}): ").strip()
        if new_min_num_blobs:
            try:
                blob_type['min_num_blobs'] = int(new_min_num_blobs)
            except ValueError:
                print("Invalid input. Keeping default min_num_blobs.")

        # Randomness
        new_randomness = input(f"Enter randomness for blob shapes (default {blob_type['randomness']}): ").strip()
        if new_randomness:
            try:
                randomness_value = float(new_randomness)
                if not 0.0 <= randomness_value <= 1.0:
                    raise ValueError
                blob_type['randomness'] = randomness_value
            except:
                print("Invalid input. Keeping default randomness.")

        # Elongation
        new_min_elongation = input(f"Enter min elongation (default {blob_type['min_elongation']}): ").strip()
        if new_min_elongation:
            try:
                blob_type['min_elongation'] = float(new_min_elongation)
            except ValueError:
                print("Invalid input. Keeping default min_elongation.")

        new_max_elongation = input(f"Enter max elongation (default {blob_type['max_elongation']}): ").strip()
        if new_max_elongation:
            try:
                blob_type['max_elongation'] = float(new_max_elongation)
            except ValueError:
                print("Invalid input. Keeping default max_elongation.")

        # Number of points per blob
        new_num_points = input(
            f"Enter number of points per blob (default {blob_type.get('num_points_per_blob', config['num_points_per_blob'])}): ").strip()
        if new_num_points:
            try:
                blob_type['num_points_per_blob'] = int(new_num_points)
            except ValueError:
                print("Invalid input. Keeping default num_points_per_blob.")
        else:
            blob_type['num_points_per_blob'] = blob_type.get('num_points_per_blob', config['num_points_per_blob'])

        # Curved shapes
        new_curved_shapes = input(
            f"Use curved shapes? yes/no (default {'yes' if blob_type.get('curved_shapes', True) else 'no'}): ").strip().lower()
        if new_curved_shapes in ['yes', 'no']:
            blob_type['curved_shapes'] = True if new_curved_shapes == 'yes' else False
        elif new_curved_shapes == '':
            blob_type['curved_shapes'] = blob_type.get('curved_shapes', True)
        else:
            print("Invalid input. Keeping default curved_shapes.")

        # Avoid overlap (for blob types after the first one)
        if i > 0:  # Starting from the second blob type
            new_avoid_overlap = input(
                f"Should blobs avoid overlapping existing blobs? yes/no (default {'no' if not blob_type.get('avoid_overlap', False) else 'yes'}): ").strip().lower()
            if new_avoid_overlap in ['yes', 'no']:
                blob_type['avoid_overlap'] = True if new_avoid_overlap == 'yes' else False
            elif new_avoid_overlap == '':
                blob_type['avoid_overlap'] = blob_type.get('avoid_overlap', False)
            else:
                print("Invalid input. Keeping default avoid_overlap.")
        else:
            # For first blob type, set avoid_overlap to False
            blob_type['avoid_overlap'] = False

        # Remove 'shape_smoothness' if present
        if 'shape_smoothness' in blob_type:
            del blob_type['shape_smoothness']

        blob_types.append(blob_type)

    config['blob_types'] = blob_types

    # Display updated configuration
    print("\nUpdated Configuration:")
    display_configuration(config)

    save_config(config)
    return config

def create_buffer(mask, buffer_size):
    """
    Creates a buffer zone around the mask using OpenCV's dilation.

    Parameters:
        mask (PIL.Image): The input mask image (mode 'L').
        buffer_size (int): The number of pixels to buffer.

    Returns:
        PIL.Image: The buffered mask image.
    """
    # Convert PIL Image to NumPy array
    mask_np = np.array(mask)

    # Define the structuring element (kernel) for dilation
    kernel_size = 2 * buffer_size + 1  # Ensure the kernel size is odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform dilation
    buffered_mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    # Convert back to PIL Image
    buffered_mask = Image.fromarray(buffered_mask_np)

    return buffered_mask

def add_blobs(image, blobs_to_add, blob_color):
    draw = ImageDraw.Draw(image)
    for blob in blobs_to_add:
        draw.polygon(blob, fill=tuple(blob_color))
    del draw  # Remove the drawing context
    return image

def generate_blob(x, y, min_radius, max_radius, average_blob_size, size_variance,
                  num_points=100, randomness=0.3, min_elongation=1.0, max_elongation=1.0,
                  curved_shapes=True):
    """
    Generate a more organic blob shape with optional elongation and curvature.
    """
    # Random elongation factor
    elongation = random.uniform(min_elongation, max_elongation)

    # Generate base angles
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Generate multiple frequencies of noise
    noise = np.zeros(num_points)

    # Adjusted frequencies to reduce high-frequency spikes
    frequencies = [1, 2, 3, 4]  # Removed higher frequencies
    for freq in frequencies:
        amplitude = (1 - randomness) + randomness * (1 / freq)
        noise += np.random.normal(0, amplitude, num_points) * np.sin(angles * freq)

    # Increase the window size for smoothing
    window_size = int(num_points * 0.1)  # 10% of the number of points
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd
    kernel = np.ones(window_size) / window_size
    noise_smooth = np.convolve(noise, kernel, mode='same')

    # Normalize and scale noise
    noise_min = noise_smooth.min()
    noise_max = noise_smooth.max()
    if noise_max - noise_min != 0:
        noise_normalized = (noise_smooth - noise_min) / (noise_max - noise_min)
    else:
        noise_normalized = noise_smooth - noise_min
    noise_scaled = (noise_normalized * 2 - 1) * randomness

    # Generate base radius from normal distribution centered at average_blob_size
    base_radius = np.random.normal(average_blob_size, size_variance)
    # Ensure radius is within min and max bounds
    base_radius = max(min_radius, min(max_radius, base_radius))

    # Apply elongation
    radii = base_radius * (1 + noise_scaled)
    radii_x = radii * elongation
    radii_y = radii / elongation

    # Calculate points
    points_x = x + radii_x * np.cos(angles)
    points_y = y + radii_y * np.sin(angles)

    if curved_shapes:
        # Use spline fitting without specifying 's' to let the function choose the smoothing factor
        try:
            tck, u = splprep([points_x, points_y], per=1)  # Removed s parameter
            # Generate smoother curve with more points
            smooth_u = np.linspace(0, 1, num_points * 2)
            smooth_points = splev(smooth_u, tck)
            return list(zip(smooth_points[0], smooth_points[1]))
        except Exception as e:
            # In case spline fitting fails, fall back to original points
            print(f"Spline fitting failed: {e}. Using original points.")
            return list(zip(points_x, points_y))
    else:
        # Return points directly for polygonal shapes
        return list(zip(points_x, points_y))

def place_blobs(image, num_blobs, blob_type, color_buffer_mask, mask_image):
    width, height = image.size
    blobs = []
    attempts = 0
    max_attempts = num_blobs * 5  # Adjust as needed
    blob_rects = []  # List to store bounding boxes of placed blobs

    while len(blobs) < num_blobs and attempts < max_attempts:
        attempts += 1
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)

        if color_buffer_mask[y, x] == 255:
            blob_points = generate_blob(
                x, y,
                min_radius=blob_type['min_blob_size'],
                max_radius=blob_type['max_blob_size'],
                average_blob_size=blob_type['average_blob_size'],
                size_variance=blob_type['size_variance'],
                num_points=blob_type['num_points_per_blob'],
                randomness=blob_type['randomness'],
                min_elongation=blob_type['min_elongation'],
                max_elongation=blob_type['max_elongation'],
                curved_shapes=blob_type.get('curved_shapes', True)
            )

            # Get bounding box of the blob
            xs, ys = zip(*blob_points)
            min_x, max_x = max(0, int(min(xs))), min(width, int(max(xs)))
            min_y, max_y = max(0, int(min(ys))), min(height, int(max(ys)))
            blob_rect = (min_x, min_y, max_x, max_y)

            # Check for overlap using bounding boxes
            overlap = False
            if blob_type.get('avoid_overlap', False):
                for rect in blob_rects:
                    if rects_intersect(blob_rect, rect):
                        overlap = True
                        break

            if not overlap:
                blob_rects.append(blob_rect)
                blobs.append(blob_points)

    if len(blobs) < num_blobs:
        print(f"Only placed {len(blobs)} blobs out of requested {num_blobs}.")

    return blobs

def rects_intersect(r1, r2):
    # Check if two rectangles intersect
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

# Main function
def main():
    # Load configuration
    config = load_config()

    # If not skipping prompts, get user input and update configuration
    if not args.no_prompt:
        config = get_user_input(config)
    else:
        print("Skipping configuration prompts. Using existing configuration.")

    # Load the image
    try:
        image = Image.open(config['input_path']).convert("RGB")
        print(f"\nLoaded image: {config['input_path']} (Size: {image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    width, height = image.size

    # Convert image to NumPy array for vectorized processing
    image_array = np.array(image)

    print("\nCreating color and black masks using vectorized operations...")

    # Vectorized color mask creation
    target_color = np.array(config['target_color'])
    color_tolerance = config['color_tolerance']

    color_diff = np.abs(image_array - target_color)
    color_mask = np.all(color_diff <= color_tolerance, axis=2).astype(np.uint8) * 255

    # Vectorized black mask creation
    black_mask = ((image_array[:, :, 0] < 10) &
                  (image_array[:, :, 1] < 10) &
                  (image_array[:, :, 2] < 10)).astype(np.uint8) * 255

    # Convert masks back to PIL Image objects
    mask_image_pil = Image.fromarray(color_mask, mode='L')
    black_mask_pil = Image.fromarray(black_mask, mode='L')

    # Create buffer zone around target color areas using OpenCV
    buffer_size = config['buffer_size']
    print(f"\nCreating a buffer zone of {buffer_size} pixels around target color areas...")
    color_buffer_mask_pil = create_buffer(mask_image_pil, buffer_size)
    color_buffer_mask = np.array(color_buffer_mask_pil)

    # Convert color mask to NumPy array for blob placement
    mask_image = color_mask  # Original color mask without buffer

    # Initialize existing blobs mask
    #existing_blobs_mask = np.zeros((height, width), dtype=bool)

    # Place blobs for each blob type
    print("\nStarting blob placement...")
    all_blobs = []
    for idx, blob_type in enumerate(config['blob_types']):
        print(f"\nPlacing blobs for {blob_type['name']}...")
        num_blobs = random.randint(blob_type['min_num_blobs'], blob_type['max_num_blobs'])
        blobs_to_add = place_blobs(
            image=image,
            num_blobs=num_blobs,
            blob_type=blob_type,
            color_buffer_mask=color_buffer_mask,
            mask_image=mask_image
        )
        # Add blobs to the image
        print(f"Adding {len(blobs_to_add)} blobs to the image...")
        image = add_blobs(image, blobs_to_add, blob_type['blob_color'])

    # Reapply black areas to cover any blobs that may have extended into them
    print("\nReapplying black areas to ensure they overlay any overlapping blobs...")
    # Reconstruct image array
    final_image_array = np.array(image)
    final_image_array[black_mask == 255] = [0, 0, 0]  # Ensure black remains black
    final_image = Image.fromarray(final_image_array)

    # Save the output image
    try:
        final_image.save(config['output_path'])
        print(f"\nImage saved to '{config['output_path']}'")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    main()
