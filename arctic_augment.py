import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise

def add_fog(img, density=0.5):
    """Add fog overlay using alpha blending"""
    arr = np.array(img).astype(float)
    
    # Create fog layer (white/gray noise)
    fog = np.ones_like(arr) * 255
    fog_noise = random_noise(fog / 255.0, mode='gaussian', var=0.01) * 255
    
    # Blend with original image
    fogged = arr * (1 - density) + fog_noise * density
    return Image.fromarray(fogged.astype(np.uint8))

def add_snow(img, intensity=0.3):
    """Add snow particles (white specks)"""
    arr = np.array(img)
    
    # Random snow particle mask
    snow_mask = np.random.random(arr.shape[:2]) < (intensity * 0.02)
    
    # Add white pixels where snow mask is True
    for i in range(3):  # RGB channels
        arr[:, :, i] = np.where(snow_mask, 255, arr[:, :, i])
    
    return Image.fromarray(arr)

def adjust_temperature(img, kelvin=6500):
    """Apply cold blue color temperature"""
    # Kelvin 5500-7500 = cold/blue tint
    factor = (kelvin - 5500) / 2000.0  # normalize to [0, 1]
    
    arr = np.array(img).astype(float)
    
    # Increase blue, decrease red for cold look
    arr[:, :, 0] *= (1 - factor * 0.2)  # R
    arr[:, :, 2] *= (1 + factor * 0.3)  # B
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def reduce_contrast(img, factor=0.6):
    """Reduce contrast (whiteout effect)"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_low_light(img, gain=0.3):
    """Simulate polar night (low light)"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(gain)

def arctic_augment(img_path, output_path, preset='moderate'):
    """
    Apply Arctic augmentation to single image
    
    Presets:
    - light: Mild fog, minimal snow
    - moderate: Medium fog + snow
    - heavy: Dense fog, heavy snow, whiteout
    - polar_night: Low light + fog
    """
    img = Image.open(img_path).convert('RGB')
    
    if preset == 'light':
        img = add_fog(img, density=random.uniform(0.2, 0.4))
        img = add_snow(img, intensity=random.uniform(0.1, 0.2))
        img = adjust_temperature(img, kelvin=random.uniform(6000, 6500))
        
    elif preset == 'moderate':
        img = add_fog(img, density=random.uniform(0.4, 0.6))
        img = add_snow(img, intensity=random.uniform(0.3, 0.5))
        img = adjust_temperature(img, kelvin=random.uniform(6200, 7000))
        img = reduce_contrast(img, factor=random.uniform(0.6, 0.8))
        
    elif preset == 'heavy':
        img = add_fog(img, density=random.uniform(0.6, 0.8))
        img = add_snow(img, intensity=random.uniform(0.5, 0.7))
        img = adjust_temperature(img, kelvin=random.uniform(6500, 7500))
        img = reduce_contrast(img, factor=random.uniform(0.4, 0.6))
        
    elif preset == 'polar_night':
        img = apply_low_light(img, gain=random.uniform(0.2, 0.4))
        img = add_fog(img, density=random.uniform(0.3, 0.5))
        img = adjust_temperature(img, kelvin=random.uniform(7000, 8000))
    
    img.save(output_path)
    print(f"Saved: {output_path}")

def batch_augment(input_dir, output_dir, versions_per_image=3):
    """
    Augment all images in directory
    
    Args:
        input_dir: Source images folder
        output_dir: Where to save augmented images
        versions_per_image: How many augmented versions per source image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    presets = ['light', 'moderate', 'heavy', 'polar_night']
    
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Generating {versions_per_image} augmented versions each...")
    
    for img_file in image_files:
        for i in range(versions_per_image):
            preset = random.choice(presets)
            out_name = f"{img_file.stem}_arctic_{preset}_{i}{img_file.suffix}"
            out_path = output_path / out_name
            
            try:
                arctic_augment(str(img_file), str(out_path), preset=preset)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    print(f"\nDone! Augmented images saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    batch_augment(
        input_dir="datasets/test_images",
        output_dir="datasets/arctic_augmented",
        versions_per_image=4  # 4 versions per image (light, moderate, heavy, polar_night)
    )

