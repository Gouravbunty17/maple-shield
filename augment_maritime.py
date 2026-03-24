from pathlib import Path
import sys
sys.path.append(".")
from arctic_augment import batch_augment

# Augment maritime dataset
batch_augment(
    input_dir="datasets/maritime_sample",
    output_dir="datasets/arctic_maritime",
    versions_per_image=4
)

print("\n" + "="*50)
print("DATASET STATS:")
input_count = len(list(Path("datasets/maritime_sample").glob("*.jpg")))
output_count = len(list(Path("datasets/arctic_maritime").glob("*.jpg")))
print(f"Input images: {input_count}")
print(f"Augmented images: {output_count}")
print(f"Augmentation factor: {output_count / max(input_count, 1):.1f}x")

