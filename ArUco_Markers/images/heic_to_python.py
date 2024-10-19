from PIL import Image
import pillow_heif
import os
import glob
import pathlib

path = os.path.dirname(__file__)
print(path)
base_path = os.path.join(path, "calib_images", "calib_iphone")
print(base_path)
dir = base_path

def heic_jpg(image_path, save_path):
    heif_file = pillow_heif.read_heif(image_path)
    for img in heif_file: 
        image = Image.frombytes(
            img.mode,
            img.size,
            img.data,
            'raw',
            img.mode,
            img.stride,
        )
    image.save(save_path, "JPEG")

image_dir = pathlib.Path(dir)
heic_path = list(image_dir.glob('**/*.HEIC'))

for i in heic_path:
    image_path = str(i)
    save_path =  str(image_dir / i.stem) + '.jpg'
    print(save_path)
    heic_jpg(image_path, save_path)