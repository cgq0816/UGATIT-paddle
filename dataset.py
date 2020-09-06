from PIL import Image, ImageOps
import os
import numpy as np



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    return images


def reader_creater(
    root,  
    cycle=True, 
    shuffle=False,
    return_name=False):

    images = make_dataset(root)
    
    def reader():
        while True:
            if shuffle:
                np.random.shuffle(images)
            for filepath in images:
                image = Image.open(filepath)
                ## Resize
                image = image.resize((286, 286), Image.BICUBIC)
                ## RandomCrop
                i = np.random.randint(0, 30)
                j = np.random.randint(0, 30)
                image = image.crop((i, j , i+256, j+256))
                # RandomHorizontalFlip
                sed = np.random.rand()
                if sed > 0.5:
                    image = ImageOps.mirror(image)
                # ToTensor
                image = np.array(image).transpose([2, 0, 1]).astype('float32')
                image = image / 255.0
                # Normalize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
                image = (image - 0.5) / 0.5
                
                if return_name:
                    yield image[np.newaxis, :], os.path.basename(filepath)
                else:
                    yield image
                
            if not cycle:
                break

    return reader