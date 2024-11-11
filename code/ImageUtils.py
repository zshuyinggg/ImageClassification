import numpy as np
from matplotlib import pyplot as plt
import albumentations as A


"""This script implements the functions for data augmentation
and preprocessing.
"""
transform = A.Compose([
    A.RandomCrop(width=32, height=32),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomRotate90(p=0.5)
])
def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """

    if training:
        if np.random.rand()>0.5:
            image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='constant',constant_values=0)
        image = transform(image=image)['image']
        #  # Resize the image to add four extra pixels on each side.
        # image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        
        # # Randomly crop a [32, 32] section of the image.
        # # randomly generate the upper left point of the crop
        # start_x = np.random.randint(0, image.shape[0] - 32)
        # start_y = np.random.randint(0, image.shape[1] - 32)
        # image = image[start_x:start_x+32, start_y:start_y+32, :]

        # # Randomly flip the image horizontally. #question
        # if np.random.rand() > 0.5:
        #     image = np.fliplr(image)

    # Subtract off the mean and divide by the standard deviation of the pixels. # in my experiments, this makes an image unrecognization.  also in slides, it says do not commonly do normaliazion
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std

    return image

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image=np.transpose(image.reshape((3, 32, 32)), [1, 2, 0])
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE