from PIL import Image


class ImageLoader:
    """
    This class provides methods for loading images
    """
    @staticmethod
    def load_from_disk(image_path: str, image_shape=(512,512)) -> Image:
        """
        Load image from disk as PIL Image

        Parameters:
            image_path: path where image is stored
            image_shape: tuple containing desired width and height

        Return:
            PIL image object
        """
        # Load the image corresponding to the selected filename
        image = Image.open(image_path).resize(image_shape)
        image = image.convert(mode='RGB')
        return image

    """
    This class provides methods for loading images
    """
    @staticmethod
    def load_from_uploaded_file(image, image_shape=(512,512)) -> Image:
        """
        Load image from disk as PIL Image

        Parameters:
            image: file object
            image_shape: tuple containing desired width and height

        Return:
            PIL image object
        """
        # Load the image corresponding to the selected filename
        image = Image.open(image).resize(image_shape)
        image = image.convert(mode='RGB')
        return image