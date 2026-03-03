"""
Image Preprocessor Module
Handles image loading, resizing, normalization, and tensor conversion.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses uploaded images for model inference.
    
    Attributes:
        target_size (tuple): Target image dimensions (width, height).
    """

    def __init__(self, target_size: tuple = (224, 224)):
        """
        Constructor — initializes the target image size.
        
        Args:
            target_size (tuple): Desired output dimensions, default (224, 224).
        
        Raises:
            ValueError: If target_size is not a tuple of two positive integers.
        """
        if (not isinstance(target_size, tuple) or len(target_size) != 2
                or not all(isinstance(d, int) and d > 0 for d in target_size)):
            raise ValueError("target_size must be a tuple of two positive integers.")
        self.target_size = target_size
        logger.info("ImagePreprocessor initialized with target_size=%s.", target_size)

    def preprocess(self, image_path: str, target_size: tuple = None) -> np.ndarray:
        """
        Load and preprocess an image from file path.
        
        Args:
            image_path (str): Absolute path to the uploaded image file.
            target_size (tuple, optional): Target size override. Defaults to self.target_size.
        
        Returns:
            np.ndarray: Preprocessed image tensor of shape (1, H, W, 3).
        
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the file is not a valid image.
            RuntimeError: If preprocessing fails.
        """
        try:
            import os
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            from tensorflow.keras.preprocessing.image import load_img, img_to_array

            # Use override if provided
            size = target_size if target_size else self.target_size

            # Load and resize image
            img = load_img(image_path, target_size=size)
            logger.info("Image loaded and resized to %s.", size)

            # Convert to numpy array
            img_array = img_to_array(img)

            # Normalize pixel values to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension → (1, H, W, 3)
            img_tensor = np.expand_dims(img_array, axis=0)

            logger.info("Image preprocessed successfully. Shape: %s", img_tensor.shape)
            return img_tensor

        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logger.error("Image preprocessing failed: %s", str(e))
            raise RuntimeError(f"Failed to preprocess image: {str(e)}") from e

    def preprocess_from_bytes(self, image_bytes: bytes, target_size: tuple = None) -> np.ndarray:
        """
        Preprocess an image directly from bytes (for in-memory processing).
        
        Args:
            image_bytes (bytes): Raw image bytes.
            target_size (tuple, optional): Target size override (H, W). Defaults to self.target_size.
        
        Returns:
            np.ndarray: Preprocessed image tensor of shape (1, H, W, 3).
        
        Raises:
            ValueError: If image bytes are invalid.
            RuntimeError: If preprocessing fails.
        """
        try:
            from io import BytesIO
            from PIL import Image

            if not image_bytes:
                raise ValueError("Empty image bytes received.")

            size = target_size if target_size else self.target_size

            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img = img.resize(size)

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            logger.info("Image from bytes preprocessed. Shape: %s", img_tensor.shape)
            return img_tensor

        except ValueError:
            raise
        except Exception as e:
            logger.error("Bytes preprocessing failed: %s", str(e))
            raise RuntimeError(f"Failed to preprocess image bytes: {str(e)}") from e
