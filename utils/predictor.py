"""
Predictor Module
Runs inference using a loaded Keras model on preprocessed images.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """
    Performs prediction using a Keras model and returns class label + confidence.
    
    Attributes:
        class_labels (list): Ordered list of class names matching model output indices.
    """

    def __init__(self, class_labels: list):
        """
        Constructor — initializes with the ordered class labels.
        
        Args:
            class_labels (list): List of class name strings.
        
        Raises:
            ValueError: If class_labels is empty or not a list.
        """
        if not isinstance(class_labels, list) or not class_labels:
            raise ValueError("class_labels must be a non-empty list.")
        self.class_labels = class_labels
        logger.info("Predictor initialized with %d classes.", len(class_labels))

    def predict(self, model, image_tensor: np.ndarray) -> dict:
        """
        Run prediction on the given image tensor using the provided model.
        
        Args:
            model: A compiled/loaded Keras model.
            image_tensor (np.ndarray): Preprocessed image tensor of shape (1, H, W, 3).
        
        Returns:
            dict: {
                'predicted_class': str,
                'confidence': float,
                'class_index': int,
                'all_probabilities': dict  # top 5 predictions
            }
        
        Raises:
            ValueError: If model or image_tensor is None/invalid.
            RuntimeError: If prediction fails.
        """
        try:
            if model is None:
                raise ValueError("Model cannot be None.")
            if image_tensor is None or not isinstance(image_tensor, np.ndarray):
                raise ValueError("image_tensor must be a valid numpy array.")

            # Run inference
            predictions = model.predict(image_tensor, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            predicted_class = self.class_labels[predicted_index]

            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            top_predictions = {
                self.class_labels[i]: round(float(predictions[0][i]) * 100, 2)
                for i in top_indices
            }

            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'class_index': predicted_index,
                'all_probabilities': top_predictions,
            }

            logger.info(
                "Prediction: %s (%.2f%% confidence).",
                predicted_class, confidence * 100
            )
            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            raise RuntimeError(f"Prediction error: {str(e)}") from e
