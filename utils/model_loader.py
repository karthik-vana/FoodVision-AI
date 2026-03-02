"""
Model Loader Module
Handles dynamic loading of TensorFlow/Keras deep learning models.
"""

import os
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and caches deep learning models dynamically.
    
    Attributes:
        model_paths (dict): Mapping of model names to file paths.
        _loaded_models (dict): Cache of already-loaded model objects.
    """

    def __init__(self, model_paths: dict):
        """
        Constructor — initializes model paths and internal cache.
        
        Args:
            model_paths (dict): Dictionary mapping model keys to .h5 file paths.
        
        Raises:
            ValueError: If model_paths is empty or not a dictionary.
        """
        if not isinstance(model_paths, dict) or not model_paths:
            raise ValueError("model_paths must be a non-empty dictionary.")
        self.model_paths = model_paths
        self._loaded_models = {}
        logger.info("ModelLoader initialized with %d model(s).", len(model_paths))

    def load_model(self, model_key: str):
        """
        Load a model by its key. Uses cache if already loaded.
        
        Args:
            model_key (str): Key identifying the model (e.g., 'custom_cnn').
        
        Returns:
            tensorflow.keras.Model: The loaded Keras model.
        
        Raises:
            ValueError: If the model_key is not recognized.
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If model loading fails for any reason.
        """
        try:
            if model_key not in self.model_paths:
                raise ValueError(
                    f"Invalid model selection: '{model_key}'. "
                    f"Available models: {list(self.model_paths.keys())}"
                )

            # Return cached model if already loaded
            if model_key in self._loaded_models:
                logger.info("Returning cached model: '%s'.", model_key)
                return self._loaded_models[model_key]

            model_path = self.model_paths[model_key]

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at: {model_path}"
                )

            # Lazy import to avoid startup overhead
            from tensorflow.keras.models import load_model as keras_load_model

            logger.info("Loading model '%s' from %s...", model_key, model_path)
            model = keras_load_model(model_path, compile=False)
            self._loaded_models[model_key] = model
            logger.info("Model '%s' loaded successfully.", model_key)
            return model

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_key, str(e))
            raise RuntimeError(f"Error loading model '{model_key}': {str(e)}") from e

    def get_available_models(self) -> list:
        """
        Returns a list of available model keys.
        
        Returns:
            list: List of model key strings.
        """
        return list(self.model_paths.keys())

    def is_model_loaded(self, model_key: str) -> bool:
        """
        Check if a model is already loaded in cache.
        
        Args:
            model_key (str): The model key to check.
        
        Returns:
            bool: True if model is cached, False otherwise.
        """
        return model_key in self._loaded_models

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._loaded_models.clear()
        logger.info("Model cache cleared.")
