"""
Nutrition Loader Module
Loads and retrieves nutritional data from the JSON file.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


class NutritionLoader:
    """
    Loads nutrition data from a JSON file and provides lookup by food class.
    
    Attributes:
        nutrition_path (str): Path to the nutrition JSON file.
        _data (dict): Parsed nutrition data.
    """

    def __init__(self, nutrition_path: str):
        """
        Constructor — initializes and loads the nutrition JSON.
        
        Args:
            nutrition_path (str): Absolute path to nutrition.json.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If JSON parsing fails.
        """
        if not os.path.exists(nutrition_path):
            raise FileNotFoundError(f"Nutrition file not found: {nutrition_path}")

        self.nutrition_path = nutrition_path
        self._data = {}
        self._load_data()

    def _load_data(self):
        """Load and parse the nutrition JSON file."""
        try:
            with open(self.nutrition_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            logger.info(
                "Nutrition data loaded: %d food items.", len(self._data)
            )
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error: %s", str(e))
            raise ValueError(f"Invalid JSON in nutrition file: {str(e)}") from e
        except Exception as e:
            logger.error("Failed to load nutrition data: %s", str(e))
            raise RuntimeError(f"Error loading nutrition data: {str(e)}") from e

    def get_nutrition(self, food_class: str) -> dict:
        """
        Get nutrition data for a predicted food class.
        Performs case-insensitive lookup with fallback strategies.
        
        Args:
            food_class (str): The predicted food class name.
        
        Returns:
            dict: Nutrition information for the food item, or default if not found.
        """
        try:
            # Direct lookup
            if food_class in self._data:
                return self._data[food_class]

            # Case-insensitive lookup
            lower_map = {k.lower(): v for k, v in self._data.items()}
            if food_class.lower() in lower_map:
                return lower_map[food_class.lower()]

            # Partial match (underscore/space variants)
            normalized = food_class.replace('_', ' ').lower()
            for key, value in self._data.items():
                if key.replace('_', ' ').lower() == normalized:
                    return value

            # Special mapping for known mismatches
            special_map = {
                'chiken_curry': 'chicken_curry',
                'kaathio_rolls': 'kaathi_rolls',
            }
            mapped = special_map.get(food_class, food_class)
            if mapped in self._data:
                return self._data[mapped]

            logger.warning("Nutrition data not found for: '%s'.", food_class)
            return self._get_default_nutrition(food_class)

        except Exception as e:
            logger.error("Error fetching nutrition for '%s': %s", food_class, str(e))
            return self._get_default_nutrition(food_class)

    def _get_default_nutrition(self, food_class: str) -> dict:
        """Return default nutrition data when lookup fails."""
        return {
            'calories_kcl': 'N/A',
            'carbs_g': 'N/A',
            'protein_g': 'N/A',
            'fat_g': 'N/A',
            'fiber_g': 'N/A',
            'sugar_g': 'N/A',
            'vitamins': {'vitamin_a_mg': 'N/A', 'vitamin_c_mg': 'N/A'},
            'minerals': {'calcium_mg': 'N/A', 'iron_mg': 'N/A', 'potassium_mg': 'N/A'},
            '_note': f'No data available for {food_class}'
        }

    def get_all_classes(self) -> list:
        """Return all food class names in the nutrition data."""
        return list(self._data.keys())

    def get_class_count(self) -> int:
        """Return total number of food classes."""
        return len(self._data)
