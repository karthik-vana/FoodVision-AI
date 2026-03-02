"""
Metrics Reader Module
Parses model validation report text files to extract performance metrics.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


class MetricsReader:
    """
    Reads and parses validation report .txt files to extract model metrics.
    
    Attributes:
        metrics_paths (dict): Mapping of model keys to validation report file paths.
    """

    def __init__(self, metrics_paths: dict):
        """
        Constructor — initializes with metrics file paths.
        
        Args:
            metrics_paths (dict): Mapping of model keys to .txt file paths.
        
        Raises:
            ValueError: If metrics_paths is empty or not a dictionary.
        """
        if not isinstance(metrics_paths, dict) or not metrics_paths:
            raise ValueError("metrics_paths must be a non-empty dictionary.")
        self.metrics_paths = metrics_paths
        logger.info("MetricsReader initialized with %d report(s).", len(metrics_paths))

    def read_metrics(self, model_key: str) -> dict:
        """
        Read and parse the validation metrics for the given model.
        
        Args:
            model_key (str): Key identifying the model (e.g., 'custom_cnn').
        
        Returns:
            dict: {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'total_tp': int,
                'total_tn': int,
                'total_fp': int,
                'total_fn': int,
                'model_name': str,
            }
        
        Raises:
            ValueError: If the model_key is not recognized.
            FileNotFoundError: If the metrics file does not exist.
            RuntimeError: If parsing fails.
        """
        try:
            if model_key not in self.metrics_paths:
                raise ValueError(
                    f"Invalid model key: '{model_key}'. "
                    f"Available: {list(self.metrics_paths.keys())}"
                )

            file_path = self.metrics_paths[model_key]
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Metrics file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            metrics = self._parse_report(content, model_key)
            logger.info("Metrics successfully parsed for model '%s'.", model_key)
            return metrics

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error("Failed to read metrics for '%s': %s", model_key, str(e))
            raise RuntimeError(f"Error reading metrics: {str(e)}") from e

    def _parse_report(self, content: str, model_key: str) -> dict:
        """
        Internal method to parse the validation report content.
        
        Args:
            content (str): Raw text content of the validation report.
            model_key (str): The model key for labeling.
        
        Returns:
            dict: Extracted metrics dictionary.
        """
        metrics = {
            'model_name': model_key,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_tp': 0,
            'total_tn': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

        try:
            # Extract Overall Test Accuracy
            acc_match = re.search(
                r'Overall Test Accuracy\s*:\s*([\d.]+)\s*\(([\d.]+)%\)', content
            )
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(2))

            # Extract weighted precision
            prec_match = re.search(
                r'Precision\s*\(weighted\)\s*:\s*([\d.]+)', content
            )
            if prec_match:
                metrics['precision'] = round(float(prec_match.group(1)) * 100, 2)

            # Extract weighted recall
            rec_match = re.search(
                r'Recall\s*\(weighted\)\s*:\s*([\d.]+)', content
            )
            if rec_match:
                metrics['recall'] = round(float(rec_match.group(1)) * 100, 2)

            # Extract weighted F1-Score
            f1_match = re.search(
                r'F1-Score\s*\(weighted\)\s*:\s*([\d.]+)', content
            )
            if f1_match:
                metrics['f1_score'] = round(float(f1_match.group(1)) * 100, 2)

            # Extract TOTAL TP, TN, FP, FN
            total_match = re.search(
                r'TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', content
            )
            if total_match:
                metrics['total_tp'] = int(total_match.group(1))
                metrics['total_tn'] = int(total_match.group(2))
                metrics['total_fp'] = int(total_match.group(3))
                metrics['total_fn'] = int(total_match.group(4))

        except Exception as e:
            logger.warning("Partial parse failure for '%s': %s", model_key, str(e))

        return metrics
