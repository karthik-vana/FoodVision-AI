"""
resnet_model.py
---------------
Trains a ResNet50 Transfer Learning model for food image classification (34 classes).
Uses OOP (Classes, Objects, Constructors) and Exception Handling.
Generates a validation report and saves it to ResNet_Model.txt.

Development Environment: PyCharm
"""

import os
import sys
import numpy as np
import warnings
import matplotlib.pyplot as plt

# ── TensorFlow / Keras Imports ──
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

# ── Scikit-learn Imports (for report) ──
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score,
                             recall_score, f1_score)

warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════
#  CLASS : FoodDataLoader
#  Purpose : Load training, validation, and testing data
# ════════════════════════════════════════════════════════════════
class FoodDataLoader:
    """Loads and prepares image data for training, validation, and testing."""

    def __init__(self, training_path, validation_path, testing_path,
                 image_size=(256, 256), batch_size=34):
        """
        Constructor — sets up paths, image size, batch size, and label list.

        Parameters
        ----------
        training_path   : str  — path to training images folder
        validation_path : str  — path to validation images folder
        testing_path    : str  — path to testing images folder
        image_size      : tuple — target size for all images (height, width)
        batch_size      : int   — number of images per batch
        """
        self.training_path = training_path
        self.validation_path = validation_path
        self.testing_path = testing_path
        self.image_size = image_size
        self.batch_size = batch_size

        # 34 food class labels (must match folder names)
        self.labels = [
            'pakode', 'kulfi', 'Hot Dog', 'dhokla', 'masala_dosa',
            'chole_bhature', 'apple_pie', 'Crispy Chicken', 'burger',
            'chapati', 'paani_puri', 'sushi', 'dal_makhani', 'Donut',
            'Fries', 'cheesecake', 'omelette', 'Baked Potato', 'ice_cream',
            'pizza', 'Taquito', 'jalebi', 'chai', 'kaathi_rolls', 'Taco',
            'chicken_curry', 'pav_bhaji', 'butter_naan', 'momos', 'samosa',
            'fried_rice', 'Sandwich', 'idli', 'kadai_paneer'
        ]

        # Data generators (will be created in load_data)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        """
        Creates ImageDataGenerators and loads data from directories.
        Training data gets augmentation; validation and test data only get rescaled.
        """
        try:
            print("=" * 60)
            print("  LOADING DATA FOR RESNET MODEL")
            print("=" * 60)

            # ── Verify all data directories exist before loading ──
            print(f"\n  Training path   : {self.training_path}")
            print(f"  Validation path : {self.validation_path}")
            print(f"  Testing path    : {self.testing_path}")

            for path_name, path_value in [('Training', self.training_path),
                                           ('Validation', self.validation_path),
                                           ('Testing', self.testing_path)]:
                if not os.path.isdir(path_value):
                    print(f"\n  ERROR: {path_name} directory does NOT exist: {path_value}")
                    print("  Please check your folder structure.")
                    return

            # ── Data augmentation rules for training ──
            train_data_rules = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                shear_range=0.2,
                zoom_range=0.2
            )

            # ── Only preprocessing for validation and testing ──
            val_data_rules = ImageDataGenerator(preprocessing_function=preprocess_input)
            test_data_rules = ImageDataGenerator(preprocessing_function=preprocess_input)

            # ── Load training data ──
            self.train_data = train_data_rules.flow_from_directory(
                self.training_path,
                classes=self.labels,
                color_mode='rgb',
                target_size=self.image_size,
                class_mode='categorical',
                batch_size=self.batch_size,
                shuffle=True
            )

            # ── Load validation data ──
            self.val_data = val_data_rules.flow_from_directory(
                self.validation_path,
                classes=self.labels,
                color_mode='rgb',
                target_size=self.image_size,
                class_mode='categorical',
                batch_size=self.batch_size,
                shuffle=False
            )

            # ── Load testing data ──
            self.test_data = test_data_rules.flow_from_directory(
                self.testing_path,
                classes=self.labels,
                color_mode='rgb',
                target_size=self.image_size,
                class_mode='categorical',
                batch_size=self.batch_size,
                shuffle=False
            )

            print(f"\n  Training samples   : {self.train_data.samples}")
            print(f"  Validation samples : {self.val_data.samples}")
            print(f"  Testing samples    : {self.test_data.samples}")
            print(f"  Number of classes  : {len(self.labels)}")

            # ── Verify images were actually found ──
            if self.train_data.samples == 0:
                print("\n  ERROR: 0 training images found! Check folder names match labels.")
                self.train_data = None
            if self.val_data.samples == 0:
                print("\n  ERROR: 0 validation images found! Check folder names match labels.")
                self.val_data = None
            if self.test_data.samples == 0:
                print("\n  ERROR: 0 testing images found! Check folder names match labels.")
                self.test_data = None

            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  CLASS : ResNetModel
#  Purpose : Build, train, evaluate, and save a ResNet50 model
# ════════════════════════════════════════════════════════════════
class ResNetModel:
    """Builds and trains a ResNet50 Transfer Learning model for food classification."""

    def __init__(self, num_classes=34, image_size=(256, 256, 3), epochs=30):
        """
        Constructor — sets up model parameters.

        Parameters
        ----------
        num_classes : int   — number of food categories
        image_size  : tuple — input image shape (height, width, channels)
        epochs      : int   — number of training epochs
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.epochs = epochs
        self.model = None       # will hold the Keras model
        self.history = None     # will hold training history

    def build_model(self):
        """
        Builds the ResNet50 Transfer Learning model:
         - Loads pre-trained ResNet50 (ImageNet weights) without top layers
         - Freezes all ResNet50 layers (no retraining)
         - Adds custom Dense layers on top for 34-class classification
        """
        try:
            print("\n" + "=" * 60)
            print("  BUILDING RESNET50 TRANSFER LEARNING MODEL")
            print("=" * 60)

            # ── Step 1: Load pre-trained ResNet50 base model ──
            resnet_base = ResNet50(
                input_shape=self.image_size,
                weights='imagenet',
                include_top=False     # remove the original top (classification) layers
            )

            # ── Step 2: Freeze most layers, fine-tune the last 10 layers ──
            for layer in resnet_base.layers[:-10]:
                layer.trainable = False

            print(f"  ResNet50 base loaded with {len(resnet_base.layers)} layers (all frozen)")

            # ── Step 3: Add custom classification layers on top ──
            one_d_values = Flatten()(resnet_base.output)

            h1_out = Dense(units=128, kernel_initializer='he_uniform',
                           activation='relu')(one_d_values)
            h2_out = Dense(units=64, kernel_initializer='he_uniform',
                           activation='relu')(h1_out)
            h3_out = Dense(units=32, kernel_initializer='he_uniform',
                           activation='relu')(h2_out)

            # ── Output Layer: 34 classes with softmax ──
            output = Dense(units=self.num_classes,
                           kernel_initializer='glorot_uniform',
                           activation='softmax')(h3_out)

            # ── Step 4: Create final model ──
            self.model = Model(inputs=resnet_base.input, outputs=output)

            # ── Step 5: Compile the model ──
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # ── Print model summary ──
            self.model.summary()
            print("  ResNet50 model built successfully!")
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')

    def train_model(self, train_data, val_data):
        """
        Trains the ResNet50 model.

        Parameters
        ----------
        train_data : DirectoryIterator — training data generator
        val_data   : DirectoryIterator — validation data generator
        """
        try:
            print("\n" + "=" * 60)
            print(f"  TRAINING RESNET50 MODEL ({self.epochs} epochs)")
            print("=" * 60)

            self.history = self.model.fit(
                train_data,
                epochs=self.epochs,
                validation_data=val_data
            )

            print("\n  Training completed successfully!")
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')

    def save_model(self, save_path="resnet_model.h5"):
        """Saves the trained ResNet50 model to disk."""
        try:
            if self.model is None:
                print("  ERROR: Model is not built yet. Cannot save.")
                return
            self.model.save(save_path)
            print(f"\n  ResNet50 Model saved to: {save_path}")

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')

    def plot_training_history(self, save_dir=None):
        """Plots training and validation accuracy/loss graphs."""
        try:
            if self.history is None:
                print("  ERROR: No training history found. Train the model first.")
                return

            epochs_ran = len(self.history.history['accuracy'])
            epoch_range = np.arange(1, epochs_ran + 1)

            plt.figure(figsize=(12, 4))

            # ── Plot 1: Training Performance ──
            plt.subplot(1, 2, 1)
            plt.title('ResNet50 — Training Performance')
            plt.plot(epoch_range, self.history.history['accuracy'],
                     color='g', label='Train Accuracy')
            plt.plot(epoch_range, self.history.history['loss'],
                     color='r', label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            # ── Plot 2: Validation Performance ──
            plt.subplot(1, 2, 2)
            plt.title('ResNet50 — Validation Performance')
            plt.plot(epoch_range, self.history.history['val_accuracy'],
                     color='g', label='Val Accuracy')
            plt.plot(epoch_range, self.history.history['val_loss'],
                     color='r', label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()

            # ── Save PNG to the script's directory (absolute path) ──
            if save_dir is None:
                save_dir = os.path.dirname(os.path.abspath(__file__))
            plot_path = os.path.join(save_dir, 'resnet_training_plot.png')
            plt.savefig(plot_path, dpi=150)
            plt.show()
            print(f"  Training plot saved to: {plot_path}")

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  CLASS : ModelEvaluator
#  Purpose : Evaluate model and generate validation report
# ════════════════════════════════════════════════════════════════
class ModelEvaluator:
    """Evaluates a trained model and generates a detailed validation report."""

    def __init__(self, model, test_data, labels, report_file="ResNet_Model.txt"):
        """
        Constructor — stores the model, test data, labels and report path.

        Parameters
        ----------
        model       : Keras model — the trained model object
        test_data   : DirectoryIterator — testing data generator
        labels      : list — list of class label names
        report_file : str — filename to save the report
        """
        self.model = model
        self.test_data = test_data
        self.labels = labels
        self.report_file = report_file

    def evaluate_and_generate_report(self):
        """
        Runs prediction on test data and computes all metrics:
         - Accuracy, TP, TN, FP, FN, Precision, Recall, F1-Score
        Saves the report to a text file.
        """
        try:
            print("\n" + "=" * 60)
            print("  EVALUATING RESNET50 MODEL ON TEST DATA")
            print("=" * 60)

            # ── Step 1: Reset test data generator and get predictions ──
            self.test_data.reset()
            predictions = self.model.predict(self.test_data)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = self.test_data.classes

            # ── Truncate predictions to match true labels length ──
            # (the last batch may pad extra samples beyond the dataset size)
            predicted_classes = predicted_classes[:len(true_classes)]

            # ── Step 2: Compute overall accuracy ──
            overall_accuracy = accuracy_score(true_classes, predicted_classes)

            # ── Step 3: Compute confusion matrix ──
            # Pass labels explicitly so the matrix is always num_classes × num_classes
            num_classes = len(self.labels)
            cm = confusion_matrix(true_classes, predicted_classes,
                                  labels=list(range(num_classes)))

            # ── Step 4: Compute TP, TN, FP, FN for each class ──
            tp_list = []
            tn_list = []
            fp_list = []
            fn_list = []

            for i in range(num_classes):
                tp = cm[i, i]
                fn = sum(cm[i, :]) - tp
                fp = sum(cm[:, i]) - tp
                tn = cm.sum() - tp - fn - fp

                tp_list.append(tp)
                tn_list.append(tn)
                fp_list.append(fp)
                fn_list.append(fn)

            # ── Step 5: Compute Precision, Recall, F1-Score ──
            precision = precision_score(true_classes, predicted_classes,
                                        average='weighted', zero_division=0,
                                        labels=list(range(num_classes)))
            recall = recall_score(true_classes, predicted_classes,
                                  average='weighted', zero_division=0,
                                  labels=list(range(num_classes)))
            f1 = f1_score(true_classes, predicted_classes,
                          average='weighted', zero_division=0,
                          labels=list(range(num_classes)))

            # ── Step 6: Get detailed classification report ──
            class_report = classification_report(
                true_classes, predicted_classes,
                labels=list(range(num_classes)),
                target_names=self.labels, zero_division=0
            )

            # ── Step 7: Build the report text ──
            report_lines = []
            report_lines.append("=" * 70)
            report_lines.append("       RESNET50 MODEL — VALIDATION REPORT")
            report_lines.append("=" * 70)
            report_lines.append("")
            report_lines.append(f"Overall Test Accuracy : {overall_accuracy:.4f} "
                                f"({overall_accuracy * 100:.2f}%)")
            report_lines.append("")
            report_lines.append("-" * 70)
            report_lines.append("  OVERALL WEIGHTED METRICS")
            report_lines.append("-" * 70)
            report_lines.append(f"  Precision (weighted) : {precision:.4f}")
            report_lines.append(f"  Recall    (weighted) : {recall:.4f}")
            report_lines.append(f"  F1-Score  (weighted) : {f1:.4f}")
            report_lines.append("")
            report_lines.append("-" * 70)
            report_lines.append("  PER-CLASS TP, TN, FP, FN")
            report_lines.append("-" * 70)
            report_lines.append(f"  {'Class':<20s} {'TP':>6s} {'TN':>6s} "
                                f"{'FP':>6s} {'FN':>6s}")
            report_lines.append("  " + "-" * 44)

            for i in range(num_classes):
                report_lines.append(
                    f"  {self.labels[i]:<20s} {tp_list[i]:>6d} {tn_list[i]:>6d} "
                    f"{fp_list[i]:>6d} {fn_list[i]:>6d}"
                )

            total_tp = sum(tp_list)
            total_tn = sum(tn_list)
            total_fp = sum(fp_list)
            total_fn = sum(fn_list)
            report_lines.append("  " + "-" * 44)
            report_lines.append(
                f"  {'TOTAL':<20s} {total_tp:>6d} {total_tn:>6d} "
                f"{total_fp:>6d} {total_fn:>6d}"
            )

            report_lines.append("")
            report_lines.append("-" * 70)
            report_lines.append("  DETAILED CLASSIFICATION REPORT")
            report_lines.append("-" * 70)
            report_lines.append(class_report)

            report_lines.append("")
            report_lines.append("-" * 70)
            report_lines.append("  CONFUSION MATRIX")
            report_lines.append("-" * 70)
            report_lines.append(str(cm))
            report_lines.append("")
            report_lines.append("=" * 70)
            report_lines.append("  END OF REPORT")
            report_lines.append("=" * 70)

            # ── Step 8: Print the report ──
            full_report = "\n".join(report_lines)
            print(full_report)

            # ── Step 9: Save report to file ──
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(full_report)

            print(f"\n  Report saved to: {self.report_file}")
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            line_no = ex_line.tb_lineno if ex_line else 'unknown'
            print(f'Issue is from {line_no} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  MAIN — Entry point of the program
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    try:
        # ── Step 1: Set up paths (auto-detect local vs Colab) ──
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)

        # Smart path detection:
        #  - Local (PyCharm) : dataset is usually at project_root/image_Dataset
        #  - Colab           : dataset might be directly inside base_dir or Colab's /content
        if os.path.isdir(os.path.join(project_root, 'image_Dataset', 'training_data')):
            data_dir = os.path.join(project_root, 'image_Dataset')
        elif os.path.isdir(os.path.join(base_dir, 'image_Dataset', 'training_data')):
            data_dir = os.path.join(base_dir, 'image_Dataset')
        elif os.path.isdir('/content/image_Dataset/training_data'):
            data_dir = '/content/image_Dataset'
        else:
            data_dir = base_dir  # Fallback

        print(f"  Script directory : {base_dir}")
        print(f"  Data directory   : {data_dir}")

        training_data_path = os.path.join(data_dir, 'training_data')
        validation_data_path = os.path.join(data_dir, 'valid_data')
        testing_data_path = os.path.join(data_dir, 'testing_data')

        # ── Step 2: Load data ──
        data_loader = FoodDataLoader(
            training_path=training_data_path,
            validation_path=validation_data_path,
            testing_path=testing_data_path,
            image_size=(256, 256),
            batch_size=34
        )
        data_loader.load_data()

        # ── Guard: Stop if data loading failed or 0 images found ──
        if data_loader.train_data is None or data_loader.val_data is None or data_loader.test_data is None:
            print("\n  ERROR: Data loading failed (0 images found). Cannot proceed.")
            print("  Check that your folder structure looks like:")
            print(f"    {data_dir}/training_data/<class_folders>/")
            print(f"    {data_dir}/valid_data/<class_folders>/")
            print(f"    {data_dir}/testing_data/<class_folders>/")
            sys.exit(1)

        # ── Step 3: Build ResNet50 model ──
        resnet_model = ResNetModel(
            num_classes=34,
            image_size=(256, 256, 3),
            epochs=30
        )
        resnet_model.build_model()

        # ── Guard: Stop if model building failed ──
        if resnet_model.model is None:
            print("\n  ERROR: Model building failed. Cannot proceed.")
            sys.exit(1)

        # ── Step 4: Train the model ──
        resnet_model.train_model(
            train_data=data_loader.train_data,
            val_data=data_loader.val_data
        )

        # ── Guard: Stop if training failed ──
        if resnet_model.history is None:
            print("\n  ERROR: Model training failed. Cannot proceed.")
            sys.exit(1)

        # ── Step 5: Save the trained model ──
        model_save_path = os.path.join(base_dir, 'resnet_model.h5')
        resnet_model.save_model(save_path=model_save_path)

        # ── Step 6: Plot training history ──
        resnet_model.plot_training_history(save_dir=base_dir)

        # ── Step 7: Evaluate and generate report ──
        report_path = os.path.join(base_dir, 'ResNet_Model.txt')
        evaluator = ModelEvaluator(
            model=resnet_model.model,
            test_data=data_loader.test_data,
            labels=data_loader.labels,
            report_file=report_path
        )
        evaluator.evaluate_and_generate_report()

        print("\n  ALL STEPS COMPLETED SUCCESSFULLY FOR RESNET50!")

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        line_no = ex_line.tb_lineno if ex_line else 'unknown'
        print(f'Issue is from {line_no} : due to : {ex_msg}')
