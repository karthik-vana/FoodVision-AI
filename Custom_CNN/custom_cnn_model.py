"""
custom_cnn_model.py
-------------------
Trains a Custom CNN model from scratch for food image classification (34 classes).
Uses OOP (Classes, Objects, Constructors) and Exception Handling.
Generates a validation report and saves it to Custom_Model.txt.

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

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
            print("  LOADING DATA")
            print("=" * 60)

            # ── Data augmentation rules for training ──
            train_data_rules = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=20,
                shear_range=0.2,
                zoom_range=0.2
            )

            # ── Only rescaling for validation and testing ──
            val_data_rules = ImageDataGenerator(rescale=1.0 / 255.0)
            test_data_rules = ImageDataGenerator(rescale=1.0 / 255.0)

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
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  CLASS : CustomCNNModel
#  Purpose : Build, train, evaluate, and save a Custom CNN
# ════════════════════════════════════════════════════════════════
class CustomCNNModel:
    """Builds and trains a Custom CNN from scratch for food classification."""

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
        """Builds the Custom CNN architecture using Sequential API."""
        try:
            print("\n" + "=" * 60)
            print("  BUILDING CUSTOM CNN MODEL")
            print("=" * 60)

            self.model = Sequential()

            # ── Convolutional Block 1: 128 filters ──
            self.model.add(Conv2D(128, (3, 3), activation='relu',
                                  kernel_initializer='he_uniform',
                                  strides=1, padding='same',
                                  input_shape=self.image_size))
            self.model.add(MaxPool2D(pool_size=(2, 2)))

            # ── Convolutional Block 2: 16 filters ──
            self.model.add(Conv2D(16, (3, 3), activation='relu',
                                  strides=1, padding='same',
                                  kernel_initializer='he_uniform'))
            self.model.add(MaxPool2D(pool_size=(2, 2)))

            # ── Convolutional Block 3: 4 filters ──
            self.model.add(Conv2D(4, (3, 3), activation='relu',
                                  strides=1, padding='same',
                                  kernel_initializer='he_uniform'))
            self.model.add(MaxPool2D(pool_size=(2, 2)))

            # ── Flatten layer ──
            self.model.add(Flatten())

            # ── Fully Connected (Dense) Layers ──
            self.model.add(Dense(units=256, activation='relu',
                                 kernel_initializer='he_uniform'))
            self.model.add(Dense(units=128, activation='relu',
                                 kernel_initializer='he_uniform'))
            self.model.add(Dense(units=64, activation='relu',
                                 kernel_initializer='he_uniform'))

            # ── Output Layer: 34 classes with softmax ──
            self.model.add(Dense(units=self.num_classes, activation='softmax',
                                 kernel_initializer='glorot_uniform'))

            # ── Compile the model ──
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # ── Print model summary ──
            self.model.summary()
            print("  Custom CNN model built successfully!")
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def train_model(self, train_data, val_data):
        """
        Trains the Custom CNN model.

        Parameters
        ----------
        train_data : DirectoryIterator — training data generator
        val_data   : DirectoryIterator — validation data generator
        """
        try:
            print("\n" + "=" * 60)
            print(f"  TRAINING CUSTOM CNN MODEL ({self.epochs} epochs)")
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
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def save_model(self, save_path="custom_cnn_model.h5"):
        """Saves the trained model to disk."""
        try:
            self.model.save(save_path)
            print(f"\n  Model saved to: {save_path}")

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

    def plot_training_history(self):
        """Plots training and validation accuracy/loss graphs."""
        try:
            epochs_ran = len(self.history.history['accuracy'])
            epoch_range = np.arange(1, epochs_ran + 1)

            plt.figure(figsize=(12, 4))

            # ── Plot 1: Training Performance ──
            plt.subplot(1, 2, 1)
            plt.title('Training Performance')
            plt.plot(epoch_range, self.history.history['accuracy'],
                     color='g', label='Train Accuracy')
            plt.plot(epoch_range, self.history.history['loss'],
                     color='r', label='Train Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            # ── Plot 2: Validation Performance ──
            plt.subplot(1, 2, 2)
            plt.title('Validation Performance')
            plt.plot(epoch_range, self.history.history['val_accuracy'],
                     color='g', label='Val Accuracy')
            plt.plot(epoch_range, self.history.history['val_loss'],
                     color='r', label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()
            plt.savefig('custom_cnn_training_plot.png', dpi=150)
            plt.show()
            print("  Training plot saved to: custom_cnn_training_plot.png")

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  CLASS : ModelEvaluator
#  Purpose : Evaluate model and generate validation report
# ════════════════════════════════════════════════════════════════
class ModelEvaluator:
    """Evaluates a trained model and generates a detailed validation report."""

    def __init__(self, model, test_data, labels, report_file="Custom_Model.txt"):
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
            print("  EVALUATING MODEL ON TEST DATA")
            print("=" * 60)

            # ── Step 1: Get predictions ──
            predictions = self.model.predict(self.test_data)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = self.test_data.classes

            # ── Step 2: Compute overall accuracy ──
            overall_accuracy = accuracy_score(true_classes, predicted_classes)

            # ── Step 3: Compute confusion matrix ──
            cm = confusion_matrix(true_classes, predicted_classes)

            # ── Step 4: Compute TP, TN, FP, FN for each class ──
            num_classes = len(self.labels)
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
                                        average='weighted', zero_division=0)
            recall = recall_score(true_classes, predicted_classes,
                                  average='weighted', zero_division=0)
            f1 = f1_score(true_classes, predicted_classes,
                          average='weighted', zero_division=0)

            # ── Step 6: Get detailed classification report ──
            class_report = classification_report(
                true_classes, predicted_classes,
                target_names=self.labels, zero_division=0
            )

            # ── Step 7: Build the report text ──
            report_lines = []
            report_lines.append("=" * 70)
            report_lines.append("       CUSTOM CNN MODEL — VALIDATION REPORT")
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
            with open(self.report_file, 'w') as f:
                f.write(full_report)

            print(f"\n  Report saved to: {self.report_file}")
            print("=" * 60)

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')


# ════════════════════════════════════════════════════════════════
#  MAIN — Entry point of the program
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    try:
        # ── Step 1: Set up paths ──
        base_dir = os.path.dirname(os.path.abspath(__file__))
        training_data_path = os.path.join(base_dir, 'image_Dataset', 'training_data')
        validation_data_path = os.path.join(base_dir, 'image_Dataset', 'valid_data')
        testing_data_path = os.path.join(base_dir, 'image_Dataset', 'testing_data')

        # ── Step 2: Load data ──
        data_loader = FoodDataLoader(
            training_path=training_data_path,
            validation_path=validation_data_path,
            testing_path=testing_data_path,
            image_size=(256, 256),
            batch_size=34
        )
        data_loader.load_data()

        # ── Step 3: Build Custom CNN model ──
        cnn_model = CustomCNNModel(
            num_classes=34,
            image_size=(256, 256, 3),
            epochs=30
        )
        cnn_model.build_model()

        # ── Step 4: Train the model ──
        cnn_model.train_model(
            train_data=data_loader.train_data,
            val_data=data_loader.val_data
        )

        # ── Step 5: Save the trained model ──
        model_save_path = os.path.join(base_dir, 'custom_cnn_model.h5')
        cnn_model.save_model(save_path=model_save_path)

        # ── Step 6: Plot training history ──
        cnn_model.plot_training_history()

        # ── Step 7: Evaluate and generate report ──
        report_path = os.path.join(base_dir, 'Custom_Model.txt')
        evaluator = ModelEvaluator(
            model=cnn_model.model,
            test_data=data_loader.test_data,
            labels=data_loader.labels,
            report_file=report_path
        )
        evaluator.evaluate_and_generate_report()

        print("\n  ALL STEPS COMPLETED SUCCESSFULLY FOR CUSTOM CNN!")

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')
