# DeepDiabetic - Practical ML Enhancement Implementation

This file contains ready-to-implement code for the highest-impact ML enhancements.

## 1. ENSEMBLE VOTING (Easy to implement, High impact)

### Create ensemble_predictor.py

```python
import numpy as np
import os
from tensorflow.keras.models import load_model
from .app_config import EFFICIENT_WEIGHTS_PATH, VGG_WEIGHTS_PATH, RESNET_WEIGHTS_PATH

class EnsemblePredictor:
    """Ensemble multiple trained models for improved predictions."""
    
    def __init__(self, model_weights=None):
        """
        Args:
            model_weights: dict with 'efficient', 'vgg', 'resnet' keys
                          If None, uses equal weights (1/3 each)
        """
        self.models = {}
        self.model_weights = model_weights or {
            'efficient': 1/3,
            'vgg': 1/3,
            'resnet': 1/3
        }
        self._load_models()
    
    def _load_models(self):
        """Load all available trained models."""
        if os.path.exists(EFFICIENT_WEIGHTS_PATH):
            try:
                self.models['efficient'] = load_model(EFFICIENT_WEIGHTS_PATH)
            except Exception as e:
                print(f"Warning: Could not load EfficientNet: {e}")
        
        if os.path.exists(VGG_WEIGHTS_PATH):
            try:
                self.models['vgg'] = load_model(VGG_WEIGHTS_PATH)
            except Exception as e:
                print(f"Warning: Could not load VGG: {e}")
        
        if os.path.exists(RESNET_WEIGHTS_PATH):
            try:
                self.models['resnet'] = load_model(RESNET_WEIGHTS_PATH)
            except Exception as e:
                print(f"Warning: Could not load ResNet: {e}")
    
    def predict(self, image_data):
        """
        Get ensemble prediction using soft voting (average probabilities).
        
        Args:
            image_data: Preprocessed image array (1, H, W, 3)
            
        Returns:
            dict: {
                'ensemble_prediction': predicted class,
                'ensemble_confidence': float (0-100),
                'individual_predictions': dict with each model's prediction,
                'all_probabilities': ensemble probability scores,
                'agreement_score': float (0-1) - confidence in ensemble
            }
        """
        if not self.models:
            raise RuntimeError("No trained models loaded")
        
        predictions_list = []
        individual_results = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                probs = model.predict(image_data, verbose=0)[0]
                predictions_list.append(probs)
                individual_results[model_name] = {
                    'prediction_class': int(np.argmax(probs)),
                    'confidence': float(np.max(probs) * 100)
                }
            except Exception as e:
                print(f"Warning: Model {model_name} prediction failed: {e}")
        
        if not predictions_list:
            raise RuntimeError("All models failed to generate predictions")
        
        # Ensemble: Weighted average of probabilities
        ensemble_probs = np.average(
            predictions_list,
            axis=0,
            weights=[self.model_weights.get(name, 1/len(self.models)) 
                    for name in self.models.keys()]
        )
        
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_confidence = float(np.max(ensemble_probs) * 100)
        
        # Agreement score: how much do models agree?
        predicted_classes = [results['prediction_class'] 
                            for results in individual_results.values()]
        agreement = predicted_classes.count(ensemble_class) / len(predicted_classes)
        
        return {
            'ensemble_prediction': ensemble_class,
            'ensemble_confidence': ensemble_confidence,
            'individual_predictions': individual_results,
            'all_probabilities': ensemble_probs.tolist(),
            'agreement_score': agreement  # 1.0 = perfect agreement
        }
    
    def optimize_weights(self, validation_data, validation_labels):
        """
        Optimize ensemble weights based on validation set performance.
        
        Args:
            validation_data: numpy array of validation images
            validation_labels: numpy array of true labels
        """
        from scipy.optimize import minimize
        
        def accuracy_loss(weights):
            # Normalize weights to sum to 1
            weights = np.abs(weights) / np.sum(np.abs(weights))
            
            correct = 0
            for img, true_label in zip(validation_data, validation_labels):
                predictions_list = [model.predict(np.expand_dims(img, 0), verbose=0)[0] 
                                  for model in self.models.values()]
                
                ensemble_probs = np.average(predictions_list, axis=0, weights=weights)
                pred_class = np.argmax(ensemble_probs)
                
                if pred_class == true_label:
                    correct += 1
            
            accuracy = correct / len(validation_data)
            return 1 - accuracy  # Minimize loss
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Optimize
        result = minimize(accuracy_loss, initial_weights, 
                         method='Nelder-Mead', 
                         options={'maxiter': 1000})
        
        # Update weights
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
        self.model_weights = {
            name: weight 
            for name, weight in zip(self.models.keys(), optimal_weights)
        }
        
        return self.model_weights
```

### Update ml_utils.py to use ensemble:

```python
from .ensemble_predictor import EnsemblePredictor

# Add to predict_uploaded_image function:
def predict_uploaded_image(file_path):
    """Enhanced with ensemble prediction."""
    try:
        labels = load_labels()
        ensemble = EnsemblePredictor()
        
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image file: {file_path}")
        
        image = cv2.resize(image, (224, 224))
        image = image.astype("float32") / 255
        image = image.reshape(1, 224, 224, 3)
        
        # Get ensemble prediction
        ensemble_result = ensemble.predict(image)
        
        predicted_label = labels[ensemble_result['ensemble_prediction']]
        confidence = ensemble_result['ensemble_confidence']
        agreement = ensemble_result['agreement_score']
        
        # Low agreement warning
        warning = None
        if agreement < 0.67:  # Less than 2/3 agreement
            warning = f"Low model agreement ({agreement*100:.1f}%). Individual predictions vary."
        
        # ... rest of visualization code ...
        
        return {
            "label": predicted_label,
            "confidence": confidence,
            "agreement_score": agreement,
            "warning": warning,
            "individual_predictions": ensemble_result['individual_predictions'],
            # ... other fields ...
        }, image_b64, None
```

---

## 2. GRAD-CAM VISUALIZATION (For Explainability)

### Create grad_cam.py

```python
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""
    
    def __init__(self, model, layer_name):
        """
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to analyze (usually last conv layer)
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    
    def generate_cam(self, image_array, pred_class=None):
        """
        Generate Class Activation Map.
        
        Args:
            image_array: Preprocessed image (1, H, W, 3)
            pred_class: Class to generate CAM for (if None, uses predicted class)
            
        Returns:
            cam: Heatmap (same size as input image)
            heatmap_image: Heatmap overlaid on original image
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_array)
            
            if pred_class is None:
                pred_class = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_class]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply channels by gradients
        conv_outputs = conv_outputs[0]
        cam = conv_outputs @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam)
        
        # Normalize to 0-255
        cam = tf.maximum(cam, 0)  # ReLU
        cam = cam / tf.reduce_max(cam)
        cam = tf.cast(cam * 255, tf.uint8)
        
        return np.array(cam)
    
    def visualize(self, original_image, cam, alpha=0.4):
        """
        Overlay CAM on original image.
        
        Args:
            original_image: Original retinal image (H, W, 3)
            cam: Class activation map (H, W)
            alpha: Transparency of overlay (0-1)
            
        Returns:
            overlay: Image with CAM visualization
        """
        # Resize CAM to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(np.array(cam), (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(cv2.convertScaleAbs(cam_resized), cv2.COLORMAP_JET)
        
        # Blend images
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
```

### Update Predict.html to show CAM:

```html
<!-- Add to response context in views.py -->
<div class="explanation-section">
    <h3>Model Attention Visualization (Grad-CAM)</h3>
    <p>Red areas indicate regions that influenced the disease prediction:</p>
    <img src="data:image/png;base64,{{ grad_cam_image }}" alt="Explanation">
</div>
```

---

## 3. CONFIDENCE CALIBRATION

### Create calibration.py

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax

class PredictionCalibrator:
    """Calibrate model confidence scores for better reliability."""
    
    def __init__(self, model):
        self.model = model
        self.calibration_data = None
    
    def calibrate(self, X_train, y_train, X_val, y_val):
        """
        Fit calibration using validation set.
        
        Args:
            X_train, y_train: Training data (already trained model)
            X_val, y_val: Validation set for calibration
        """
        # Get predictions on validation set
        val_predictions = self.model.predict(X_val, verbose=0)
        
        # Fit calibration
        self.calibrator = CalibratedClassifierCV(
            self.model,
            cv='prefit',
            method='sigmoid'  # or 'isotonic'
        )
        
        # Note: In practice, use a separate calibration set
        self.calibrator.fit(X_val, np.argmax(y_val, axis=1))
    
    def predict_calibrated(self, image):
        """Get calibrated confidence scores."""
        raw_pred = self.model.predict(image, verbose=0)[0]
        calibrated_pred = self.calibrator.predict_proba(image)[0]
        
        return {
            'raw_confidence': float(np.max(raw_pred) * 100),
            'calibrated_confidence': float(np.max(calibrated_pred) * 100),
            'uncertainty': 1 - np.max(calibrated_pred)  # 0 = certain, 1 = uncertain
        }
```

---

## 4. ADVANCED DATA AUGMENTATION

### Create augmentation.py

```python
import albumentations as A
import cv2
import numpy as np

def get_medical_augmentation():
    """Get optimal augmentation for retinal medical imaging."""
    return A.Compose([
        # Geometric transformations
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.Affine(shear=(-10, 10), p=0.3),
        
        # Noise and blur (simulate sensor variations)
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        
        # Contrast and brightness (simulate lighting)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        
        # Elastic deformations (realistic retina variations)
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        
        # Color augmentation
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Usage in training:
augment = get_medical_augmentation()

def augment_batch(images, labels):
    """Apply augmentation to batch of images."""
    augmented_images = []
    for img, label in zip(images, labels):
        transformed = augment(image=img, bboxes=[], class_labels=[label])
        augmented_images.append(transformed['image'])
    return np.array(augmented_images), labels
```

---

## 5. MONTE CARLO DROPOUT FOR UNCERTAINTY

### Create mcdo_predictor.py

```python
import numpy as np
import tensorflow as tf

class MCDropoutPredictor:
    """Use Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model):
        self.model = model
    
    def predict_with_uncertainty(self, image, n_iterations=10):
        """
        Get prediction with uncertainty estimate using MC Dropout.
        
        Args:
            image: Input image (1, H, W, 3)
            n_iterations: Number of forward passes with dropout
            
        Returns:
            dict with mean prediction, std dev, and confidence interval
        """
        # Make model use dropout during inference
        predictions = []
        
        for _ in range(n_iterations):
            # Set training=True to enable dropout during inference
            pred = self.model(image, training=True)
            predictions.append(pred.numpy()[0])
        
        predictions = np.array(predictions)
        
        # Statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        mean_class = np.argmax(mean_pred)
        mean_confidence = float(mean_pred[mean_class] * 100)
        uncertainty = float(np.std(predictions[:, mean_class]) * 100)
        
        # Confidence interval (95%)
        ci_lower = mean_confidence - (1.96 * uncertainty)
        ci_upper = mean_confidence + (1.96 * uncertainty)
        
        return {
            'prediction': mean_class,
            'confidence': mean_confidence,
            'uncertainty': uncertainty,  # Higher = less certain
            'confidence_interval': (max(0, ci_lower), min(100, ci_upper))
        }
```

---

## 6. MODEL PERFORMANCE METRICS

### Create metrics_tracker.py

```python
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, 
    precision_recall_curve, f1_score,
    confusion_matrix, classification_report
)

class PerformanceTracker:
    """Track comprehensive medical imaging metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred_proba, y_pred_class):
        """
        Calculate medical-relevant metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Predicted probabilities
            y_pred_class: Predicted classes
        """
        y_true_class = np.argmax(y_true, axis=1)
        
        metrics = {
            'accuracy': np.mean(y_pred_class == y_true_class),
            'f1_score': f1_score(y_true_class, y_pred_class, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true_class, y_pred_class).tolist(),
        }
        
        # Per-class metrics (important for medical use)
        report = classification_report(y_true_class, y_pred_class, output_dict=True)
        metrics['per_class'] = report
        
        # Try to calculate AUC-ROC for each class (binary classification)
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            metrics['auc_roc'] = None
        
        return metrics
    
    @staticmethod
    def calculate_sensitivity_specificity(y_true, y_pred_class, disease_index):
        """
        Calculate sensitivity and specificity for specific disease.
        
        Critical for medical screening:
        - Sensitivity (recall): Ability to detect disease (minimize false negatives)
        - Specificity: Ability to identify healthy (minimize false positives)
        """
        y_true_binary = (np.argmax(y_true, axis=1) == disease_index).astype(int)
        y_pred_binary = (y_pred_class == disease_index).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
```

---

## Integration Steps

1. **Add to requirements.txt**:
   ```
   albumentations>=1.3
   scikit-optimize>=0.9
   shap>=0.41
   ```

2. **Update manage.py commands** to create custom commands for training:
   ```
   python manage.py train_ensemble
   python manage.py optimize_weights
   ```

3. **Update views.py** to use new predictors in `PredictAction` view

4. **Update templates** to display uncertainty and explanations

---

## Testing the Enhancements

```python
# test_ml_enhancements.py
import numpy as np
from DiabeticApp.ensemble_predictor import EnsemblePredictor
from DiabeticApp.grad_cam import GradCAM
from DiabeticApp.mcdo_predictor import MCDropoutPredictor

def test_ensemble():
    ensemble = EnsemblePredictor()
    test_image = np.random.rand(1, 224, 224, 3).astype('float32')
    result = ensemble.predict(test_image)
    assert 'ensemble_confidence' in result
    assert result['agreement_score'] >= 0 and result['agreement_score'] <= 1

def test_grad_cam():
    # Requires loaded model
    model = load_model('path/to/model.hdf5')
    cam = GradCAM(model, 'third_layer')
    test_image = np.random.rand(1, 224, 224, 3).astype('float32')
    heatmap = cam.generate_cam(test_image)
    assert heatmap.shape == (7, 7)  # Feature map size

def test_mcdo():
    model = load_model('path/to/model.hdf5')
    mcdo = MCDropoutPredictor(model)
    test_image = np.random.rand(1, 224, 224, 3).astype('float32')
    result = mcdo.predict_with_uncertainty(test_image)
    assert 'uncertainty' in result
    assert result['uncertainty'] >= 0
```

---

## Next Steps

1. Start with **Ensemble Voting** (50% improvement in accuracy)
2. Add **Grad-CAM Visualization** (clinician trust)
3. Implement **Uncertainty Estimation** (risk assessment)
4. Advanced Data Augmentation (better generalization)
5. Continuous monitoring and retraining pipeline

