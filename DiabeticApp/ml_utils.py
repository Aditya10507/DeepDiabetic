import base64
import io
import os
import uuid
import numpy as np

from .app_config import CM_PATH
from .app_config import DATA_PATH
from .app_config import DATASET_DIR
from .app_config import EFFICIENT_WEIGHTS_PATH
from .app_config import IMAGE_SIZE
from .app_config import LEGACY_CM_PATH
from .app_config import LEGACY_METRIC_PATH
from .app_config import LEGACY_EFFICIENT_WEIGHTS_PATH
from .app_config import METRIC_PATH
from .app_config import STATIC_DIR
from .app_config import X_PATH
from .app_config import Y_PATH


_dataset_cache = {
    "labels": None,
    "X": None,
    "Y": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
}


def _load_ml_dependencies():
    import cv2
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

    return {
        "cv2": cv2,
        "plt": plt,
        "sns": sns,
        "train_test_split": train_test_split,
        "Conv2D": Conv2D,
        "Dense": Dense,
        "Flatten": Flatten,
        "Input": Input,
        "MaxPooling2D": MaxPooling2D,
        "Sequential": Sequential,
        "load_model": load_model,
        "to_categorical": to_categorical,
    }

DISEASE_DETAILS = {
    "Cataract": {
        "summary": "Lens opacity is suspected. Cataracts can reduce clarity, contrast, and night vision over time.",
        "recommendation": "Arrange an ophthalmology evaluation to confirm severity and discuss treatment options.",
        "tone_class": "result-cataract",
    },
    "DME": {
        "summary": "Possible diabetic macular edema is indicated. This can affect the central retina and impact sharp vision.",
        "recommendation": "Prompt retina specialist review is recommended, especially if the patient has diabetes symptoms or blurred central vision.",
        "tone_class": "result-dme",
    },
    "DR": {
        "summary": "Features compatible with diabetic retinopathy are present. DR can progress silently before vision changes become obvious.",
        "recommendation": "A diabetic eye examination is recommended to confirm stage and guide early management.",
        "tone_class": "result-dr",
    },
    "Glaucoma": {
        "summary": "Glaucoma-related changes are suspected. This can damage the optic nerve gradually and may not show early symptoms.",
        "recommendation": "Follow up with an eye specialist for pressure testing, optic nerve assessment, and visual field evaluation.",
        "tone_class": "result-glaucoma",
    },
}


def load_labels():
    """Load disease class labels from dataset directory structure.
    
    Walks through DATASET_DIR and extracts unique folder names as disease labels.
    Results are cached for performance optimization.
    
    Returns:
        list: List of disease class labels (e.g., ['Cataract', 'DME', 'DR', 'Glaucoma'])
    """
    if _dataset_cache["labels"] is not None:
        return _dataset_cache["labels"]
    labels = []
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    for root, dirs, files in os.walk(DATASET_DIR):
        for _ in files:
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    _dataset_cache["labels"] = labels
    return labels


def get_label_index(name):
    """Get the index of a disease label.
    
    Args:
        name (str): Disease label name to find
        
    Returns:
        int: Index of the label, or -1 if not found
    """
    labels = load_labels()
    for index, label in enumerate(labels):
        if label == name:
            return index
    return -1


def build_dataset_arrays():
    """Build feature and target arrays from dataset images.
    
    Walks through DATASET_DIR, reads images, resizes them to IMAGE_SIZE,
    and assigns labels. Results are saved to X_PATH and Y_PATH.
    
    Returns:
        tuple: (features_array, targets_array) as numpy arrays
        
    Raises:
        FileNotFoundError: If dataset directory not found
    """
    ml = _load_ml_dependencies()
    cv2 = ml["cv2"]
    features = []
    targets = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            if "Thumbs.db" in filename:
                continue
            label_name = os.path.basename(root)
            image_path = os.path.join(root, filename)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image = cv2.resize(image, IMAGE_SIZE)
                features.append(image)
                targets.append(get_label_index(label_name))
            except Exception as e:
                # Log but continue processing other images
                print(f"Warning: Failed to process image {image_path}: {str(e)}")
                continue
    features = np.asarray(features)
    targets = np.asarray(targets)
    np.save(X_PATH, features)
    np.save(Y_PATH, targets)
    return features, targets


def ensure_dataset_loaded():
    """Ensure dataset is loaded and preprocessed.
    
    Loads dataset from saved files (or builds from images), performs preprocessing
    (normalization, shuffling, one-hot encoding), and splits into train/test sets.
    Results are cached for performance optimization.
    
    Returns:
        dict: Cache dictionary with X, Y, X_train, X_test, y_train, y_test arrays
    """
    if _dataset_cache["X"] is not None and _dataset_cache["X_train"] is not None:
        return _dataset_cache

    try:
        ml = _load_ml_dependencies()
        if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
            X = np.load(X_PATH)
            Y = np.load(Y_PATH)
        else:
            X, Y = build_dataset_arrays()

        X = X.astype("float32") / 255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = ml["to_categorical"](Y)
        X_train, X_test, y_train, y_test = ml["train_test_split"](X, Y, test_size=0.2)

        if os.path.exists(DATA_PATH):
            data = np.load(DATA_PATH, allow_pickle=True)
            X_train, X_test, y_train, y_test = data

        _dataset_cache["X"] = X
        _dataset_cache["Y"] = Y
        _dataset_cache["X_train"] = X_train
        _dataset_cache["X_test"] = X_test
        _dataset_cache["y_train"] = y_train
        _dataset_cache["y_test"] = y_test
        return _dataset_cache
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")


def load_metrics():
    """Load model evaluation metrics and confusion matrices.
    
    Attempts to load metrics from saved files (current path first, then legacy path).
    Returns default zero-filled confusion matrices if no metrics found.
    
    Returns:
        dict: Dictionary with keys 'accuracy', 'precision', 'recall', 'fscore' (lists)
              and 'efficient_cm', 'vgg_cm', 'resnet_cm' (confusion matrices)
    """
    try:
        labels = load_labels()
        default_cm = np.zeros((len(labels), len(labels)), dtype=int)
        metrics = {
            "model_names": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "fscore": [],
            "efficient_cm": default_cm,
            "vgg_cm": default_cm,
            "resnet_cm": default_cm,
        }

        if os.path.exists(METRIC_PATH) and os.path.exists(CM_PATH):
            metric = np.load(METRIC_PATH, allow_pickle=True)
            metrics["model_names"].append("ResNet152V2")
            metrics["accuracy"].append(round(float(metric[0]) * 100, 3))
            metrics["precision"].append(round(float(metric[1]), 3))
            metrics["recall"].append(round(float(metric[2]), 3))
            metrics["fscore"].append(round(float(metric[3]), 3))
            metrics["resnet_cm"] = np.load(CM_PATH)
            return metrics

        if os.path.exists(LEGACY_METRIC_PATH) and os.path.exists(LEGACY_CM_PATH):
            legacy_metric = np.load(LEGACY_METRIC_PATH, allow_pickle=True)
            metrics["model_names"].append("Legacy CNN")
            metrics["accuracy"].append(round(float(legacy_metric[0]) * 100, 3))
            metrics["precision"].append(round(float(legacy_metric[1]), 3))
            metrics["recall"].append(round(float(legacy_metric[2]), 3))
            metrics["fscore"].append(round(float(legacy_metric[3]), 3))
            metrics["resnet_cm"] = np.load(LEGACY_CM_PATH)

        return metrics
    except Exception as e:
        print(f"Warning: Failed to load metrics: {str(e)}")
        # Return basic structure with zeros
        labels = load_labels() if os.path.exists(DATASET_DIR) else []
        default_cm = np.zeros((len(labels) if labels else 4, len(labels) if labels else 4), dtype=int)
        return {
            "model_names": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "fscore": [],
            "efficient_cm": default_cm,
            "vgg_cm": default_cm,
            "resnet_cm": default_cm,
        }


def save_uploaded_image(upload):
    """Save an uploaded image file to static directory.
    
    Args:
        upload: Django UploadedFile object with 'name' and 'read()' method
        
    Returns:
        str: Path to saved file
        
    Raises:
        IOError: If file cannot be saved
    """
    try:
        original_name = os.path.basename(upload.name or "")
        _, extension = os.path.splitext(original_name)
        extension = extension.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise ValueError("Unsupported image type. Please upload a JPG, PNG, BMP, or TIFF image.")

        os.makedirs(STATIC_DIR, exist_ok=True)
        filename = f"upload_{uuid.uuid4().hex}{extension}"
        file_path = os.path.abspath(os.path.join(STATIC_DIR, filename))
        static_dir_path = os.path.abspath(STATIC_DIR)
        if os.path.commonpath([static_dir_path, file_path]) != static_dir_path:
            raise ValueError("Invalid upload path.")

        with open(file_path, "wb") as file:
            file.write(upload.read())
        return file_path
    except (IOError, ValueError) as e:
        raise type(e)(f"Failed to save uploaded image: {str(e)}")


def build_classifier(input_shape, num_classes):
    """Build a simple CNN classifier model.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.models.Sequential: Compiled CNN model ready for training/inference
    """
    ml = _load_ml_dependencies()
    model = ml["Sequential"](
        [
            ml["Input"](shape=input_shape),
            ml["Conv2D"](32, (3, 3), activation="relu"),
            ml["MaxPooling2D"](pool_size=(2, 2)),
            ml["Conv2D"](32, (3, 3), activation="relu"),
            ml["MaxPooling2D"](pool_size=(2, 2)),
            ml["Flatten"](),
            ml["Dense"](units=256, activation="relu"),
            ml["Dense"](units=num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def predict_uploaded_image(file_path):
    """Load a trained model and predict disease from uploaded retina image.
    
    Loads model weights, preprocesses the image, runs inference, and generates
    visualizations with clinical recommendations based on disease predictions.
    
    Args:
        file_path (str): Path to uploaded image file
        
    Returns:
        tuple: (prediction_dict, base64_image, error_message)
            - prediction_dict: Dict with keys 'label', 'confidence', 'summary', 
              'recommendation', 'tone_class', 'prediction_text', 'model_variant', 
              'input_size', 'uploaded_b64', or None if error
            - base64_image: Base64 encoded prediction visualization or None
            - error_message: Error description if prediction failed, else None
            
    Raises:
        FileNotFoundError: If image file cannot be read
    """
    try:
        ml = _load_ml_dependencies()
        cv2 = ml["cv2"]
        plt = ml["plt"]
        labels = load_labels()
        model = None
        inference_size = IMAGE_SIZE
        model_variant = "224x224 enhanced model"

        if os.path.exists(EFFICIENT_WEIGHTS_PATH):
            try:
                model = ml["load_model"](EFFICIENT_WEIGHTS_PATH)
            except Exception:
                model = build_classifier((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), len(labels))
                model.load_weights(EFFICIENT_WEIGHTS_PATH)
        elif os.path.exists(LEGACY_EFFICIENT_WEIGHTS_PATH):
            inference_size = (32, 32)
            model_variant = "32x32 legacy fallback model"
            model = build_classifier((32, 32, 3), len(labels))
            model.load_weights(LEGACY_EFFICIENT_WEIGHTS_PATH)
        else:
            return None, None, "No trained model weights are available yet. Train a model first to test prediction."

        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image file: {file_path}")
        
        image = cv2.resize(image, inference_size)
        image = image.astype("float32") / 255
        image = image.reshape(1, inference_size[0], inference_size[1], 3)
        probabilities = model.predict(image, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        predicted_label = labels[prediction]
        confidence = round(float(probabilities[prediction]) * 100, 2)
        details = DISEASE_DETAILS.get(
            predicted_label,
            {
                "summary": "The model detected a disease pattern in the uploaded retina image.",
                "recommendation": "Please confirm the result with a qualified eye specialist.",
                "tone_class": "result-default",
            },
        )

        preview = cv2.imread(file_path)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = cv2.resize(preview, (400, 300))
        prediction_text = "Predicted As: " + predicted_label
        cv2.putText(preview, prediction_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        plt.imshow(preview)
        plt.title(prediction_text)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.clf()
        plt.cla()

        uploaded_preview = cv2.imread(file_path)
        uploaded_preview = cv2.cvtColor(uploaded_preview, cv2.COLOR_BGR2RGB)
        uploaded_preview = cv2.resize(uploaded_preview, (400, 300))
        plt.imshow(uploaded_preview)
        plt.axis("off")
        uploaded_buffer = io.BytesIO()
        plt.savefig(uploaded_buffer, format="png", bbox_inches="tight", pad_inches=0.05)
        uploaded_b64 = base64.b64encode(uploaded_buffer.getvalue()).decode()
        plt.clf()
        plt.cla()
        return {
            "label": predicted_label,
            "confidence": confidence,
            "summary": details["summary"],
            "recommendation": details["recommendation"],
            "tone_class": details["tone_class"],
            "prediction_text": prediction_text,
            "model_variant": model_variant,
            "input_size": f"{inference_size[0]}x{inference_size[1]}",
            "uploaded_b64": uploaded_b64,
        }, image_b64, None
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"


def build_metrics_plot():
    """Build and visualize confusion matrices for all trained models.
    
    Creates a 1x3 subplot figure showing confusion matrices for EfficientNetB0,
    VGG16, and ResNet152V2 models as heatmaps. Matrices are loaded from saved
    metric files.
    
    Returns:
        tuple: (metrics_dict, base64_image)
            - metrics_dict: Dictionary with accuracy, precision, recall, fscore lists
              and confusion matrices for each model
            - base64_image: Base64 encoded PNG visualization of all three confusion matrices
    """
    try:
        ml = _load_ml_dependencies()
        plt = ml["plt"]
        sns = ml["sns"]
        labels = load_labels()
        metrics = load_metrics()
        available_confusion_matrices = []
        if metrics["model_names"]:
            available_confusion_matrices.append((metrics["model_names"][0], metrics["resnet_cm"]))
        else:
            available_confusion_matrices.append(("No saved model metrics", metrics["resnet_cm"]))

        figure, axis = plt.subplots(
            nrows=1,
            ncols=len(available_confusion_matrices),
            figsize=(max(4, 4 * len(available_confusion_matrices)), 4),
        )
        if not isinstance(axis, np.ndarray):
            axis = np.asarray([axis])

        for subplot, (model_name, matrix) in zip(axis, available_confusion_matrices):
            subplot.set_title(model_name)
            heatmap = sns.heatmap(
                matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                cmap="viridis",
                fmt="g",
                ax=subplot,
            )
            heatmap.set_ylim([0, len(labels)])

        figure.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.clf()
        plt.cla()
        return metrics, image_b64
    except Exception as e:
        print(f"Warning: Failed to build metrics plot: {str(e)}")
        # Return empty metrics structure
        labels = load_labels() if os.path.exists(DATASET_DIR) else []
        default_cm = np.zeros((len(labels) if labels else 4, len(labels) if labels else 4), dtype=int)
        return {
            "model_names": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "fscore": [],
            "efficient_cm": default_cm,
            "vgg_cm": default_cm,
            "resnet_cm": default_cm,
        }, None
