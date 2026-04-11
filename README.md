# DeepDiabetic: An Identification System of Diabetic Eye Diseases Using Deep Neural Networks

This project is a Django-based web application for identifying diabetic eye diseases using deep neural networks. It allows users to upload retina images and get a prediction of whether the image shows signs of Cataract, DME (Diabetic Macular Edema), DR (Diabetic Retinopathy), or Glaucoma.

## Features

-   User registration and authentication.
-   Dashboard for uploading retina images for prediction.
-   Comparison of different deep learning models (EfficientNetB0, VGG16, ResNet152V2).
-   Visualization of model performance metrics and confusion matrices.

## Project Structure

```
.
├── Diabetic/         # Django project folder
├── DiabeticApp/      # Main Django app
│   ├── migrations/
│   ├── static/
│   ├── templates/
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── ml_utils.py   # Core ML logic
│   ├── urls.py
│   └── views.py
├── model/            # Trained models and data arrays (ignored by git)
├── Dataset/          # Image dataset (ignored by git)
├── manage.py         # Django's command-line utility
├── requirements.txt  # Python dependencies
├── render.yaml       # Deployment configuration for Render
└── testtrain.py      # Script for training the models
```

## Dataset

The model was trained on the "Eye Diseases Classification" dataset from Kaggle. You can find the dataset here: [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

## Local Development

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```

4.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

## Deployment on Render

This project is configured for deployment on [Render](https://render.com/).

### Deployment Steps

1.  Push the deployable project files to a GitHub repository.
2.  Create a new "Web Service" on Render and connect it to your GitHub repository.
3.  Render will automatically detect the `render.yaml` file and configure the service.
4.  Attach the persistent disk defined in `render.yaml`.
5.  Deploy the service.

### Environment Variables

The following environment variables are required for deployment:

-   `SECRET_KEY`: A secret key for Django. Render can generate this for you.
-   `DEBUG`: Set to `False` in production.
-   `APP_DB_BACKEND`: `sqlite` (default) or `mysql`.
-   `ALLOWED_HOSTS`: A comma-separated list of allowed hostnames (e.g., `.render.com`).

For optional MySQL support, you also need to set:

-   `MYSQL_HOST`
-   `MYSQL_PORT`
-   `MYSQL_USER`
-   `MYSQL_PASSWORD`
-   `MYSQL_DATABASE`
