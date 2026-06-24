---
title: DeepDiabetic
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DeepDiabetic

DeepDiabetic is a Django-based web application for identifying diabetic and retinal eye disease patterns from fundus images using trained deep learning model weights. The project combines a protected web dashboard, sample retina-image inference, saved model metrics, and a Docker deployment configured for Hugging Face Spaces.

Live website:

https://adi080122-deepdiabetic.hf.space

Default demo login:

```text
Email: adityaws10507@gmail.com
Password: Aditya@8122
```

## Recruiter Demo Flow

The live Hugging Face deployment is designed so recruiters can test the model without downloading the full training dataset.

1. Open the live website.
2. Login with the default credentials above.
3. Go to `Predict Disease`.
4. Use either:
   - `Upload retina image` to test your own fundus image.
   - `Use sample` under the bundled sample images to run prediction immediately.
5. Review the prediction label, confidence score, model variant, input size, uploaded image preview, and prediction overlay.

The full raw training dataset is intentionally not bundled in the free deployment because it is large. The deployed demo includes trained model weights, saved metrics, and sample retina images so the inference workflow is still testable.

## Project Features

- Email/password authentication with a default demo account.
- Protected dashboard for screening workflow navigation.
- Dataset label summary.
- Processed-data demo summary for public deployment.
- Saved model performance table and confusion matrix.
- Retina image upload for disease prediction.
- Bundled sample retina images for quick recruiter testing.
- Clinical recommendation text based on predicted disease class.
- Docker deployment on Hugging Face Spaces.

## Supported Disease Classes

The application is configured for four retinal disease categories:

- Cataract
- DME
- DR
- Glaucoma

## Technology Stack

Frontend:

- HTML templates using Django Template Language.
- CSS in `DiabeticApp/static/style.css`.
- Responsive clinical dashboard layout.
- Static sample images served from `DiabeticApp/static/`.

Backend:

- Python
- Django
- Django authentication system
- SQLite by default
- Gunicorn for production serving
- WhiteNoise for static files

Machine Learning:

- TensorFlow / Keras
- NumPy
- OpenCV headless
- scikit-learn
- Matplotlib
- Seaborn
- HDF5 model weights

Deployment:

- Docker
- Hugging Face Spaces
- CPU Basic free hardware
- Port `7860`

## System Architecture

```text
User / Recruiter
      |
      v
Django Templates + CSS
      |
      v
Django Views and URL Routes
      |
      +--> Authentication
      |       - Login
      |       - Logout
      |       - Default seeded demo user
      |
      +--> Dashboard Pages
      |       - Dataset labels
      |       - Processed-data summary
      |       - Saved model metrics
      |
      +--> Prediction Workflow
              - Upload image or select bundled sample
              - Save/read image
              - Preprocess with OpenCV
              - Load cached TensorFlow/Keras model
              - Run inference
              - Render result and visualization
```

## Repository Structure

```text
.
|-- Diabetic/                         # Django project settings and root URLs
|-- DiabeticApp/                      # Main Django application
|   |-- management/commands/          # Startup command for default demo user
|   |-- middleware/                   # Error handling middleware
|   |-- migrations/                   # Database migrations
|   |-- static/                       # CSS and sample/demo images
|   |-- templates/                    # Django HTML templates
|   |-- app_config.py                 # Paths and ML configuration
|   |-- forms.py                      # Login and signup forms
|   |-- ml_utils.py                   # Dataset, metrics, and prediction helpers
|   |-- models.py                     # User profile model
|   |-- urls.py                       # App routes
|   `-- views.py                      # Page and prediction logic
|-- model/                            # Tracked model and metric artifacts
|   |-- efficient_weights.hdf5
|   |-- metric.npy
|   `-- cm.npy
|-- Dockerfile                        # Hugging Face Docker deployment
|-- manage.py                         # Django CLI entry point
|-- requirements.txt                  # Python dependencies
`-- testtrain.py                      # Training / experiment script
```

## Included Model Artifacts

The public repository tracks the deployment-ready artifacts:

```text
model/efficient_weights.hdf5
model/metric.npy
model/cm.npy
```

Some larger local training artifacts may exist on the developer machine but are ignored by Git, for example:

```text
model/data_224.npy
model/X_224.npy
model/resnet_weights.hdf5
model/vgg_weights.hdf5
```

Those files are not required for recruiters to test the live prediction demo.

## Clone and Run Locally

Use Python 3.10 for best compatibility with `tensorflow==2.13.0`.

1. Clone the repository:

```bash
git clone https://github.com/Aditya10507/DeepDiabetic.git
cd DeepDiabetic
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Apply database migrations:

```bash
python manage.py migrate
```

5. Create or update the default demo user:

```bash
python manage.py ensure_default_user
```

6. Start the local server:

```bash
python manage.py runserver
```

7. Open the app:

```text
http://127.0.0.1:8000
```

Login with:

```text
Email: adityaws10507@gmail.com
Password: Aditya@8122
```

## Local Environment Variables

The app works locally without extra environment variables. These optional variables can be used to customize behavior:

```text
SECRET_KEY                 Django secret key for production
DEBUG                      true or false
ALLOWED_HOSTS              Comma-separated allowed hosts
APP_DATA_DIR               Directory for SQLite database and runtime files
DEFAULT_LOGIN_EMAIL        Default seeded user email
DEFAULT_LOGIN_PASSWORD     Default seeded user password
DEFAULT_LOGIN_USERNAME     Default seeded username
DATABASE_URL               PostgreSQL connection URL
DB_ENGINE                  Use postgres/postgresql for PostgreSQL mode
```

## Deployment on Hugging Face Spaces

The app is deployed with Docker on Hugging Face Spaces.

Current live Space:

```text
https://adi080122-deepdiabetic.hf.space
```

Docker behavior:

1. Uses `python:3.10-slim`.
2. Installs system runtime dependency `libgomp1`.
3. Installs Python packages from `requirements.txt`.
4. Copies the repository into `/app`.
5. Runs `collectstatic`.
6. On startup, runs:

```bash
python manage.py migrate --noinput
python manage.py ensure_default_user
gunicorn Diabetic.wsgi:application --bind 0.0.0.0:${PORT} --workers 1 --timeout 180
```

Hugging Face settings:

```text
SDK: Docker
Port: 7860
Hardware: CPU Basic
```

## Deploy Your Own Hugging Face Space

1. Login to Hugging Face CLI:

```bash
hf auth login
```

2. Create a Docker Space:

```bash
hf repos create YOUR_USERNAME/deepdiabetic --type space --space-sdk docker --flavor cpu-basic --public --exist-ok
```

3. Upload the project:

```bash
hf upload YOUR_USERNAME/deepdiabetic . . --repo-type space --commit-message "Deploy DeepDiabetic"
```

4. Wait for build completion:

```bash
hf spaces wait YOUR_USERNAME/deepdiabetic
```

5. Check logs:

```bash
hf spaces logs YOUR_USERNAME/deepdiabetic
```

## Important Notes

- This is a demonstration system and should not be used as a clinical diagnosis tool.
- The prediction output should be confirmed by a qualified eye specialist.
- The free Hugging Face deployment uses SQLite in temporary runtime storage, so user-created accounts may reset after rebuilds or restarts.
- The default demo account is recreated automatically at startup.
- Full training data is not included in the public deployment; sample retina images are included for testing inference.

## Useful Routes

```text
/                         Landing page
/login/                   Login page
/dashboard/               Protected dashboard
/LoadDatasetAction        Dataset label summary
/ProcessData              Demo processed-data summary
/RunML                    Saved metrics and confusion matrix
/Predict                  Upload/sample prediction page
/health/                  Health check endpoint
```

## Validation Performed

- Django system check passes.
- Default user creation works.
- Hugging Face Space builds and runs.
- Live login works with the default credentials.
- Live sample prediction works from the `Predict Disease` page.
