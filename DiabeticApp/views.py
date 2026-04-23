import logging
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.shortcuts import render
from django.http import JsonResponse

from .forms import LoginForm
from .forms import SignUpForm
from .ml_utils import build_metrics_plot
from .ml_utils import ensure_dataset_loaded
from .ml_utils import load_labels
from .ml_utils import predict_uploaded_image
from .ml_utils import save_uploaded_image
from .models import UserProfile

logger = logging.getLogger(__name__)

def health_check(request):
    return JsonResponse({'status': 'ok'})


def _render_auth_page(request, template_name, form, title, subtitle, form_mode):
    return render(
        request,
        template_name,
        {
            "form": form,
            "title": title,
            "subtitle": subtitle,
            "form_mode": form_mode,
        },
    )


def request_user_display_name(request=None):
    if request is not None and request.user.is_authenticated:
        return request.user.first_name or request.user.username
    return "User"


def authenticate_with_email(email, password):
    """Authenticate user by email and password.
    
    Supports login using email address instead of username. Looks up user
    by email and attempts authentication with provided password.
    
    Args:
        email (str): User's email address
        password (str): User's password (plaintext)
        
    Returns:
        User: Authenticated user object if credentials are valid, else None
    """
    try:
        user = User.objects.get(email=email.strip().lower())
    except User.DoesNotExist:
        return None
    return authenticate(username=user.username, password=password)


def index(request):
    return render(request, "index.html", {})


def Register(request):
    if request.user.is_authenticated:
        return redirect("Dashboard")
    form = SignUpForm()
    return _render_auth_page(
        request,
        "Register.html",
        form,
        "Create your screening account",
        "Register with your email and profile details. No OTP is required.",
        "signup",
    )


def RegisterAction(request):
    if request.method != "POST":
        return redirect("Register")

    form = SignUpForm(request.POST)
    if not form.is_valid():
        return _render_auth_page(
            request,
            "Register.html",
            form,
            "Create your screening account",
            "Register with your email and profile details. No OTP is required.",
            "signup",
        )

    user = User.objects.create_user(
        username=form.cleaned_data["username"],
        email=form.cleaned_data["email"],
        password=form.cleaned_data["password"],
    )
    UserProfile.objects.create(
        user=user,
        contact_no=form.cleaned_data["contact_no"],
        address=form.cleaned_data["address"],
    )
    
    logger.info(f"New user created: {user.username}")

    login_form = LoginForm(initial={"email": form.cleaned_data["email"]})
    return _render_auth_page(
        request,
        "UserLogin.html",
        login_form,
        "Account created",
        "Please login with the account you just created to continue to retinal disease screening.",
        "login",
    )


def UserLogin(request):
    if request.user.is_authenticated:
        return redirect("Dashboard")
    form = LoginForm()
    return _render_auth_page(
        request,
        "UserLogin.html",
        form,
        "Login to continue",
        "Use your registered email and password to access the screening dashboard.",
        "login",
    )


def UserLoginAction(request):
    if request.method != "POST":
        return redirect("UserLogin")

    form = LoginForm(request.POST)
    if not form.is_valid():
        return _render_auth_page(
            request,
            "UserLogin.html",
            form,
            "Login to continue",
            "Use your registered email and password to access the screening dashboard.",
            "login",
        )

    user = authenticate_with_email(form.cleaned_data["email"], form.cleaned_data["password"])
    if user is None:
        logger.warning(f"Failed login attempt for email: {form.cleaned_data['email']}")
        form.add_error(None, "Invalid email or password.")
        return _render_auth_page(
            request,
            "UserLogin.html",
            form,
            "Login to continue",
            "Use your registered email and password to access the screening dashboard.",
            "login",
        )

    login(request, user)
    logger.info(f"User logged in: {user.username}")
    return redirect("Dashboard")


@login_required
def Dashboard(request):
    return render(
        request,
        "UserScreen.html",
        {
            "dashboard_mode": True,
            "welcome_title": f"Welcome, {request_user_display_name(request)}",
            "welcome_text": "Select a step below to review the dataset, run the model comparison, or upload a retina image for prediction.",
        },
    )


@login_required
def LogoutUser(request):
    logout(request)
    return redirect("index")


@login_required
def Predict(request):
    return render(
        request,
        "Predict.html",
        {
            "dashboard_mode": True,
            "welcome_title": "Prediction workflow",
            "welcome_text": "Upload a retina image to classify the likely eye disease category.",
        },
    )


@login_required
def PredictAction(request):
    if request.method != "POST":
        return redirect("Predict")

    logger.info(f"Prediction requested by user: {request.user.username}")
    upload = request.FILES.get("t1")
    if upload is None:
        return render(
            request,
            "Predict.html",
            {
                "data": "Please choose a retina image before running prediction.",
                "dashboard_mode": True,
                "welcome_title": "Prediction workflow",
                "welcome_text": "Upload a retina image to classify the likely eye disease category.",
            },
        )

    try:
        file_path = save_uploaded_image(upload)
    except (IOError, ValueError) as exc:
        logger.error(f"Prediction upload failed for user {request.user.username}: {exc}")
        return render(
            request,
            "Predict.html",
            {
                "data": str(exc),
                "dashboard_mode": True,
                "welcome_title": "Prediction workflow",
                "welcome_text": "Upload a retina image to classify the likely eye disease category.",
            },
        )

    result, image_b64, error_message = predict_uploaded_image(file_path)
    if error_message:
        logger.error(f"Prediction failed for user {request.user.username}: {error_message}")
        return render(
            request,
            "Predict.html",
            {
                "data": error_message,
                "dashboard_mode": True,
                "welcome_title": "Prediction workflow",
                "welcome_text": "Upload a retina image to classify the likely eye disease category.",
            },
        )
    
    logger.info(f"Prediction successful for user {request.user.username}: {result['prediction_text']}")
    context = {
        "data": result["prediction_text"],
        "img": image_b64,
        "result": result,
        "prediction_mode": True,
        "welcome_title": f"Welcome, {request_user_display_name(request)}",
        "welcome_text": "Review the uploaded retina image and the model prediction details below.",
    }
    return render(request, "UserScreen.html", context)


@login_required
def ProcessData(request):
    dataset = ensure_dataset_loaded()
    output = "<div class='stats-grid'>"
    output += "<div class='stat-box'><span>Total images</span><strong>" + str(dataset["X"].shape[0]) + "</strong></div>"
    output += "<div class='stat-box'><span>Features per image</span><strong>" + str(dataset["X"].shape[1] * dataset["X"].shape[2] * dataset["X"].shape[3]) + "</strong></div>"
    output += "<div class='stat-box'><span>Training samples</span><strong>" + str(dataset["X_train"].shape[0]) + "</strong></div>"
    output += "<div class='stat-box'><span>Testing samples</span><strong>" + str(dataset["X_test"].shape[0]) + "</strong></div>"
    output += "</div>"
    return render(
        request,
        "UserScreen.html",
        {
            "dashboard_mode": True,
            "welcome_title": "Processed dataset summary",
            "welcome_text": "The resized training data and split information are summarized below.",
            "data": output,
        },
    )


@login_required
def LoadDatasetAction(request):
    labels = load_labels()
    chips = "".join([f"<span class='label-chip'>{label}</span>" for label in labels])
    output = "<div class='label-chip-row'>" + chips + "</div>"
    return render(
        request,
        "UserScreen.html",
        {
            "dashboard_mode": True,
            "welcome_title": "Dataset loaded",
            "welcome_text": "These disease categories are currently available in the retinal dataset.",
            "data": output,
        },
    )


@login_required
def RunML(request):
    metrics, image_b64 = build_metrics_plot()
    output = '<table border="1" align="center" width="100%" class="metrics-table"><tr>'
    columns = ["Algorithm Name", "Accuracy", "Precision", "Recall", "F-Score"]
    for column in columns:
        output += '<th><font size="3" color="black">' + column + "</th>"
    output += "</tr>"

    if not metrics["model_names"]:
        output += '<tr><td colspan="5"><font size="3" color="black">No saved model metrics are available yet. Train and save a model to populate this table.</td></tr>'

    for i, model_name in enumerate(metrics["model_names"]):
        output += '<tr><td><font size="3" color="black">' + model_name + "</td>"
        output += '<td><font size="3" color="black">' + str(metrics["accuracy"][i]) + "</td>"
        output += '<td><font size="3" color="black">' + str(metrics["precision"][i]) + "</td>"
        output += '<td><font size="3" color="black">' + str(metrics["recall"][i]) + "</td>"
        output += '<td><font size="3" color="black">' + str(metrics["fscore"][i]) + "</td></tr>"
    output += "</table>"
    return render(
        request,
        "UserScreen.html",
        {
            "dashboard_mode": True,
            "welcome_title": "Model performance",
            "welcome_text": "Compare the current saved model metrics and the confusion matrix summary.",
            "data": output,
            "img": image_b64,
        },
    )
