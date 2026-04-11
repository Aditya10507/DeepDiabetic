from django import forms
from django.contrib.auth.models import User


class SignUpForm(forms.Form):
    username = forms.CharField(max_length=150)
    email = forms.EmailField(max_length=254)
    password = forms.CharField(widget=forms.PasswordInput)
    contact_no = forms.CharField(max_length=20)
    address = forms.CharField(max_length=120, widget=forms.Textarea(attrs={"rows": 3}))

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("This username is already taken.")
        return username

    def clean_contact_no(self):
        contact_no = self.cleaned_data["contact_no"].strip()
        # Support international formats: remove spaces, hyphens, parentheses, plus sign, and country codes
        cleaned_no = contact_no.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").lstrip("+")
        # Remove country code prefix if present (e.g., 1 for US/Canada, 91 for India)
        if cleaned_no.startswith("1") and len(cleaned_no) == 11:
            cleaned_no = cleaned_no[1:]  # Remove leading 1 for US/Canada
        elif cleaned_no.startswith("91") and len(cleaned_no) == 12:
            cleaned_no = cleaned_no[2:]  # Remove leading 91 for India
        
        if not cleaned_no.isdigit():
            raise forms.ValidationError("Contact number must contain digits only (after removing spaces, hyphens, etc.).")
        if len(cleaned_no) < 9:
            raise forms.ValidationError("Contact number must be at least 9 digits long.")
        if len(cleaned_no) > 15:
            raise forms.ValidationError("Contact number must be at most 15 digits long.")
        return contact_no

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholders = {
            "username": "Choose a username",
            "email": "Enter your email",
            "password": "Create a password",
            "contact_no": "10-digit contact number",
            "address": "Enter your address",
        }
        for name, field in self.fields.items():
            field.widget.attrs.update(
                {
                    "placeholder": placeholders.get(name, ""),
                    "class": "auth-input",
                }
            )


class LoginForm(forms.Form):
    email = forms.EmailField(max_length=254)
    password = forms.CharField(widget=forms.PasswordInput)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["email"].widget.attrs.update({"placeholder": "Enter your email", "class": "auth-input"})
        self.fields["password"].widget.attrs.update({"placeholder": "Enter your password", "class": "auth-input"})
