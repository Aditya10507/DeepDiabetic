from django.test import TestCase
from django.contrib.auth.models import User
from .models import UserProfile
from .forms import SignUpForm, LoginForm


class UserModelTests(TestCase):
    """Test cases for User and UserProfile models."""

    def setUp(self):
        """Create test user and profile."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        self.profile = UserProfile.objects.create(
            user=self.user,
            contact_no="9876543210",
            address="123 Test Street"
        )

    def test_user_profile_creation(self):
        """Test that user profile is created successfully."""
        self.assertEqual(self.profile.user, self.user)
        self.assertEqual(self.profile.contact_no, "9876543210")
        self.assertEqual(self.profile.address, "123 Test Street")

    def test_user_profile_str(self):
        """Test user profile string representation."""
        self.assertEqual(str(self.profile), "test@example.com")

    def test_user_profile_one_to_one(self):
        """Test that user profile has one-to-one relationship with user."""
        # Trying to create another profile for same user should raise error
        with self.assertRaises(Exception):
            UserProfile.objects.create(
                user=self.user,
                contact_no="1234567890",
                address="Another address"
            )


class SignUpFormTests(TestCase):
    """Test cases for SignUp form validation."""

    def test_valid_signup_form(self):
        """Test valid signup form data."""
        form_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepass123",
            "contact_no": "9876543210",
            "address": "123 Main Street"
        }
        form = SignUpForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_invalid_contact_too_short(self):
        """Test that contact number too short is rejected."""
        form_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepass123",
            "contact_no": "12345",
            "address": "123 Main Street"
        }
        form = SignUpForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("contact_no", form.errors)

    def test_invalid_contact_letters(self):
        """Test that contact number with letters is rejected."""
        form_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepass123",
            "contact_no": "abcd1234ef",
            "address": "123 Main Street"
        }
        form = SignUpForm(data=form_data)
        self.assertFalse(form.is_valid())

    def test_international_contact_format(self):
        """Test that international contact formats are accepted."""
        valid_formats = [
            "+1-234-567-8901",  # US with country code
            "91-9876543210",      # India with country code
            "(234) 567-8901",     # US format with parentheses
        ]
        for contact in valid_formats:
            form_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": "pass",
                "contact_no": contact,
                "address": "123 Street"
            }
            form = SignUpForm(data=form_data)
            # Contact format should be valid after parsing
            self.assertNotIn("contact_no", form.errors, f"Failed for contact: {contact}")

    def test_duplicate_email(self):
        """Test that duplicate email is rejected."""
        User.objects.create_user(
            username="existing",
            email="existing@example.com",
            password="pass"
        )
        form_data = {
            "username": "newuser",
            "email": "existing@example.com",
            "password": "pass",
            "contact_no": "9876543210",
            "address": "123 Street"
        }
        form = SignUpForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)

    def test_duplicate_username(self):
        """Test that duplicate username is rejected."""
        User.objects.create_user(
            username="existing",
            email="existing@example.com",
            password="pass"
        )
        form_data = {
            "username": "existing",
            "email": "newuser@example.com",
            "password": "pass",
            "contact_no": "9876543210",
            "address": "123 Street"
        }
        form = SignUpForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("username", form.errors)


class LoginFormTests(TestCase):
    """Test cases for Login form validation."""

    def test_valid_login_form(self):
        """Test valid login form data."""
        form_data = {
            "email": "user@example.com",
            "password": "testpass123"
        }
        form = LoginForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_missing_email(self):
        """Test that missing email is rejected."""
        form_data = {
            "email": "",
            "password": "testpass123"
        }
        form = LoginForm(data=form_data)
        self.assertFalse(form.is_valid())

    def test_missing_password(self):
        """Test that missing password is rejected."""
        form_data = {
            "email": "user@example.com",
            "password": ""
        }
        form = LoginForm(data=form_data)
        self.assertFalse(form.is_valid())

    def test_invalid_email_format(self):
        """Test that invalid email format is rejected."""
        form_data = {
            "email": "notanemail",
            "password": "testpass123"
        }
        form = LoginForm(data=form_data)
        self.assertFalse(form.is_valid())


class ViewSecurityTests(TestCase):
    """Test cases for view security and authentication."""

    def setUp(self):
        """Create test user."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        UserProfile.objects.create(
            user=self.user,
            contact_no="9876543210",
            address="123 Test Street"
        )

    def test_health_check_endpoint(self):
        """Test health check endpoint returns OK status."""
        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_index_page_loads(self):
        """Test that index page loads successfully."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_dashboard_requires_login(self):
        """Test that dashboard requires authentication."""
        response = self.client.get("/dashboard/")
        # Should redirect to login
        self.assertEqual(response.status_code, 302)

    def test_dashboard_accessible_after_login(self):
        """Test that dashboard is accessible after login."""
        self.client.login(username="testuser", password="testpass123")
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)

