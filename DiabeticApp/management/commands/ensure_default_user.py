import os

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from DiabeticApp.models import UserProfile


class Command(BaseCommand):
    help = "Create or update the default demo login account."

    def handle(self, *args, **options):
        email = os.environ.get("DEFAULT_LOGIN_EMAIL", "adityaws10507@gmail.com").strip().lower()
        password = os.environ.get("DEFAULT_LOGIN_PASSWORD", "Aditya@8122")
        username = os.environ.get("DEFAULT_LOGIN_USERNAME", "aditya10507").strip()

        user = User.objects.filter(email=email).first()
        if user is None:
            user = User.objects.create_user(username=username, email=email, password=password)
            created = True
        else:
            created = False
            user.username = username
            user.email = email
            user.set_password(password)
            user.save(update_fields=["username", "email", "password"])

        UserProfile.objects.update_or_create(
            user=user,
            defaults={
                "contact_no": os.environ.get("DEFAULT_LOGIN_CONTACT", "0000000000"),
                "address": os.environ.get("DEFAULT_LOGIN_ADDRESS", "Default demo account"),
            },
        )

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} default login user: {email}"))
