from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    contact_no = models.CharField(max_length=20)
    address = models.CharField(max_length=120)

    def __str__(self):
        return self.user.email or self.user.username
