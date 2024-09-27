# myapp/models.py
# myapp/models.py
from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    USER_ROLES = (
        ('approver', 'Approver'),
        ('staff', 'Staff'),
        ('substitute','Substitute'),
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=USER_ROLES)
    is_present = models.BooleanField(default=True)

    def __str__(self):
        return self.user.username

class Expense(models.Model):
    CATEGORY_CHOICES = [
        ('Travel', 'Travel'),
        ('Meals', 'Meals'),
        ('Training', 'Training'),
        ('Wifi', 'Wifi'),
        ('Office Supplies', 'Office Supplies'),
        ('Maintenance', 'Maintenance'),
        ('Software', 'Software'),
        ('Miscellaneous', 'Miscellaneous'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    description = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to='expense_images/', blank=True, null=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return f"{self.category} expense on {self.date} by {self.user.username}"


class Approver(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)  # Temporarily allow null values
    is_present = models.BooleanField(default=True)
    def __str__(self):
        return self.name

class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class ApproverCategory(models.Model):
    approver = models.ForeignKey(Approver, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.approver.name} - {self.category.name}"

class Substitute(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)  # Temporarily allow null values

    def __str__(self):
        return self.name

class ApproverSubstitute(models.Model):
    approver = models.ForeignKey(Approver, on_delete=models.CASCADE)
    substitute = models.ForeignKey(Substitute, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.approver.name} - {self.substitute.name}"