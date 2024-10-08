# Generated by Django 4.1.13 on 2024-08-07 03:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0006_alter_expense_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='expense',
            name='status',
            field=models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')], default='pending', max_length=10),
        ),
    ]
