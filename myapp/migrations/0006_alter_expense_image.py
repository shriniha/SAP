# Generated by Django 4.1.13 on 2024-08-07 03:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0005_alter_expense_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='expense',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='expense_images/'),
        ),
    ]
