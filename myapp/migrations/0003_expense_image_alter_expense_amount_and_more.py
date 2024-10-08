# Generated by Django 4.2.15 on 2024-08-07 01:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_expense'),
    ]

    operations = [
        migrations.AddField(
            model_name='expense',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='expenses/'),
        ),
        migrations.AlterField(
            model_name='expense',
            name='amount',
            field=models.DecimalField(decimal_places=2, max_digits=10),
        ),
        migrations.AlterField(
            model_name='expense',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
    ]
