# Generated by Django 4.1.13 on 2024-08-07 08:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0011_approver_is_present_alter_userprofile_role'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='is_present',
            field=models.BooleanField(default=True),
        ),
    ]
