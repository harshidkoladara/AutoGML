# Generated by Django 3.0.4 on 2020-08-23 18:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_models_time'),
    ]

    operations = [
        migrations.AddField(
            model_name='models',
            name='trained',
            field=models.BooleanField(default=False),
        ),
    ]
