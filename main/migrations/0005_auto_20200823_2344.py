# Generated by Django 3.0.4 on 2020-08-23 18:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_models_trained'),
    ]

    operations = [
        migrations.AlterField(
            model_name='models',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
