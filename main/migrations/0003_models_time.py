# Generated by Django 3.0.4 on 2020-08-23 18:10

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_models'),
    ]

    operations = [
        migrations.AddField(
            model_name='models',
            name='time',
            field=models.DateTimeField(default=datetime.datetime(2020, 8, 23, 18, 10, 17, 805304, tzinfo=utc)),
            preserve_default=False,
        ),
    ]