# Generated by Django 3.1.1 on 2024-07-21 19:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dissapp', '0012_bemorlar_limfotsit'),
    ]

    operations = [
        migrations.AddField(
            model_name='bemorlar',
            name='turi',
            field=models.IntegerField(blank=True, default=None, null=True),
        ),
    ]
