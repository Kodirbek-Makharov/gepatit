# Generated by Django 3.1.1 on 2023-07-07 09:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dissapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasets',
            name='name',
            field=models.CharField(blank=True, max_length=250, null=True),
        ),
        migrations.AlterField(
            model_name='datasets',
            name='header',
            field=models.BooleanField(verbose_name='Tanlanma faylida atributlar sarlavhalari mavjud'),
        ),
    ]
