# Generated by Django 4.0.3 on 2023-02-27 10:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_rel_obj_uuid'),
    ]

    operations = [
        migrations.AddField(
            model_name='rel',
            name='sub_uuid',
            field=models.CharField(default='', max_length=100, verbose_name='结点2ID'),
        ),
        migrations.AlterField(
            model_name='rel',
            name='obj_uuid',
            field=models.CharField(default='', max_length=100, verbose_name='结点1ID'),
        ),
    ]
