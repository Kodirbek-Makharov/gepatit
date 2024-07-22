from django.db import models
from django.conf import settings

# Create your models here.
class Datasets(models.Model):
    ftypes = [
        ("xls", "Excel fayl"),
        ("csv", "CSV fayl"),
        ("txt", "txt fayl"),
    ]
    name = models.CharField(max_length=250, blank=True, null=True)
    file = models.FileField(upload_to="uploads/%Y/%m/", verbose_name="Tanlanma fayli")
    ftype = models.CharField(max_length=50, choices=ftypes, verbose_name="Tanlanma fayli turi")
    header = models.BooleanField(verbose_name="Tanlanma faylida atributlar sarlavhalari mavjud")
    class_column = models.CharField(max_length=200, blank=True, null=True, verbose_name="Sinf alomat maydoni")
    class_column_values = models.TextField(blank=True, null=True)
    contains_nans = models.BooleanField(blank=True, null=True)
    types = models.TextField(blank=True, null=True)

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        verbose_name = "Tanlanma"
        verbose_name_plural = "Tanlanmalar"


class Allowability_intervals(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    feature1 = models.TextField()
    feature2 = models.TextField()
    median1 = models.FloatField()
    median2 = models.FloatField()
    left = models.FloatField()
    right = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class Domination_intervals(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    feature1 = models.TextField()
    interval_count = models.IntegerField()
    intervals = models.TextField()
    membership_values = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class Stability(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    stabilities = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class Feature_ranking(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    rankings = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)


class Classification(models.Model):
    dataset = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    type = models.CharField(max_length=255)
    k = models.IntegerField(null=True, blank=True)
    nbayes = models.FloatField()
    knn = models.FloatField()
    svm = models.FloatField()
    random_forest = models.FloatField()
    dtree = models.FloatField()
    lda = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True, blank=True)

class Bemorlar(models.Model):
    fio = models.CharField(max_length=255)
    yosh = models.IntegerField()
    pol = models.IntegerField()
    vdnk = models.BigIntegerField()
    qHBsAg = models.BigIntegerField()
    rnk = models.BigIntegerField()
    gemoglabin = models.FloatField()
    eritrotsit = models.FloatField()
    rang = models.FloatField()
    trombosit = models.FloatField()
    leykosit = models.FloatField()
    sya = models.FloatField()
    monotsit = models.FloatField()
    limfotsit = models.FloatField()
    echt = models.FloatField()
    alt = models.FloatField()
    ast = models.FloatField()
    bilurbin = models.FloatField()
    kreatinin = models.FloatField()
    mochevina = models.FloatField()
    uzi = models.IntegerField() # nominal
    belok = models.FloatField()
    fibroskan = models.FloatField()
    albumin = models.FloatField()
    pvt = models.IntegerField() 
    shf = models.FloatField()
    klass = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    turi = models.IntegerField(default=None, null=True, blank=True) # 1 = train 


