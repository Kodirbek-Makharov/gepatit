from django.contrib import admin
from .models import Datasets

# Register your models here.
class DatasetAdmin(admin.ModelAdmin):
    model = Datasets
    list_display = ["name"]

    def save_model(self, request, obj, form, change):
        if obj.user is None:
            obj.user = request.user
        obj.save()


admin.site.register(Datasets, DatasetAdmin)
