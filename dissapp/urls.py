from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("hepatitis-b", views.gepatit_haqida, name="about-gepatit"),
    path("sirroz", views.sirroz_haqida, name="about-sirroz"),
    path("hepatitis-b-davolash", views.gepatit_davolash, name="davolash-gepatit"),
    path("loyiha-haqida", views.loyiha_haqida, name="loyiha-haqida"),

    # path("le", views.load_excel, name="le"),
    path("login", views.login, name="login"),
    path("logout", views.logout, name="logout"),
    path("datasets/<int:id>", views.datasets, name="datasets"),
    path("datasets", views.datasets, name="datasets"),
    path("datasets/changetype/<int:id>/<int:type_n>", views.changetype, name="changetype"),
    path("datasets/changeclasscolumn/<int:id>", views.change_class_column, name="change_class_column"),
    path("select_dataset", views.select_dataset_ajax, name="select_dataset_ajax"),
    path("stability", views.stability, name="stability"),
    path("classification", views.classification, name="classification"),
    path("allowability", views.allowability, name="allowability"),
    path("new-obj", views.new_object, name="new_object"),
    path("bemorlar", views.bemorlar, name="bemorlar_list"),
    path("bemorlar/<int:id>", views.bemor, name="bemor_info"),
    path("bemorlar/<int:id>/pdf", views.bemor_pdf, name="bemor_info_pdf"),
    path("new_objects_result_file", views.new_objects_result_file, name="new_objects_result_file"),
    path("statistika", views.stat, name="statistika"),
    path("bemor-tashhis/<int:id>", views.bemor_set_class, name="bemor-tashhis"),
    
]
