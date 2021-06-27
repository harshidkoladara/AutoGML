from django.db import models


class User(models.Model):
    fname = models.CharField(max_length=32, blank=False)
    lname = models.CharField(max_length=32, blank=False)
    email = models.EmailField(max_length=254, unique=True)
    contact = models.CharField(max_length=10, blank=False)
    password = models.CharField(max_length=32, blank=False)

    def __str__(self):
        return self.fname + ' ' + self.lname


class Models(models.Model):
    id = models.AutoField(primary_key=True)
    is_tables = models.BooleanField(default=False)
    is_classification = models.BooleanField(default=False)
    is_object = models.BooleanField(default=False)
    is_face = models.BooleanField(default=False)
    is_api = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=32, blank=False)
    project_id = models.CharField(max_length=72, blank=False)
    time = models.DateTimeField()

    def __str__(self):
        typeModel = str()
        if self.is_tables:
            typeModel = 'Tables'
        elif self.is_classification:
            typeModel = 'Image Classification'
        elif self.is_face:
            typeModel = 'Facial Recognition'
        elif self.is_object:
            typeModel = 'Object Detection'
        return str(self.id) + ' ' + self.name + ' - ' + self.project_id + ' - ' + typeModel
    # make directory with name of project_id+email make subdirectory data, model, csv
    # csv contains all modified data
