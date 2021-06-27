from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, HttpResponse
from django.http import StreamingHttpResponse
from main.models import User, Models
from random import randint, choice
from datetime import datetime
import pandas as pd
import numpy as np
import os
import io
from AutoGMLTest.settings import MEDIA_URL
import pickle
import zipfile
import shutil
import yaml
import json
from .start_training import start_training
from .start_detection import start_detection
# GENERAL METHODS


def get_path():
    directory = os.path.abspath(os.getcwd())
    directory = directory.replace('\\', '/')
    directory = directory + '/project/'
    return directory


def caller_od(request, id):
    if request.session.get('email') and request.session.get('id'):
        request.session['model'] = id
        return redirect('import_obj')
    return redirect('create_od')


# INDEX PAGE
def index_obj(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        try:
            model = Models.objects.filter(user=user, is_object=True)
            return render(request, 'index_obj.html', {'index': True, 'user': user, 'models': model})
        except:
            return render(request, 'index_obj.html', {'index': True, 'user': user})
    return redirect('login')
# CREATE PROJECT


def generateID(request):
    if request.session.get('email') and request.session.get('id'):
        lst = [[randint(0, 9)], [chr(randint(65, 90))], [
            chr(randint(97, 122))], [choice(['@', '$'])]]
        psw = list()
        for x in range(14):
            psw.append(choice(lst)[0])
        fnl_psw = ''.join(map(str, psw))
        return fnl_psw
    return redirect('login')


def create_obj(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        if request.method == 'POST':
            model = Models(is_object=True, user=user, name=request.POST['name'], project_id=generateID(
                request)+str(user.id), time=datetime.now())
            model.save()
            request.session['model'] = model.project_id
            return redirect('import_obj')
        return render(request, 'create_obj.html', {'create': True, 'user': user})
    return redirect('login')


# IMPORT DATASET IMAGES
def unzip_data_file(file, path):
    fs = FileSystemStorage(location=path)
    fs.save('{}'.format(file[0].name), file[0])
    with zipfile.ZipFile(f'{path}/{file[0].name}', 'r') as zip_ref:
        zip_ref.extractall(path)
    dir_list = os.listdir(path)
    allowed_dirs = ['runs', 'train', 'test', 'valid', 'data', 'data.yaml']
    [os.remove(f'{path}/{dir}') for dir in dir_list if dir not in allowed_dirs]


def import_obj(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        data = {
            'user': user,
            'model': model,
            'uploaded': False
        }
        if request.POST:
            directory = get_path()
            try:
                path = directory + '/' + request.session['model'] + '/data/'
                if os.path.exists(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(e)

            try:
                path = os.path.join(directory, request.session['model'])
                os.mkdir(path)
                path = os.path.join(path, 'data')
                os.mkdir(path)
            except:
                path = directory + '/' + request.session['model'] + '/data/'

            try:
                file = request.FILES.getlist('zip-file')
                unzip_data_file(file, path)
                data['uploaded'] = True
                return render(request, 'import_obj.html', data)
            except:
                data['uploaded'] = True
                return render(request, 'import_obj.html', data)

        try:
            if request.session['incorrect']:
                data['incorrect'] = True
                del request.session['incorrect']
                return render(request, 'import_obj.html', data)
        except:
            pass

        return render(request, 'import_obj.html', data)
    return redirect('login')

# TRAIN MODEL AND SSUMMERY PAGE


def get_training_data(path):
    data_yaml = yaml.load(open(f'{path}/data.yaml', 'r'))
    yaml_file = data_yaml
    yaml_file['train'] = f'{path}/train/images'
    yaml_file['val'] = f'{path}/valid/images'
    data_yaml['train_instances'] = len(os.listdir(f'{path}train/images'))
    data_yaml['test_instances'] = len(os.listdir(f'{path}test/images'))
    data_yaml['valid_instances'] = len(os.listdir(f'{path}valid/images'))

    with open(f'{path}/data.yaml', "w") as f:
        yaml.dump(yaml_file, f)
    return data_yaml


def train_obj(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'

        if not (os.path.exists(directory+'test') and os.path.exists(directory+'train') and os.path.exists(directory+'valid') and os.path.exists(directory+'data.yaml')):
            request.session['incorrect'] = True
            return redirect('import_obj')

        training_data = get_training_data(directory)
        data = {
            'user': user,
            'model': model,
            'training': training_data['train_instances'],
            'test': training_data['test_instances'],
            'valid': training_data['valid_instances'],
            'nc': training_data['nc'],
            'labels': training_data['names']
        }
        return render(request, 'train_obj.html', data)
    return redirect('login')


def training_ongoing(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            print("request")
            directory = get_path() + request.session['model'] + '/data/'

            name = request.GET['name']

            start_training(f'{directory}/data.yaml', 100, 8, 416,
                           f'{directory}/runs/train', name, f'{directory}/runs/train/{name}')

            return HttpResponse()
    return redirect('login')


# TESTING ON ONJECT DETECTION

def get_model_path(directory):
    directory = directory + 'runs/train/'
    model_name = os.listdir(directory)[-1]
    return directory + model_name


def upload_test_dirs(path, files, fs):
    for file in files:
        fs.save(f'{path}/testData/{file.name}', file)


def get_precesion_data(directory):
    directory = get_model_path(directory) + '/results.txt'
    with open(directory, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        result_list = [x for x in last_line.split("  ") if x.strip()]
        return result_list[8:12]


def zipdir(path, images):
    fs = FileSystemStorage()
    download_list = []
    for img in images:
        fileUrl = fs.url(f'{path}/{img}')
        size = fs.size(f'{path}/{img}')
        download_list.append([img, fileUrl, size])

    return download_list


def detect(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            directory = get_path() + request.session['model'] + '/data/'
            data_directory = get_model_path(directory)
            model_path = data_directory + '/weights/best.pt'
            source_path = directory + '/testData/'
            try:
                testResult_path = os.path.join(directory, 'testResult')
                os.mkdir(testResult_path)
            except:
                pass
            project_name = Models.objects.get(
                project_id=request.session.get('model')).name

            try:
                start_detection(model_path, source_path, 416,
                                0.25, 'cpu', testResult_path, project_name)
            except:
                return redirect('test_obj')
            result_path = os.listdir(f'{directory}/testResult')[-1]

            result_images = [x for x in os.listdir(
                f'{directory}/testResult/{result_path}') if x[-4:] != '.zip']

            download_list = zipdir(
                f'{directory}/testResult/{result_path}', result_images)

            img_path = MEDIA_URL + \
                request.session['model'] + '/data/testResult/' + result_path
            result_images.append(img_path)

            result_images.append(download_list)
        return HttpResponse(json.dumps(result_images))
    return redirect('login')


def test_obj(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'

        model_path = get_model_path(directory) + '/weights/best.pt'
        if not(os.path.exists(model_path)):
            return redirect('train_obj')

        fs = FileSystemStorage()
        fileUrl = fs.url(model_path)
        size = fs.size(model_path)

        if request.POST:
            files = request.FILES.getlist('local-dirs')
            upload_test_dirs(directory, files, fs)

        result_list = get_precesion_data(directory)

        data = {
            'user': user,
            'model': model,
            'result': result_list,
            'fileUrl': fileUrl,
            'fileName': 'Object-Detection-best-weights.pt',
            'size': size,
        }
        return render(request, 'test_obj.html', data)
    return redirect('login')
