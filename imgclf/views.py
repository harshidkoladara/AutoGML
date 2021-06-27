from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, HttpResponse
from main.models import User, Models
from random import randint, choice
from datetime import datetime
from csv import writer
import pandas as pd
import numpy as np
import os
import io
import shutil
import zipfile
import pickle
import json
from AutoGMLTest.settings import MEDIA_URL
from .train_img_classifier import train, predict
# GENERIC METHODS


def get_path():
    directory = os.path.abspath(os.getcwd())
    directory = directory.replace('\\', '/')
    directory = directory + '/project/'
    return directory


def caller_clf(request, id):
    if request.session.get('email') and request.session.get('id'):
        request.session['model'] = id
        return redirect('import_clf')
    return redirect('index_table')


# INDEX PAGE
def index_clf(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        try:
            model = Models.objects.filter(user=user, is_classification=True)
            return render(request, 'index_clf.html', {'index': True, 'user': user, 'models': model})
        except:
            return render(request, 'index_clf.html', {'index': True, 'user': user})
    return redirect('login')


def caller_clf(request, id):
    if request.session.get('email') and request.session.get('id'):
        request.session['model'] = id
        return redirect('import_clf')
    return redirect('index_table')


# CREATE PROJECT
def generateID():
    lst = [[randint(0, 9)], [chr(randint(65, 90))], [
        chr(randint(97, 122))], [choice(['@', '$'])]]
    psw = list()
    for x in range(14):
        psw.append(choice(lst)[0])
    fnl_psw = ''.join(map(str, psw))
    return fnl_psw


def createProject_clf(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        if request.method == 'POST':
            model = Models(is_classification=True, user=user, name=request.POST['name'], project_id=generateID(
            )+str(user.id), time=datetime.now())
            model.save()
            request.session['model'] = model.project_id
            return redirect('import_clf')
        return render(request, 'create_clf.html', {'create': True, 'user': user})
    return redirect('login')


# IMPORT DATA IMAGES

def unzip_data(file, fs, path):
    fs.save('{}'.format(file[0].name), file[0])
    with zipfile.ZipFile(f'{path}/{file[0].name}', 'r') as zip_ref:
        zip_ref.extractall(path)

    file_name = file[0].name.split(".")[0]
    dir_list = os.listdir(f'{path}/{file_name}')
    for f in dir_list:
        try:
            shutil.move(f'{path}/{file_name}/{f}', path)
        except:
            pass
    os.remove(f'{path}/{file[0]}')
    shutil.rmtree(f'{path}/{file_name}')


def upload_dirs(path, dir_name, files, fs):
    try:
        path = os.path.join(path, dir_name)
        os.mkdir(path)
    except:
        pass

    for file in files:
        fs.save(f'{path}/{file.name}', file)


def import_clf(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path()
        try:
            path = os.path.join(directory, request.session['model'])
            os.mkdir(path)
            path = os.path.join(path, 'data')
            os.mkdir(path)
        except:
            path = directory + '/' + request.session['model'] + '/data/'

        if request.method == 'POST' and request.FILES:
            fs = FileSystemStorage(location=path)

            if request.POST.get('upload-zip'):
                file = request.FILES.getlist('zip-file')
                unzip_data(file, fs, path)
                return render(request, 'import_clf.html', {'user': user, 'model': model, 'uploded': "True"})

            elif request.POST['upload-dir']:
                files = request.FILES.getlist('local-dirs')
                upload_dirs(path, request.POST['dir-name'], files, fs)
                return render(request, 'import_clf.html', {'user': user, 'model': model, 'uploded': "True"})

        # print(os.listdir(path))
        return render(request, 'import_clf.html', {'user': user, 'model': model})
    return redirect('index_clf')

# PROCESS DATA IMAGES


def get_12_images(directory, img_path, i, j):
    image_dirs = [dir for dir in os.listdir(
        directory) if (dir[-3:] != '.h5' and dir != 'testFiles' and dir[-4:] != '.csv')]
    image_list = []
    for dir in image_dirs:
        for img in os.listdir(f'{directory}/{dir}'):
            image_list.append([f'{img_path}{dir}/{img}', dir])
    if i >= 0 and j <= len(image_list):
        return image_list[i: j]
    else:
        return image_list[0: 20]


def get_discription(directory):
    image_dirs = [dir for dir in os.listdir(
        directory) if (dir[-3:] != '.h5' and dir != 'testFiles' and dir[-4:] != '.csv')]
    discription = [
        [x, len(os.listdir(f'{directory}/{x}'))] for x in image_dirs if (x[-3:] != '.h5' and x != 'testFiles' and x[-4:] != '.csv')]
    return discription


def del_image(del_img):
    for img in del_img:
        dd = os.path.abspath(os.getcwd())
        dd = dd.replace('\\', '/')
        os.remove(f'{dd}/{img}')


def image_clf(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'
        img_path = MEDIA_URL + request.session['model'] + '/data/'

        if request.method == 'POST':
            if request.POST.get('next-images'):
                index = int(request.POST.get('next-images'))
                image_list = get_12_images(
                    directory, img_path, index + 20, index + 40)
                discription = get_discription(directory)
                return render(request, 'image_clf.html', {'user': user, 'model': model, 'images': image_list, 'dir': img_path, 'discription': discription, 'img_index': index+20})

            elif request.POST.get('last-images'):
                index = int(request.POST.get('last-images'))
                image_list = get_12_images(
                    directory, img_path, index - 20, index)
                discription = get_discription(directory)
                return render(request, 'image_clf.html', {'user': user, 'model': model, 'images': image_list, 'dir': img_path, 'discription': discription, 'img_index': index-20})

            elif request.POST.get('delete'):
                try:
                    del_img = request.POST.getlist('selected_img')
                    del_image(del_img)
                except Exception as e:
                    print(e)
        try:
            image_list = get_12_images(directory, img_path, 0, 20)
            discription = get_discription(directory)
        except Exception as e:
            print(e)
            return redirect('import_clf')

        return render(request, 'image_clf.html', {'user': user, 'model': model, 'images': image_list, 'dir': img_path, 'discription': discription, 'img_index': 0})
    return redirect('login')


# TRAIN MODEL
def get_input_data(directory):
    discription = get_discription(directory)
    instances = 0
    for x in discription:
        instances = instances + x[1]
    return [len(discription), instances]


def training_ongoing(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            directory = get_path() + request.session['model'] + '/data/'
            train(request.GET['epochs'], directory,
                  get_input_data(directory)[0])
        return HttpResponse()
    return redirect('login')


def train_clf(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'

        data = {
            'user': user,
            'model': model,
            'data': get_input_data(directory)
        }
        return render(request, 'train_clf.html', data)
    return redirect('login')


# TESTING PAGE
def upload_test_dirs(path, files, fs):
    for file in files:
        fs.save(f'{path}/{file.name}', file)


def detect(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            directory = get_path() + request.session['model'] + '/data/'
            test_directory = get_path() + request.session['model']
            result = predict(directory, test_directory)

            df = pd.DataFrame(result, columns=['Image', 'class'])
            df.to_csv(f'{directory}/result.csv')

            fs = FileSystemStorage()
            fileUrl = fs.url(f'{directory}/result.csv')
            size = fs.size(f'{directory}/result.csv')
            result.append('result.csv')
            result.append(fileUrl)
            result.append(size)
        return HttpResponse(json.dumps(result))
    return redirect('login')


def test_clf(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'
        img_path = MEDIA_URL + request.session['model'] + '/testFiles'

        test_directory = get_path() + request.session['model']
        try:
            test_directory = os.path.join(test_directory, 'testFiles')
            os.mkdir(test_directory)
        except:
            pass

        if not (os.path.exists(f'{directory}/model_resnet50.h5')):
            return redirect('train_clf')

        fs = FileSystemStorage()
        fileUrl = fs.url(f'{directory}/model_resnet50.h5')
        size = fs.size(f'{directory}/model_resnet50.h5')

        accuracy_data = pickle.load(open(f'{directory}/score.h5', 'rb'))

        if request.POST:
            files = request.FILES.getlist('local-dirs')
            upload_test_dirs(test_directory, files, fs)
        data = {
            'user': user,
            'model': model,
            'fileUrl': fileUrl,
            'fileName': 'model_resnet50.h5',
            'size': size,
            'data': accuracy_data,
            'dir': img_path,
        }
        return render(request, 'test_clf.html', data)
    return redirect('login')


def classifier_clf(request):
    return render(request, 'model_clf.html')
