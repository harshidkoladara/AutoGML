from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, HttpResponse
from django.http import StreamingHttpResponse
from main.models import User, Models
from random import randint, choice
from datetime import datetime
from csv import writer
import pandas as pd
import numpy as np
import os
import io
from AutoGMLTest.settings import MEDIA_URL
import cv2
import face_recognition
import pickle
from .camera import IPWebCam, VideoCamera
# GENERAL METHODS


def get_path():
    directory = os.path.abspath(os.getcwd())
    directory = directory.replace('\\', '/')
    directory = directory + '/project/'
    return directory


def caller_fr(request, id):
    if request.session.get('email') and request.session.get('id'):
        request.session['model'] = id
        return redirect('image_face')
    return redirect('create_face')


def index_fr(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        try:
            model = Models.objects.filter(user=user, is_face=True)
            return render(request, 'index_fr.html', {'index': True, 'user': user, 'models': model})
        except:
            return render(request, 'index_fr.html', {'index': True, 'user': user})
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


def create_face(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        if request.method == 'POST':
            model = Models(is_face=True, user=user, name=request.POST['name'], project_id=generateID(
                request)+str(user.id), time=datetime.now())
            model.save()
            request.session['model'] = model.project_id
            return redirect('import_face')
        return render(request, 'create_face.html', {'create': True, 'user': user})
    return redirect('login')


# ADD/ IMPORT IMAGES
def generate_class_csv(path, exist=False):
    data_images = [x[:x.find('.')] for x in os.listdir(
        path) if (x[x.find('.'):] != '.csv' and x[-4:] != '.sav')]
    if exist:
        df = pd.read_csv(f'{path}/class.csv', index_col=0)
        new_class = [x for x in data_images if x not in list(df.classes)]

        df1 = pd.DataFrame(new_class, columns=['classes'])
        df = df.append(df1, ignore_index=True)
        df.to_csv('{}/class.csv'.format(path))
    else:
        data_images = [x[:x.find('.')] for x in os.listdir(
            path) if (x[x.find('.'):] != '.csv' and x[-4:] != '.sav')]
        df = pd.DataFrame(data_images, columns=['classes'])
        df.to_csv('{}/class.csv'.format(path))


def generate_data_csv(path, exist=False):
    data_images = [x for x in os.listdir(
        path) if (x[x.find('.'):] != '.csv' and x[-4:] != '.sav')]
    img_dict = {img: img[:img.find('.')] for img in data_images}
    if exist:
        df = pd.read_csv(f'{path}/data.csv', index_col=0)
        new_images = [x for x in data_images if x not in list(df.img_name)]
        new_img_dict = {img: img[:img.find('.')] for img in new_images}
        df1 = pd.DataFrame(new_img_dict.items(), columns=[
            'img_name', 'classes'])
        df = df.append(df1, ignore_index=True)
        df.to_csv('{}/data.csv'.format(path))

    else:
        df = pd.DataFrame(img_dict.items(), columns=['img_name', 'classes'])
        df.to_csv('{}/data.csv'.format(path))


def import_face(request):
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
                path = os.path.join(directory, request.session['model'])
                os.mkdir(path)
                path = os.path.join(path, 'data')
                os.mkdir(path)
            except:
                path = directory + '/' + request.session['model'] + '/data/'

            files = files = request.FILES.getlist('images')

            fs = FileSystemStorage(location=path)
            for x in files:
                fs.save('{}'.format(x.name), x)

            if os.path.exists(path+'data.csv'):
                generate_data_csv(path, True)
            else:
                generate_data_csv(path)

            if os.path.exists(path+'class.csv'):
                generate_class_csv(path, True)
            else:
                generate_class_csv(path)
            data['uploaded'] = True
            return render(request, 'import_face.html', data)
        return render(request, 'import_face.html', data)
    return redirect('login')


# IMAGES PAGE FOR MODIFICATION AND VISUALIZATION
def remove_image(path, image_list):
    df = pd.read_csv(f'{path}/data.csv', index_col=0)
    id_list = [df.index[df['img_name'] == img].tolist()[0]
               for img in image_list]
    df = df.drop(id_list, axis=0)
    df = df.reset_index(drop=True)
    df.to_csv('{}/data.csv'.format(path))


def remove_label(path, label):
    df_data = pd.read_csv(f'{path}/data.csv', index_col=0)
    df_class = pd.read_csv(f'{path}/class.csv', index_col=0)
    if label in list(df_class.classes) and label not in list(df_data.classes):
        id = df_class.index[df_class['classes'] == label].tolist()[0]
        df_class = df_class.drop([id], axis=0)
        df_class.to_csv('{}/class.csv'.format(path))


def change_class(path, selected_img, change_cls):
    df = pd.read_csv(f'{path}/data.csv', index_col=0)
    for img in selected_img:
        df.iloc[df.index[df['img_name'] == img].tolist()[0],
                1] = change_cls
    df.to_csv('{}/data.csv'.format(path))


def add_new_class(path, new_class):
    df = pd.read_csv(f'{path}/class.csv', index_col=0)
    if new_class not in list(df.classes):
        df1 = pd.DataFrame([new_class], columns=['classes'])
        df = df.append(df1, ignore_index=True)
        df.to_csv('{}/class.csv'.format(path))


def image_face(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'
        media_dir = MEDIA_URL + request.session['model'] + '/data/'

        if request.method == 'POST':
            if request.POST.get('delete'):
                try:
                    del_img = request.POST.getlist('selected_img')
                    for img in del_img:
                        os.remove(directory+img)
                    remove_image(directory, del_img)
                except Exception as e:
                    print(e)
                    return redirect("import_face")

            if request.POST.get('verify'):
                try:
                    selected_img = request.POST.getlist('selected_img')
                    change_cls = request.POST['class_verify']
                    change_class(directory, selected_img, change_cls)
                except Exception as e:
                    print(e)
                    return redirect("import_face")

            if request.POST.get('add-class'):
                try:
                    new_class = request.POST['add_class']
                    add_new_class(directory, new_class)
                except Exception as e:
                    print(e)
                    return redirect("import_face")

            if request.POST.get('delete-label'):
                try:
                    delete_cls = request.POST['class_delete']
                    remove_label(directory, delete_cls)
                except Exception as e:
                    print(e)
                    return redirect("import_face")

        try:
            class_df = pd.read_csv(f'{directory}/class.csv', index_col=0)
            data_df = pd.read_csv(f'{directory}/data.csv', index_col=0)
        except:
            return redirect("import_face")

        data_images = []
        for name, class_name in zip(data_df.img_name, data_df.classes):
            data_images.append([name, class_name])
        class_names = list(class_df.classes)
        data = {
            'user': user,
            'model': model,
            'images': data_images,
            'dir': media_dir,
            'image_names': class_names,
        }
        return render(request, 'image_face.html', data)
    return redirect('login')


# TRAIN MODEL AND SSUMMERY PAGE

def train_face(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'
        class_df = pd.read_csv(f'{directory}/class.csv', index_col=0)
        data_df = pd.read_csv(f'{directory}/data.csv', index_col=0)
        data = {
            'user': user,
            'model': model,
            'data': [len(list(data_df.img_name)), len(list(class_df.classes))]
        }
        return render(request, 'train_face.html', data)
    return redirect('login')


def findEncoding(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def training_ongoing(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            directory = get_path() + request.session['model'] + '/data/'
            df = pd.read_csv(f'{directory}/data.csv', index_col=0)
            image_list, image_class = list(df.img_name), list(df.classes)

            new_images = []
            for cl in image_list:
                curImg = cv2.imread(f'{directory}/{cl}')
                new_images.append(curImg)
            encodeListKnown = findEncoding(new_images)

            filename = f'{directory}/finalized_model.sav'
            pickle.dump([encodeListKnown, image_class], open(filename, 'wb'))
        return HttpResponse()
    return redirect('login')


# TESTING ON FACIAL RECOGNITION  MODEL
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Method for laptop camera


def video_feed(request):
    directory = get_path() + request.session['model'] + '/data/'
    return StreamingHttpResponse(gen(VideoCamera(directory)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# Method for phone camera


def webcam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def test_face(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'
        if not (os.path.exists(f'{directory}/finalized_model.sav')):
            return redirect('train_face')

        fs = FileSystemStorage()
        fileUrl = fs.url(f'{directory}/finalized_model.sav')
        size = fs.size(f'{directory}/finalized_model.sav')

        data = {
            'user': user,
            'model': model,
            'fileUrl': fileUrl,
            'fileName': 'finalized_model.sav',
            'size': size,
        }
        return render(request, 'test_face.html', data)
    return redirect('login')
