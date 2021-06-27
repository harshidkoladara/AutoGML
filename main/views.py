from django.shortcuts import render, redirect, get_object_or_404
from random import randint
import smtplib
from .models import User, Models


def index(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        return render(request, 'index.html', {'user': user, 'log': True})
    return render(request, 'index.html', {'auth': True})


def login(request):
    if request.session.get('email') and request.session.get('id'):
        return redirect('index')
    if request.POST:
        try:
            user = User.objects.get(email=request.POST['email'])
            if user.password == request.POST['password']:
                request.session['email'] = user.email
                request.session['id'] = user.id
                return redirect('index')
            else:
                return render(request, 'signin.html', {'login': True, 'error': 'Invalid Credentials!'})
        except:
            return render(request, 'signin.html', {'login': True, 'error': 'Invalid Credentials!'})
    return render(request, 'signin.html', {'login': True})


def smtpService(otp, email):
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login('hackspatel3624@gmail.com', '118@hacks@3624')
        subject = 'AutoGML Confirmation Code'
        body = f'Hello Dev,\nYour otp is {otp}.\nWelcome to AutoGML.'
        msg = f'Subject : {subject} \n\n{body}'
        smtp.sendmail('hackspatel3624@gmail.com', email, msg)


def signup(request):
    if request.session.get('email') and request.session.get('id'):
        return redirect('index')
    if request.method == 'POST':
        try:
            global credentials
            credentials = {
                'fname': request.POST['fname'],
                'lname': request.POST['lname'],
                'email': request.POST['email'],
                'contact': request.POST['contact'],
                'password': request.POST['password']
            }
            userObj = User.objects.all()
            if len([x for x in userObj if x.email == request.POST['email']]) > 0:
                return render(request, 'signup.html', {'signup': True, 'error': 'Email already Existed! Try Another.'})
            global otp
            otp = randint(111111, 999999)
            smtpService(otp, request.POST['email'])
        except:
            if int(otp) == int(request.POST['otp']):
                user = User(fname=credentials['fname'], lname=credentials['lname'], email=credentials['email'],
                            contact=credentials['contact'], password=credentials['password'])
                user.save()
                request.session['email'] = user.email
                request.session['id'] = user.id
                return redirect('index')
            else:
                request.session['confirm'] = 'confirm'
                return render(request, 'confirmation.html', {'confirm': True, 'error': 'Enter Correct Confirmation Code!'})
        request.session['confirm'] = 'confirm'
        return render(request, 'confirmation.html', {'confirm': True})
    return render(request, 'signup.html', {'signup': True})


def confirmation(request):
    if request.session.get('confirm'):
        if request.method == 'POST':
            print(request.POST['otp'], request.POST['info'])
        return render(request, 'confirmation.html', {'confirm': True})
    return render(request, 'signin.html', {'login': True, })


def forgotpwd(request):
    if request.session.get('email') and request.session.get('id'):
        return redirect('index')
    if request.method == 'POST':
        try:
            userObj = User.objects.all()
            if len([x for x in userObj if x.email == request.POST['email']]) == 0:
                return render(request, 'forgotpwd.html', {'forgotpwd': True, 'error': "Email Doesn't Exist!"})
            global user
            user = User.objects.get(email=request.POST['email'])
            global otp
            otp = randint(111111, 999999)
            smtpService(otp, request.POST['email'])
        except:
            try:
                if int(otp) == int(request.POST['otp']):
                    return render(request, 'resetpswd.html', {'resetpswd': True})
                else:
                    request.session['confirm'] = "confirm"
                    return render(request, 'confirmation.html', {'confirm': True, 'error': 'Enter Correct Confirmation Code!'})
            except:
                user.password = request.POST['password']
                user.save()
                return redirect('login')
        request.session['confirm'] = "confirm"
        return render(request, 'confirmation.html', {'confirm': True, 'forgotpwd': True})
    return render(request, 'forgotpwd.html', {'forgotpwd': True})


def allProject(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        table = Models.objects.filter(user=user, is_tables=True)
        img_clf = Models.objects.filter(user=user, is_classification=True)
        face_recog = Models.objects.filter(user=user, is_face=True)
        obj_detect = Models.objects.filter(user=user, is_object=True)
        return render(request, 'all_projects.html', {'all': True, 'user': user, 'table': table, 'img_clf': img_clf, 'face_recog': face_recog, 'obj_detect': obj_detect, 'dash': True})
    return redirect('login')


def logout(request):
    try:
        del request.session['email']
        del request.session['id']
        del request.session['model']
    except:
        pass
    return redirect('index')
