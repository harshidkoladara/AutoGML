from django.shortcuts import render, redirect, HttpResponse
from main.models import User, Models
from random import randint, choice
from datetime import datetime
import pandas as pd
import requests
import io
import os
from . import data_cleaning
from . import ml_model
from django.core.files.storage import FileSystemStorage
import numpy as np
import pickle
import json
# GENERIC METHODs


def get_path():
    directory = os.path.abspath(os.getcwd())
    directory = directory.replace('\\', '/')
    directory = directory + '/project/'
    return directory


def generateID():
    lst = [[randint(0, 9)], [chr(randint(65, 90))], [
        chr(randint(97, 122))], [choice(['@', '$'])]]
    psw = list()
    for x in range(14):
        psw.append(choice(lst)[0])
    fnl_psw = ''.join(map(str, psw))
    return fnl_psw


def caller(request, id):
    if request.session.get('email') and request.session.get('id'):
        request.session['model'] = id
        return redirect('imported_table')
    return redirect('index_table')


# INDEX PAGE
def index(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        try:
            model = Models.objects.filter(user=user, is_tables=True)
            return render(request, 'index_table.html', {'index': True, 'user': user, 'models': model})
        except:
            return render(request, 'index_table.html', {'index': True, 'user': user})
    return redirect('login')


# CRETE PROJECT
def createProject(request):
    if request.session.get('email') and request.session.get('id'):
        user = User.objects.get(email=request.session['email'])
        if request.method == 'POST':
            model = Models(is_tables=True, user=user, name=request.POST['name'], project_id=generateID(
            )+str(user.id), time=datetime.now())
            model.save()
            request.session['model'] = model.project_id

            directory = get_path()

            try:
                path = os.path.join(directory, request.session['model'])
                os.mkdir(path)
                path = os.path.join(path+'/', 'data')
                os.mkdir(path)
            except:
                try:
                    path = os.path.join(path+'/', 'data')
                    os.mkdir(path)
                except:
                    path = directory+'/'+request.session['model']+'/data/'

            return redirect('import_table')
        return render(request, 'create_table.html', {'create': True, 'user': user})
    return redirect('login')


# IMPORT PAGE


def import_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        path = get_path() + request.session['model'] + '/data/'

        dataset = list(
            [x for x in os.listdir(path) if x != 'data.csv' and x[-3:] != '.h5'])

        if request.method == 'POST':
            if request.POST.get('local-import'):
                myfile = request.FILES['local']

                fs = FileSystemStorage(location=path)

                if len(dataset) > 0:
                    try:
                        df = pd.read_csv(path+dataset[0])
                    except:
                        df = pd.read_csv(path+dataset[0], encoding='latin-1')
                    df2 = pd.read_csv(myfile)
                    if (len(df2.columns) == len(df.columns)) and (False not in list((df2.columns == df.columns))):
                        fs.save(f'raw{len(dataset)}-{myfile.name}', myfile)
                    else:
                        return render(request, 'import_table.html', {'table': True, 'user': user, 'model': model, 'error': 'Dataset must be same!'})
                else:
                    fs.save(f'raw{len(dataset)}-{myfile.name}', myfile)
                return redirect('imported_table')

            if request.POST.get('github-import'):
                url = request.POST['github']
                download = requests.get(url).content
                df = pd.read_csv(io.StringIO(download.decode('utf-8')))
                a = url[::-1]
                i = a.find('/')
                if len(dataset) > 0:
                    try:
                        first_df = pd.read_csv(path+dataset[0])
                    except:
                        first_df = pd.read_csv(
                            path+dataset[0], encoding='latin-1')
                    if (len(df.columns) == len(first_df.columns)) and (False not in list((df.columns == first_df.columns))):
                        df.to_csv(r'{}/raw{}-{}'.format(path,
                                                        len(dataset), url[-i:]))
                    else:
                        return render(request, 'import_table.html', {'table': True, 'user': user, 'model': model, 'error': 'Dataset must be same!'})
                df.to_csv(r'{}/raw{}-{}'.format(path,
                                                len(dataset), url[-i:]), index=False)
                return redirect('imported_table')

        return render(request, 'import_table.html', {'table': True, 'user': user, 'model': model})
    return redirect('index_table')


# VISUALIZE TABLES


def visualize_csv_table(request, id):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'

        df = pd.read_csv(directory+id)
        columns = df.columns.tolist()
        values = df.values.tolist()
        return render(request, 'csv_table.html', {'table': True, 'user': user, 'model': model, 'columns': columns, 'values': values})
    return redirect('index_table')


def imported_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + request.session['model'] + '/data/'

        if request.method == 'POST':
            os.remove(directory + request.POST['delete'])

        files = [x for x in os.listdir(
            directory) if x != 'data.csv' and x[-3:] != '.h5']
        if len(files) == 0:
            try:
                os.remove(directory+'data.csv')
            except:
                pass
            return redirect('import_table')
        return render(request, 'imported_table.html', {'table': True, 'user': user, 'model': model, 'files': files})
    return redirect('index_table')


# TABLE SCHEMA
def process_csv(directory, file):
    df1 = pd.read_csv(directory+file)
    process_dict = dict()
    for col, dtype in zip(df1.columns, df1.dtypes):
        process_dict[col] = dtype

    df = pd.DataFrame.from_dict(process_dict, orient='index')
    df.to_csv('{}/data.csv'.format(directory))
    df = pd.read_csv(directory+'data.csv')
    df.columns = ['Columns', 'Dtype']
    df['Target'] = False
    df['Nullable'] = False
    df.to_csv('{}/data.csv'.format(directory), index=False)


def schema_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'

        files = [x for x in os.listdir(
            directory) if x != 'data.csv' and x[-3:] != '.h5']

        if not os.path.exists(directory+'/data.csv'):
            process_csv(directory, files[0])

        df = pd.read_csv(directory+files[0])
        data = pd.read_csv(directory+'/data.csv')

        if request.POST:
            for x in df.columns:
                try:
                    data.at[data.index[data['Columns'] == x]
                            [0], 'Target'] = False
                    data.at[data.index[data['Columns'] == x][0],
                            'Dtype'] = str(request.POST['drop'+str(x)])
                    data.at[data.index[data['Columns'] == x][0],
                            'Nullable'] = str(request.POST['check'+str(x)])
                except Exception as e:
                    data.at[data.index[data['Columns'] == x][0],
                            'Nullable'] = False
            else:
                try:
                    data.at[data.index[data['Columns'] ==
                                       request.POST['target']][0], 'Target'] = True
                except:
                    pass
            data.to_csv('{}/data.csv'.format(directory), index=False)

        process = dict()
        for col, dtype, nullability in zip(data['Columns'], data['Dtype'], data['Nullable']):
            process[col] = [str(dtype), nullability]

        try:
            selected_target = data[data['Target'] == True]['Columns'].all()
        except Exception as e:
            print('Exeption:', e)

        return render(request, 'schema_table.html', {'table': True, 'user': user, 'model': model, 'data': process, 'target': selected_target})
    return redirect('index_table')


# ANALYZE

def change_dtype(data, df):
    for col, dtype in zip(data['Columns'], data['Dtype']):
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            print(e)
    return df


def analyze_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'

        data = pd.read_csv(directory+'/data.csv')

        na = ['-', 'na', 'nan', 'NaN', 'NA']
        csv_files = [pd.read_csv(directory+csv_file, na_values=na, index_col=None)
                     for csv_file in os.listdir(directory) if csv_file != 'data.csv' and csv_file[-3:] != '.h5']
        df = pd.concat(csv_files, ignore_index=True)
        df = change_dtype(data, df)
        # print(df.dtypes)

        missing = dict()
        for x in df.columns:
            try:
                m = round(df[x].mean(), 2)
                s = round(df[x].std(), 2)
            except:
                m = 'NA'
                s = 'NA'
            missing[x] = [str(round((df[x].isnull().sum()*100) /
                                    df.shape[0], 2))+str(" %"), len(df[x].unique()), m, s]
        supervised = (data.Target.any() == True)
        target_column = data[data['Target'] == True]['Columns'].values
        process = dict()
        for col, dtype, nullability in zip(data['Columns'], data['Dtype'], data['Nullable']):
            try:
                process[col] = [str(dtype), nullability, missing[col][0],
                                missing[col][1], missing[col][2], missing[col][3]]
            except Exception as e:
                print(e)
            try:
                if supervised:
                    process[col].append(
                        round(df[col].corr(df[target_column[0]]), 2))
                else:
                    process[col].append('NA')
            except:
                process[col].append('NA')
        return render(request, 'analyze_table.html', {'table': True, 'user': user, 'model': model, 'analyze': process})
    return redirect('index_table')

# START TRAINING


def training_process_start_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        if request.GET:
            directory = get_path() + \
                request.session['model'] + '/data/'

            dataCleaning = data_cleaning.DataCleaning(directory)

            raw = dataCleaning.combine_and_remove_duplicate()
            raw = dataCleaning.remove_missing_value_column(raw)
            raw = dataCleaning.remove_missing_value_row(raw)
            raw = dataCleaning.transform_data(raw)
            raw = dataCleaning.fill_missing(raw)

            mlModel = ml_model.ML_Model(directory, raw)
            if mlModel.type == "supervised" and mlModel.learning == "classification":
                print("Classification")
                mlModel.classification_model()
            elif mlModel.type == "supervised" and mlModel.learning == "regression":
                print("Regression")
                mlModel.regression_model()
            return HttpResponse()
    return redirect('index_table')


def train_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])

        directory = get_path() + \
            request.session['model'] + '/data/'
        data = pd.read_csv(directory+'/data.csv')

        na = ['-', 'na', 'nan', 'NaN', 'NA']
        csv_files = [pd.read_csv(directory+csv_file, na_values=na, index_col=None)
                     for csv_file in os.listdir(directory) if csv_file[0:3] == 'raw']
        df = pd.concat(csv_files, ignore_index=True)

        df = change_dtype(data, df)

        global target, input_features, input_instances
        target = list(data[data['Target'] == True]['Columns'])
        input_features = df.shape[1] - 1
        input_instances = df.shape[0]

        return render(request, 'train_table.html', {'table': True, 'user': user, 'model': model, 'target': target[0], 'input_f': input_features, 'input_i': input_instances})
    return redirect('index_table')


# TEST PAGE
def test_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'
        fs = FileSystemStorage(location=directory)

        _, model_type, accuracy = pickle.load(
            open(f'{directory}/model.h5', 'rb'))

        fileUrl = fs.url(f'{directory}/model.h5')
        size = fs.size(f'{directory}/model.h5')

        data = pd.read_csv(directory+'/data.csv')
        columns = False
        try:
            if (data.Target.any() == True):
                target_column = data[data['Target'] == True]['Columns'].values
                columns = list(data.Columns)
                columns.remove(target_column)
            nullables = data[data['Nullable'] == True]['Columns'].values
            if len(nullables) > 0:
                [columns.remove(x) for x in nullables]
        except:
            pass

        data = {'table': True,
                'user': user,
                'model': model,
                'accuracy': accuracy,
                'model_type': model_type,
                'fileUrl': fileUrl,
                'fileName': 'custom_model.h5',
                'size': size,
                'columns': columns
                }
        if request.POST:
            files = request.FILES.get('local')
            test_df = pd.read_csv(files)
            print(columns, test_df.columns)
            if (len(columns) == len(test_df.columns)) and (False not in list((columns == test_df.columns))):
                [os.remove(f'{directory}/{x}')
                 for x in os.listdir(directory) if x[:5] == 'test-']
                fs.save(f'test-{files.name}', files)
            else:
                data['error'] = True
                return render(request, 'test_table.html', data)
        return render(request, 'test_table.html', data)
    return redirect('index_table')


def predict_value(directory, data, values):
    columns = list(data.Columns)
    if (data.Target.any() == True):
        target_column = data[data['Target'] == True]['Columns'].values
        columns = list(data.Columns)
        columns.remove(target_column)

    nullables = data[data['Nullable'] == True]['Columns'].values
    if len(nullables) > 0:
        [columns.remove(x) for x in nullables]

    trained_model, _, _ = pickle.load(
        open(f'{directory}/model.h5', 'rb'))
    encoder = pickle.load(open(f'{directory}/encoder.h5', 'rb'))

    new_values = []
    for i in range(len(columns)):
        col_dtype = data[data['Columns'] == columns[i]]['Dtype']

        if list(col_dtype)[0] == 'int64':
            new_values.append([values[i].astype(np.int)])
        elif list(col_dtype)[0] == 'float64':
            new_values.append([values[i].astype(np.float)])
        else:
            new_values.append([values[i]])
        if columns[i] in encoder:
            new_values[i] = encoder.get(
                columns[i]).fit_transform(new_values[i])

    new_values = np.array(new_values)
    result = trained_model.predict(new_values.reshape(1, -1))
    if list(target_column)[0] in encoder:
        le = encoder.get(list(target_column)[0])
        result = le.inverse_transform(result.astype(np.int64).reshape(-1, 1))
    return (f'The predicted result for {list(target_column)[0]} is {result[0]}.')


def single_predict(request):
    if request.GET:
        directory = get_path() + request.session['model'] + '/data/'
        data = pd.read_csv(directory+'/data.csv')
        target_column = data[data['Target'] == True]['Columns'].values
        columns = list(data.Columns)
        columns.remove(target_column)
        values = np.array(json.loads(request.GET['data']))
        result = predict_value(directory, data, values)
        return HttpResponse(result)


def batch_predict(request):
    if request.GET:
        directory = get_path() + request.session['model'] + '/data/'
        test_file = [x for x in os.listdir(directory) if x[:5] == 'test-']
        data = pd.read_csv(f'{directory}/data.csv')
        target_column = data[data['Target'] == True]['Columns'].values
        test_data = pd.read_csv(f'{directory}/{test_file[0]}')
        trained_model, _, _ = pickle.load(
            open(f'{directory}/model.h5', 'rb'))
        encoder = pickle.load(open(f'{directory}/encoder.h5', 'rb'))

        for col in test_data.columns:
            test_data[col] = test_data[col].astype(
                data[data['Columns'] == col]['Dtype'].values[0])

            if col in encoder:
                test_data[col] = encoder.get(col).fit_transform(test_data[col])
        result = trained_model.predict(test_data.values)

        if list(target_column)[0] in encoder:
            le = encoder.get(list(target_column)[0])
            result = le.inverse_transform(
                result.astype(np.int64).reshape(-1, 1))
        test_data[target_column] = result
        print(test_data)
        test_data.to_csv(f'{directory}/testResult.csv')
        return HttpResponse()


def result_table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        directory = get_path() + request.session['model'] + '/data/'

        df = pd.read_csv(f'{directory}/testResult.csv')
        columns = df.columns.tolist()
        values = df.values.tolist()

        fs = FileSystemStorage(location=directory)
        fileUrl = fs.url(f'{directory}/testResult.csv')
        size = fs.size(f'{directory}/testResult.csv')

        data = {
            'table': True,
            'user': user,
            'model': model,
            'columns': columns,
            'values': values,
            'test': True,
            'fileUrl': fileUrl,
            'fileName': 'Test-Result.csv',
            'size': size,
        }
        return render(request, 'csv_table.html', data)

    return redirect('index_table')


def table(request):
    if request.session.get('email') and request.session.get('id') and request.session.get('model'):
        user = User.objects.get(email=request.session['email'])
        model = Models.objects.get(project_id=request.session['model'])
        return render(request, 'model_table.html', {'table': True, 'user': user, 'model': model, 'active_tab': 'tab2'})
    return redirect('index_table')
