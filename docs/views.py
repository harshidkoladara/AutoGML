from django.shortcuts import render


def index_doc_fr(request):
    return render(request, 'docs_fr.html')


def index_doc_img(request):
    return render(request, 'docs_img.html')


def index_doc_od(request):
    return render(request, 'docs_od.html')


def index_doc_table(request):
    return render(request, 'docs_table.html')
