import sys
import os
import numpy as np
import logging
import time
import shutil
import json

#################### web related
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
import bottle
from bottle import route, template
from bottle import HTTPError, HTTPResponse
bottle.BaseRequest.MEMFILE_MAX = 5 * 1024 * 1024

from myreader import ImageData


def readlabels(labelfile):
    alllabels = []
    isclassifylabel = True
    for line in open(labelfile, 'rb'):
        cols = line.split('\t')
        alllabels.append(cols)
        if len(cols) != 2:
            isclassifylabel = False

    labeldict = {}
    for cols in alllabels:
        label = int(cols[-1])
        if label not in labeldict:
            labeldict[label] = []
        if isclassifylabel:
            labeldict[label].append(cols[0])
        else:
            labeldict[label].append(cols[:-1])
        labelkeys = sorted(labeldict.keys())
    return isclassifylabel, labeldict, labelkeys


class Feadb(object):
    def __init__(self, datasets):
        self.datasets = {}
        for datasetname, datasetprefix in datasets:
            imagedata = ImageData(datasetprefix + '.data')
            labeldata = readlabels(datasetprefix + '.label')
            self.datasets[datasetname] = (imagedata, labeldata)

    def getimage(self, datsetname, key):
        return self.datasets[datsetname][0].getvalue(key)

    def getsampledatas(self, datsetname, classnum=5, sampleperclass=10):
        if datsetname not in self.datasets:
            return None
        isclassifylabel, labeldict, labelkeys = self.datasets[datsetname][1]

        sampledatas = []

        selectlabelidx = np.random.choice(
            len(labelkeys), min(len(labelkeys), classnum), replace=False)
        labeldata = []
        for labelidx in sorted(selectlabelidx):
            label = labelkeys[labelidx]
            datas = labeldict[label]
            selectsampleidx = np.random.choice(
                len(datas), min(len(datas), sampleperclass), replace=False)
            labeldata = [datas[sampleidx] for sampleidx in selectsampleidx]
            sampledatas.append((label, labeldata))
        return isclassifylabel, sampledatas

    def getdatasets(self):
        return self.datasets


_feadb = None


@route('/thumbs/<datasetname:path>/<key:re:[a-zA-Z0-9_]+>.jpg')
def thumfile(datasetname, key):
    global _feadb
    imagecontent = _feadb.getimage(datasetname, key)
    if imagecontent is None:
        return HTTPError(404, "File not found.")
    headers = {}
    headers['Content-Type'] = 'image/png'
    headers['Content-Length'] = len(imagecontent)
    body = imagecontent
    headers["Accept-Ranges"] = "bytes"
    return HTTPResponse(body, **headers)


@route('/datasets/')
def getdataset():
    global _feadb
    datasets = _feadb.getdatasets()
    htmltemplate = '''
    <html>
        <body>
            %for datasetname in datasets:
                %labelnum = len(datasets[datasetname][1][1])
                <p>
                    <a href="/datasets/{{datasetname}}/"> {{datasetname}} </a>  
                    labelnum:{{labelnum}}
                </p>
                    
            %end
        </body>
    </html>
    '''

    return template(htmltemplate, datasets=datasets)


@route('/datasets/<datasetname:path>/')
def getsample(datasetname):

    global _feadb
    if datasetname not in _feadb.getdatasets():
        return HTTPError(404, "dataset not found.")

    isclassifylabel, sampledatas = _feadb.getsampledatas(datasetname)
    htmltemplate = '''
    <html>
        <body>
            %for label, labeldata in sampledatas:
                <h1>Label {{label}}</h1>
                %if isclassifylabel:
                    %for key in labeldata:
                        <p>{{key}} <img src="/thumbs/{{datasetname}}/{{key}}.jpg"></img></p>
                    %end
                %else:
                    %for row, keys in enumerate(labeldata):
                        <h2> row {{row}}, label {{label}} </h2>
                        %for key in keys:
                            <p>{{key}} <img src="/thumbs/{{datasetname}}/{{key}}.jpg"></img></p>
                        %end
                    %end
                %end
            %end
        </body>
    </html>
    '''

    return template(
        htmltemplate,
        datasetname=datasetname,
        sampledatas=sampledatas,
        isclassifylabel=isclassifylabel)


def main():
    ip = '0.0.0.0'
    port = '8000'
    datasets = [
        ('cifar10train', './dataset/cifar10/cifar10_train'),
        ('facetrain', './dataset/face_ms1m_small/train'),
        ('facetest', './dataset/face_ms1m_small/test'),
    ]
    global _feadb
    _feadb = Feadb(datasets)
    bottle.run(host=ip, port=port, debug=True)


if __name__ == '__main__':
    main()
