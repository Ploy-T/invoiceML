from fileinput import filename
import os
import uuid
import json
from azure.storage.blob import BlobClient,BlobServiceClient
import azure.functions as func
from azure.identity import DefaultAzureCredential
from PyPDF2 import PdfFileWriter, PdfFileReader
import logging
import traceback
import PyPDF2 
import re
from nltk.corpus import stopwords
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageSequence, ImageOps
import pytesseract
import pickle
from sklearn.metrics import classification_report
from gensim.models import Word2Vec,Doc2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
import itertools
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
# from keras.utils import shuffle

import main2

LOCAL_PATH='./uploads'

class AzureBlobFile:
    def __init__(self,accountName,container_name,upload_container_name):
        accountUrl =  "https://"+accountName+".blob.core.windows.net"

        self.accountUrl = accountUrl
        self.accountUrl2 = accountUrl
        self.container_name=container_name
        self.upload_container_name = upload_container_name
        self.blob_service_client = BlobServiceClient(self.accountUrl,DefaultAzureCredential())
        #self.conn_str = conn_str
        self.blob_service_client1 = BlobServiceClient(self.accountUrl2,DefaultAzureCredential())

    # def upload_pdf_file(self,upload_file_path,blob):
    #     logging.info("we are in upload_pdf_file")
    #     print("upload_file_path: ",upload_file_path)
    #     pathToFile = upload_file_path.split('/')[-1].split('.pdf')[0].split('_')[0]
    #     blob_client = self.blob_service_client1.get_blob_client(container=self.upload_container_name.lower()+"/email"+"/"+pathToFile,
    #                                                             blob=blob)
    #     with open(upload_file_path, "rb") as data:
    #         blob_client.upload_blob(data,overwrite=True)

    def upload_pdf_file(self,upload_file_path,pdfPath, blob):
        logging.info("we are in upload_pdf_file")
        print("upload_file_path: ",upload_file_path)
        # pathToFile = upload_file_path.split('/')[-1].split('.pdf')[0]
        pathToFile = upload_file_path.split('/')[-1].split('.pdf')[0].split('_')[0:-1]
        pathToFile = '_'.join(str(j) for j in pathToFile)
        # check how many level
        level = len(pdfPath.split('/'))-1
        if level == 1:
            prefixPath = os.path.join(pdfPath.split('/')[0])
        elif level == 2:
            prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1])
        elif level == 3:
            prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2])
        elif level == 4:
            prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2], pdfPath.split('/')[3])
        elif level == 5:
            prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2], pdfPath.split('/')[3], pdfPath.split('/')[4])
        blob_client = self.blob_service_client1.get_blob_client(container=self.upload_container_name.lower()+"/"+prefixPath+"/"+pathToFile,
                                                                blob=blob)
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data,overwrite=True)

    def getBlobStream(self, file_path):

        file_name = file_path#.split("/")[-1]
        
        logging.info("file_name: "+str(file_name))
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        isExist = blob_client.exists()
        logging.info(isExist)

        if isExist:
            # download blob file
            stream = blob_client.download_blob()
            logging.info("File exists and is readable")
            bytes1 = stream.readall()

            return bytes1
        else:
            return ""
        return ""

def make_folder():
    folder_name = os.path.join("process_files",uuid.uuid4().hex)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return True

def createPath(file_name):
    folder_name = make_folder()
    upload_path = os.path.join(folder_name, "uploads") 
    create_folder(upload_path)  
    return upload_path

def Invoice(modelName,fileName, blob, blob2):
  
    folder_name = make_folder()
    upload_path = os.path.join(folder_name, "uploads")

    create_folder(upload_path)
    #data_file.save(filename)
    upload_model_path = os.path.join(upload_path, modelName)
    print('upload_model_path: ',upload_model_path)
    with open(upload_model_path, "wb") as file:
        file.write(blob)

    upload_pdf_path = os.path.join(upload_path, fileName)
    print('upload_pdf_path: ',upload_pdf_path)
    with open(upload_pdf_path, "wb") as file:
        file.write(blob2)

    outFile = main2.separatePDF(upload_pdf_path)
    return outFile

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass

    if req_body:
        fileName = req_body["fileName"] #pdf
        modelName = req_body["modelName"] #model
        pdfPath = req_body["pdfPath"]
        modelPath = req_body["modelPath"]
        container_name = req_body['containerName']
        upload_container_path = req_body['containerName']
        accountName = req_body['accountName']
        try:
            azure_blob_file = AzureBlobFile(accountName,container_name,upload_container_path)
            # blob to get json file
            blob = azure_blob_file.getBlobStream(modelPath)
            # need a second blob to get pdf file to put in process_files folder           
            blob2 = azure_blob_file.getBlobStream(pdfPath)

            fName = Invoice(modelName,fileName,blob,blob2)

            out = PdfFileWriter()
            for i in range(len(fName)):
                bl = fName[i]
                print('********')
                print('bl: ',bl)
                blob = bl.split('/')[-1]
                print('#########')
                print('blob:',blob)
                with open(blob, 'wb') as f:
                    out.write(f)
                azure_blob_file.upload_pdf_file(bl,pdfPath, blob)
        
            # return func.HttpResponse("", status_code=200)
            print('fName: ', fName)
            if len(fName) != 0:
                fSplit = 'true'
            else:
                fSplit = 'false'
            output = req_body           
            output['fileSplit'] = fSplit
            level = len(pdfPath.split('/'))-1
            if level == 1:
                prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('.pdf')[0].split('/')[-1])
            elif level == 2:
                prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('.pdf')[0].split('/')[-1])
            elif level == 3:
                prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2], pdfPath.split('.pdf')[0].split('/')[-1])
            elif level == 4:
                prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2], pdfPath.split('/')[3], pdfPath.split('.pdf')[0].split('/')[-1])
            elif level == 5:
                prefixPath = os.path.join(pdfPath.split('/')[0], pdfPath.split('/')[1], pdfPath.split('/')[2], pdfPath.split('/')[3], pdfPath.split('/')[4], pdfPath.split('.pdf')[0].split('/')[-1])
            newFile = []
            for i in range(len(fName)):
                p = prefixPath + '/'+fName[i].split('/')[-1]
                print('p: ',p)
                newFile.append(p)
            output['newFilePath'] = newFile

            return json.dumps(output)
            # if fName is not None:
            #     return json.dumps(fName)
            # else:
            #     return "No data extracted"       

        except Exception as ex:
            print(f"Exception Sentence : {str(ex)}")
            logging.info(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            response_msg = {'Error':str(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))}
            response_msg = json.dumps(response_msg)

            return response_msg
    
    return response_msg