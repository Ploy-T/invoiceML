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

from PyPDF2 import PdfFileWriter, PdfFileReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import extractText2

# filename = '38c5a2ad-508a-443d-b314-e30c9996d24a.pdf'
# data = extract(filename)
# newData = pd.Series(data)

def run(filename):
    data = extractText2.extract(filename)
    newData = pd.Series(data)
    numPage = len(data) +1

    ## D2V_LR
    from gensim.models.doc2vec import Doc2Vec

    model= Doc2Vec.load("d2v_ver3.model")
    D2V_LR_model = pickle.load(open('D2V_LR_model_ver3.sav','rb'))
    #to find the vector of a document which is not in training data
    P = []
    for i in range(len(data)):
        test_data = word_tokenize(data[i])
        v1 = model.infer_vector(test_data)
        vec = v1.reshape(1,-1)
        ypred = D2V_LR_model.predict(vec).tolist()
        P.append(ypred)
    y_pred = [val for sublist in P for val in sublist]
    print('D2V_LR prediction: ', y_pred)

    return y_pred, numPage

def checkIfDuplicates_3(listOfElems):
    ''' Check if given list contains any duplicates '''    
    for elem in listOfElems:
        if listOfElems.count(elem) > 1:
            return True
    return False

def separatePDF(filename):
    y_pred, numPage = run(filename)
    page  = list(range(numPage))

    T = []
    for i,x in enumerate(y_pred):
        if x == 0:
            temp = page[i:i+2]
            T.append([temp])

    TT = [val for sublist in T for val in sublist]
    tt = [val for sublist in TT for val in sublist]

    if checkIfDuplicates_3(tt) == True:
        TT = [[*set(tt)]]

    TTT = [[val] for val in page if val not in tt]

    separate = TTT + TT

    infile = PdfFileReader(filename, 'rb')
    fName = []
    for j in range(len(separate)):
        print(separate[j])
        output = PdfFileWriter()
        for k in separate[j]:
            p = infile.getPage(k)
            output.addPage(p)
        
        fname = filename.split('.pdf')[0]+'_'+str(j)+'.pdf'
        fName.append(fname)
        with open(fname, 'wb') as f:
            output.write(f) 
    return fName


