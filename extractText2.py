import PyPDF2 
import re
from nltk.corpus import stopwords
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageSequence, ImageOps
import pytesseract

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = text.replace('\n', ' ')
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    # text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    # text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

def convert_pdf2images(inputpath, outputpath):
    images = convert_from_path(inputpath)
    for i in range(len(images)):
        images[i].save(outputpath+'_page_'+ str(i) +'.png', 'JPEG')
    return images

def extractText(filename):
    if '.docx' in filename:
        convert(filename,filename + '.pdf')
        filename = filename + '.pdf'
    images = convert_from_path(filename,grayscale=True) 
    text_all = []
    for i in range(len(images)):
        images[i].save('page_'+ str(i) +'.jpg', 'JPEG')
        with Image.open('page_'+ str(i) +'.jpg') as img:
            for frame in ImageSequence.Iterator(img):
                text = str(((pytesseract.image_to_string(frame, lang='eng',config="--psm 4"))))
                text = text.splitlines()
                text_all.append(text) 
                for idx,i in enumerate(text_all):
                    text_all[idx] = [x for x in text_all[idx] if x != '' and x != ' ']
    return text_all

# trainingData = pd.read_csv('trainingData.csv')
# filename = trainingData['filename']
# file_name = trainingData['filename'].drop_duplicates().reset_index(drop=True)
# page1 = trainingData['page1']
# page2 = trainingData['page2']
# filename = '9dd3d0b7-d1e1-405f-abe5-d03818b7996f.pdf'

def extract(filename):
    keyword = ['bill', 'invoice']
    ## use keyword then words before and after
    data = extractText(filename)
    WORD = []
    I = []
    numPage = len(data)
    for i in range(numPage):
        eachPage = ' '.join(data[i]).split()        
        for idx,word in enumerate(eachPage):
            if word.lower() in keyword:
                I.append(idx) 
        firstWord = I[0]
        # find x words before and after
        if firstWord < 11:
            beforeWord = eachPage[0:I[0]]
        else:
            beforeWord = eachPage[I[0]-11:I[0]]
        afterWord = eachPage[I[0]:I[0]+11]
        allWord = beforeWord + afterWord
        WORD.append(allWord)
    # concatenate consecutive page
    CONCAT = []
    for ii in range(len(WORD)-1):
        p1 = ' '.join(WORD[ii])
        p2 = ' '.join(WORD[ii+1])
        p = p1 + ' ' + p2
        CONCAT.append(p.lower())
    return CONCAT














