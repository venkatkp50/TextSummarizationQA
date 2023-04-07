
import io
#import os
#import re
import json
import time
import math
#import spacy
#from spacy import displacy
#import zipfile
#import logging
import requests
#import openai
import rouge
#import nltk
import numpy as np
import pandas as pd
#import altair as alt
from PIL import Image
import streamlit as st
#from pprint import pprint
from nltk.corpus import stopwords
#from copy import deepcopy
#from tqdm.notebook import tqdm
#from streamlit_chat import message
import seaborn as sns
import matplotlib.pyplot as plt
import re, os, string, random, requests
#from subprocess import Popen, PIPE, STDOUT
from haystack.nodes import EmbeddingRetriever
from haystack.utils import clean_wiki_text
from haystack.utils import convert_files_to_docs
from haystack.utils import fetch_archive_from_http,print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from summarizer import Summarizer,TransformerSummarizer
from bert_score import score
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate import meteor
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
new_stopwords = ["What"]
BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
import warnings
warnings.filterwarnings('ignore')



# Stopword = stopwords.words('english') 
# Stopword.extend(new_stopwords)
# NER = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


imagename2 = Image.open('images/Sidebar2.jpg')
st.sidebar.image(imagename2)
st.sidebar.title('Settings')
modelSelected = st.sidebar.selectbox('Choose Reader Model',options=('deepset/roberta-base-squad2-covid','deepset/roberta-base-squad2','deepset/covid_bert_base'))
imagename = Image.open('images/caronavirus banner.jpg')
st.image(imagename)
st.text_input("Your Query", key="input_text",value='')


user_message = st.session_state.input_text


col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]
#data = pd.DataFrame(cleaned_files, columns=col_names)
data = pd.read_csv('json2csv.csv')

print('created json2csv ')

text_file_path = 'text_file'
abstract_file_path = 'abstract_file'
bert_file_summary_path = 'summary_file/BERT'
gpt_file_summary_path = 'summary_file/GPT'

doc_dir = text_file_path

document_store = InMemoryDocumentStore(use_bm25=True)
docs = convert_files_to_docs(dir_path=doc_dir,clean_func=clean_wiki_text,split_paragraphs=True)
document_store.write_documents(docs)
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path=modelSelected, use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)



print('completed reader ')


if user_message != '':
    print('inside user_meassage block')
    results = pipe.run(query=user_message,params={"Retriever": {"top_k": 10},"Reader": {"top_k": 5}})
    ans = []
    doc = []
    score = []
    context = []
    id =[]
    for result in results['answers']:
        ans.append(result.answer)
        score.append(result.score)
        context.append(result.context)
        id.append(result.meta['name'])
 
    print('.....10')
    responsedf = pd.DataFrame({'Probable Anwsers':ans,'Score':score,'Context':context,'Source File Name':id})
    ans = responsedf['Probable Anwsers'].values.tolist()
    ids = responsedf['Source File Name'].values.tolist()
    scorelist = responsedf['Score'].values.tolist()
    scorelist = [ x*100 for x in scorelist]

    responsedf = responsedf.astype(str).apply(lambda x: x.str[:30])
    ansfig = responsedf['Probable Anwsers'].values.tolist()
    
    max_score = float(responsedf['Score'].max())
    if max_score >  0.9:
        scoremultiplier = 90        
    elif max_score > 0.7:
            scoremultiplier = 150
    elif max_score > 0.4:
            scoremultiplier = 175
    else:
            scoremultiplier = 200

    score100 = [scr*scoremultiplier for scr in score]
    
    #colorcode = ['rgb(116, 191, 0)', 'rgb(60, 194, 0)', 'rgb(2, 198, 0)', 'rgb(0, 210, 186)', 'rgb(0, 174, 213)']
    colorcode = ['rgb(102, 0, 51)', 'rgb(204, 0, 102)', 'rgb(255, 51, 153)', 'rgb(102, 255, 255)', 'rgb(204, 204, 255)']
    opacitycode = [0.8, 0.6, 0.5, 0.4,0.3]
    fig = go.Figure(data=[go.Scatter(x=ansfig, y=scorelist,marker=dict(color=colorcode,opacity=opacitycode,size=score100,))])
    st.subheader('Responses..')
    st.markdown('----')
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

    
    col1.write(ans[0])
    col2.write(ans[1])
    col3.write(ans[2])
    col4.write(ans[3])
    col5.write(ans[4])
    st.markdown('----')
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    col1.write(str(round(score[0],2)*100)+'%')
    col2.write(str(round(score[1],2)*100)+'%')
    col3.write(str(round(score[2],2)*100)+'%')
    col4.write(str(round(score[3],2)*100)+'%')
    col5.write(str(round(score[4],2)*100)+'%')
    st.markdown('----')
    st.subheader('Score %')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True,)
    filecount = 0

    def getTextSummarization(filecount,summarizationFor,std_text,max_abstract_token_size,max_sent_size):
        if summarizationFor == 'std':
            #st.write('Calling from inside',data[data['paper_id'] == id[filecount].replace('.txt','')]['abstract'].values[0],np.nan)
            if data[data['paper_id'] == id[filecount].replace('.txt','')]['abstract'].values[0] is np.nan:
                #st.write('setting text to blank')
                return_text = ''
            else:
                return_text = data[data['paper_id'] == id[filecount].replace('.txt','')]['abstract'].values[0]
        elif summarizationFor == 'BERT':
            print('BERT inside .................1')
            header =[]
            berttext = []
            para = []
            #bert_model = Summarizer() 
            print('tot_words_ref =',max_abstract_token_size,'BERT_MAX_TOKEN=',BERT_MAX_TOKEN)
            if max_abstract_token_size > BERT_MAX_TOKEN:
                print('BERT inside .................2.1')
                for line in std_text:
                    if len(line) > 1:
                        if len(line) < 100:
                            header.append(line)
                        else:
                            para.append(line)
                for parabody in para:
                    berttext.append(bert_model(body=parabody,max_length=100))
                    berttext = Summarizer(body=parabody,max_length=max_abstract_token_size,num_sentences=max_sent_size)
                    return_text = ''.join( lines for lines in berttext) 
            else:
                print('BERT inside .................2.2')
                for line in std_text:
                    para.append(line) 
                berttext = ''.join( lines for lines in para) 
                berttext = bert_model(body=berttext,max_length=max_abstract_token_size,num_sentences=max_sent_size)
                return_text = ''.join( lines for lines in berttext)               
        elif summarizationFor == 'GPT2':            

            header =[]
            para = []
            gpt2text = []
            
            print('tot_words_ref =',max_abstract_token_size,'BERT_MAX_TOKEN=',BERT_MAX_TOKEN)
            if max_abstract_token_size > GPT2_MAX_TOKEN:
                for line in std_text:
                    if len(line) > 1:
                        if len(line) < 100:
                            header.append(line)
                        else:
                            para.append(line)                  
                for parabody in para:
                    gpt2text.append(GPT2_model(body=parabody, max_length=100))
                
                gpt2text_full = ''.join(text for text in gpt2text)
                return_text = GPT2_model(body=gpt2text_full, max_length=max_abstract_token_size,num_sentences=max_sent_size)
            else:
                for line in std_text:
                    para.append(line) 

                gpt2text = ''.join( lines for lines in para) 
                gpt2text = GPT2_model(body=gpt2text,max_length=max_abstract_token_size,num_sentences=max_sent_size)
                return_text = ''.join( lines for lines in gpt2text) 
        return return_text


    tab1, tab2 = st.tabs(["Single Document Summarization", "Multi Document Summarization"])

#     mystyle = '''
#     <style>
#         p {
#             text-align: justify;
#         }
#     </style>
#     '''
    with tab1:
        print('inside tab1 .................')
        col1 , col2 , col3 = st.columns([1,1,1])
        col1.error('Reference Standard')
        col2.error('BERT Summarization')
        col3.error('GPT-2 Summarization')
        
        print('created summ header .................')

        gold_text = getTextSummarization(filecount,'std','',0,0) 
        
        print('getTextSummarization for std .................')
        while (gold_text == '' and filecount < 5):
            filecount = filecount+1
            gold_text = getTextSummarization(filecount,'std','',0,0)
        # print('STD filecount=',filecount)
        # print(len(word_tokenize(gold_text)))
        col1.write(gold_text, align_text='justify')  
        

        #gold_token = len(word_tokenize(gold_text))
        tot_words_ref = len(word_tokenize(gold_text))
        max_abstract_token_size  = math.ceil(tot_words_ref / 100) * 100
        max_sent_size = math.ceil(len(sent_tokenize(gold_text))/10)*10
        full_text = data[data['paper_id'] == id[filecount].replace('.txt','')]['text'].values[0]
        st.write('full_text ................len=.',len(full_text) , 'tot_words_ref = ',tot_words_ref ,'max_sent_size=',max_sent_size,'max_abstract_token_size=',max_abstract_token_size,'BERT_MAX_TOKEN=',BERT_MAX_TOKEN)
        #bert_summary = getTextSummarization(filecount,'BERT',full_text,tot_words_ref,max_sent_size)  
        header =[]
        berttext = []
        para = []
        if max_abstract_token_size > BERT_MAX_TOKEN:    
            st.write('if .........max_abstract_token_size > BERT_MAX_TOKEN ')
            for line in full_text:
                    if len(line) > 1:
                        if len(line) < 100:
                            header.append(line)
                        else:
                            para.append(line)            
            st.write('complted para creation .......para len =',len(para))
            for parabody in para:
                
                berttext.append(bert_model(body=parabody,max_length=100))
                st.write('tot_words_ref=',tot_words_ref)
                st.write('max_sent_size=',max_sent_size)
                bert_model = Summarizer()
                berttext = bert_model(parabody,max_length=max_abstract_token_size,num_sentences=max_sent_size)
                #return_text = ''.join( lines for lines in berttext)
                bert_summary = ''.join( lines for lines in berttext)
        else:
            st.write('else .........')
            print('else .........')
            for line in full_text:
                para.append(line) 
            st.write('else  .........  ...para len',len(para))
            print('else  .........  ...para len',len(para))
            berttext = ''.join( lines for lines in para) 
            st.write('else ......... in para max_sent_size=',max_sent_size,'...para len',len(para))
            print('else ......... in para max_sent_size=',max_sent_size,'...para len',len(para))
            bert_model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
            st.write('bert_model instantised  ..............')
            print('bert_model instantised  ..............')
            berttext = bert_model(berttext,max_length=max_abstract_token_size,num_sentences=max_sent_size)
            #return_text = ''.join( lines for lines in berttext)  
            bert_summary = ''.join( lines for lines in berttext)  
            
        st.write('BERT summary len :.....................',len(bert_summary))      
        print('BERT summary len :.....................',len(bert_summary))      
        ##col2.write('Abstract : This article describes,' + bert_summary)  
        col2.write(bert_summary)  
        
        st.write('................GPT filecount=',filecount)
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
        st.write('................GPT model created')
        gpt2text_summary = getTextSummarization(filecount,'GPT2',full_text,tot_words_ref,max_sent_size)
        st.write('................GPT summary')
        #col3.write('Abstract : This article describes,' + gpt2text_summary )  
        col3.write( gpt2text_summary )  
        st.markdown('----')
        st.subheader('Summarization Statistics')
