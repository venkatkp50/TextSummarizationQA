
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

new_stopwords = ["What"]
BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
import warnings
warnings.filterwarnings('ignore')

#bert_model = Summarizer() 
#GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")

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

    # selected_radio = st.radio('Choose File for Summarization',options=(ans[0],ans[1],ans[2],ans[3],ans[4]))
    # file4Summ = ''
    # filecount = 0
    # #file4Summ = id[0]
    # if selected_radio == ans[0]:
    #     filecount = 0
    # elif selected_radio == ans[1]:
    #     filecount = 1
    # elif selected_radio == ans[2]:
    #     filecount = 2
    # elif selected_radio == ans[3]:
    #     filecount = 3
    # else:
    #     filecount = 4


