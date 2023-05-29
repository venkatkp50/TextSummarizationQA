
import io
import json
import time
import math
import requests
import rouge
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import re, os, string, random, requests
#from summarizer import Summarizer,TransformerSummarizer
#from bert_score import score
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate import meteor


from BERTSummarizer import getBERTSummary

from GPT2Summarizer import getGPT2Summary
from multiDocumnentSummaryGPT2 import preprocess_documents,generate_summary
from haystackReader import getReaderResult
from textSentiment import getLable,pretty_print_zero_shot,getSentiment
from posAnalysis import getDispacy
from textSimilarity import getSimilarityScore
from cosineSimilarity import getcosineSimilarity,getjaccardSimilarity
from keySentences import analyze,get_top_sentences
import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
new_stopwords = ["What"]
BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
doc_dir = 'text_file'


import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
imagename = Image.open('images/caronavirus banner.jpg')



import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

rouge = rouge.Rouge()
imagename2 = Image.open('images/Sidebar2.jpg')
st.sidebar.image(imagename2)
st.sidebar.title('Settings')
modelSelected = st.sidebar.selectbox('Choose Reader Model',options=('deepset/roberta-base-squad2-covid','deepset/roberta-base-squad2','deepset/covid_bert_base'))

# mystyle = '''
#     <style>
#         p {
#             text-align: justify;
#         }
#     </style>
#     '''
def rerun():
    file2 = open("sessioncount.txt","w+")
    file2.write('0')
    file2.close()
    st.session_state.input_text = ''
    #st.experimental_rerun()

def getTextfromList(text):
    header =[]
    para = []
    for line in sent_tokenize(text):
        if not line is np.nan:
            if len(line) > 1:
                if len(line) < 100:
                    header.append(line)
                else:
                    para.append(line)
                    
    text =  ''.join(line for line in para)
    return text

def runSumm():     
    file2 = open("sessioncount.txt","w+")
    file2.write('1')    
    file2.close()
    file3 = open("sessioncount.txt","r")
    print('1...............',file3.read())
    file3.close()
    user_message = st.session_state.input_text
   
    if user_message != '':
        data = pd.read_csv('json2csv.csv')
        print('created json2csv .....')
        print('inside user_meassage block')
        imagename = Image.open('images/caronavirus banner.jpg')
        st.image(imagename)
        st.write(user_message)
        #results = pipe.run(query=user_message,params={"Retriever": {"top_k": 10},"Reader": {"top_k": 5}})
        results = getReaderResult(doc_dir,modelSelected,user_message)
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
 
#     print('.....10')
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
                if data[data['paper_id'] == id[filecount].replace('.txt','')]['abstract'].values[0] is np.nan:
                    return_text = ''
                else:
                    return_text = data[data['paper_id'] == id[filecount].replace('.txt','')]['abstract'].values[0]
            return return_text


        tab1, tab2 , tab3 = st.tabs(["Single Document Summarization", "Summarized Text Analysis","Multi Document Summarization"])

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
            col1.write(gold_text, align_text='justify')  

            tot_words_ref = len(word_tokenize(gold_text))
            max_abstract_token_size  = math.ceil(tot_words_ref / 100) * 100
            max_sent_size = math.ceil(len(sent_tokenize(gold_text))/10)*10
            full_text = data[data['paper_id'] == id[filecount].replace('.txt','')]['text'].values[0]

            bert_summary = getBERTSummary(filecount,'BERT',full_text,max_abstract_token_size,max_sent_size)
            print('Outside BERT ..............')
            col2.write(bert_summary)  
  
            gpt2_summary = getGPT2Summary(filecount,'GPT2',full_text,max_abstract_token_size,max_sent_size)
            col3.write( gpt2_summary )  
        
        
            st.markdown('----')
            st.subheader('Summarization Statistics')
        
            col1 , col2, col3 = st.columns(3)
        
            tot_words_bert = len((bert_summary.split()))
            tot_words_gpt3 = len((gpt2_summary.split()))
            col1.metric('Total Words Reference Text',tot_words_ref)
            col2.metric("Total Words BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
            col3.metric("Total Words GPT-2 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )

            tot_words_ref = len(sent_tokenize(gold_text))
            tot_words_bert = len(sent_tokenize(bert_summary))
            tot_words_gpt3 = len(sent_tokenize(gpt2_summary))
        
            col1.metric('Total Sentences Reference Text',tot_words_ref)        
            col2.metric("Sentences in BERT Summarization", tot_words_bert,(tot_words_bert - tot_words_ref))
            col3.metric(" Sentences in GPT-2 Summarization",tot_words_gpt3,(tot_words_gpt3 - tot_words_ref) )
            st.markdown('----')

            st.subheader('Performance Analysis of Text-Summary')     
            
            bertscores = rouge.get_scores(hyps=gold_text, refs=bert_summary, avg=True)        
            gpt2scores = rouge.get_scores(hyps=gold_text, refs=gpt2_summary, avg=True)   

            col1, col2, col3 = st.columns(3)
        
            col2.write('BERT Score')
            bertscore = pd.DataFrame(bertscores)
            col2.table(bertscore)

            col3.write('GPT-2 Score')
            gpt2score = pd.DataFrame(gpt2scores)
            col3.table(gpt2score)

            dfbert = bertscore.T
            dfbert['Model'] = 'BERT'
            dfgpt = gpt2score.T
            dfgpt['Model'] = 'GPT-2'
            df = pd.concat([dfbert,dfgpt])

            st.markdown('---')

            target  = ['BERT','GPT-2']
            r1 = [bertscore.loc['f','rouge-1'],gpt2score.loc['f','rouge-1']]
            r2 = [bertscore.loc['f','rouge-2'],gpt2score.loc['f','rouge-2']]
            r3 = [bertscore.loc['f','rouge-l'],gpt2score.loc['f','rouge-l']]
            refs = []
            lines =[]
            for sent in sent_tokenize(gold_text):
                for line in sent.split():
                    lines.append(line)
                refs.append(lines)


            bert_cands = [ cand for cand in bert_summary.split()]
            bert_beluscore = sentence_bleu(refs, bert_cands)
        #st.write(bert_beluscore)
            gpt_cands  = [cand for cand in gpt2_summary.split()]
            gpt_beluscore = sentence_bleu(refs, gpt_cands)
            belu = [ bert_beluscore,gpt_beluscore]

            bert_metor = meteor([word_tokenize(gold_text)],word_tokenize(bert_summary))
            gpt_metor = meteor([word_tokenize(gold_text)],word_tokenize(gpt2_summary))

            metor= [ bert_metor,gpt_metor]
            radardf = pd.DataFrame()
            radardf['ROUGE-1 F1'] = r1
            radardf['ROUGE-2 F1'] = r2
            radardf['ROUGE-L F1 '] = r3
            radardf['BELU'] = belu
            radardf['METOR'] = metor

            fig = go.Figure()
            colors= ["dodgerblue", "yellow", "tomato" ]
            for i in range(2):
                fig.add_trace(go.Scatterpolar(r=radardf.loc[i].values, theta=radardf.columns,fill='toself',
                                              name=target[i],
                                              fillcolor=colors[i], line=dict(color=colors[i]),showlegend=True, opacity=0.6))
            st.subheader("Performance of Models over different evaluation metrics")
            radarmax = radardf.max()
            radmaxval =  radarmax.max()             
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.0, radmaxval])),)
            st.write(fig)
            st.table(radardf)
            st.markdown('-----')

        with tab2:
             st.write('Tab 2')
            # pol, sub = getSentiment(gpt2_summary)
            # st.subheader('Text Sentiment')
            # st.markdown('----')
            # col1, col2= st.columns(2)
            # col1.slider('Polarity', -1.0, 1.0, pol)
            # col2.slider('Subhectivity', 0.0, 1.0, sub)
            # st.subheader('Text Classification')
            # st.markdown('----')
            # sentiment = pretty_print_zero_shot(gpt2_summary)
            # labels = getLable()
            # senti = sentiment
            # col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8,gap="small")
            # if labels[0] == senti:
            #     col1.error(labels[0])
            # else:
            #     col1.write(labels[0])

            # if labels[1] == senti:
            #     col2.error(labels[1])
            # else:
            #     col2.write(labels[1])

            # if labels[2] == senti:
            #     col3.error(labels[2])
            # else:
            #     col3.write(labels[2])

            # if labels[3] == senti:
            #     col4.error(labels[3])
            # else:
            #     col4.write(labels[3])

            # if labels[4] == senti:
            #     col5.error(labels[4])
            # else:
            #     col5.write(labels[4])

            # if labels[5] == senti:
            #     col6.error(labels[5])
            # else:
            #     col6.write(labels[5])

            # if labels[6] == senti:
            #     col7.error(labels[6])
            # else:
            #     col7.write(labels[6])

            # if labels[7] == senti:
            #     col8.error(labels[7])
            # else:
            #     col8.write(labels[7])

            # st.markdown('----')
            # st.subheader('Parts Of Speeach Analysis')
            # #st.write(gpt2_summary)
            
            # spacyDisplay = getDispacy(gpt2_summary)
            # st.markdown(spacyDisplay, unsafe_allow_html=True)
            
            # st.markdown('----')
        with tab3:
            polarity = []
            subjectivity = []
            senti = []
            similarity_score = []
            cosine_similarity = []
            jaccard_similarity = []
            text1 = data[data['paper_id'] == ids[0].replace('.txt','')]['text'].values[0]                
            text1Str = getTextfromList(text1)
            
            for id in ids: 
                text2 = data[data['paper_id'] == id.replace('.txt','')]['text'].values[0]
                text2Str = getTextfromList(text2)                 
                pol, sub = getSentiment(text2)

                cosine_similarity.append(getcosineSimilarity(text1Str,text2Str))
                jaccard_similarity.append(getjaccardSimilarity(text1Str,text2Str))
                similarity_score.append( getSimilarityScore(text1Str,text2Str))
                polarity.append(pol)
                subjectivity.append(sub)

            df3 = pd.DataFrame({'File_name':ids,'Score':score,'Polarity':polarity,'Subjectivity':subjectivity,'BERT Similarity':similarity_score,'Cosine similarity ':cosine_similarity,'Jaccard Similarity':jaccard_similarity})
            st.table(df3)
            
            for highlight in analyze(gpt2_summary):
                st.write(highlight)


            st.markdown('----')

        new_query = st.button('New Query',on_click=rerun)




doc_dir = 'text_file'
file1 = open("sessioncount.txt","r")
runtime = file1.read()
print('2................',runtime)
file1.close()
if runtime == '0':
    st.image(imagename)
    textinput = st.text_input("Your Query", key="input_text",value='',on_change=runSumm)    


    




