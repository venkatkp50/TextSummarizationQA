#from haystack.telemetry import tutorial_running
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
from haystack.nodes.retriever.multimodal import MultiModalRetriever
import os
from haystack import Pipeline
import torch
import streamlit as st



@st.cache
def getImageSessionpipeline():
    print(query,'........................getImageSession')
    sent_trans = 'sentence-transformers/nli-bert-base'
    document_store = InMemoryDocumentStore(embedding_dim=512)
    doc_dir = 'images/content'
    images = [Document(content=f"{doc_dir}/{filename}", content_type="image") for filename in os.listdir('images/content/') ]
    print('Images loaded........................getImageSession')
    document_store.write_documents(images)
    print('added to store........................getImageSession',images)
    #print(images)
    retriever_text_to_image = MultiModalRetriever(
    document_store=document_store,
    #query_embedding_model="sentence-transformers/clip-ViT-L-14",
    query_embedding_model="sentence-transformers/clip-ViT-B-32",
    #query_embedding_model=sent_trans,
    query_type="text",
    document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},)
    print('Retriver ........................getImageSession')
    document_store.update_embeddings(retriever=retriever_text_to_image)
    print('update embedding ........................getImageSession')
    pipeline = Pipeline()
    pipeline.add_node(component=retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])
    
    return pipeline

def getImage(query):
    # print(query,'........................')
    # sent_trans = 'sentence-transformers/nli-bert-base'
    # document_store = InMemoryDocumentStore(embedding_dim=512)
    # doc_dir = 'images/content'
    # images = [Document(content=f"{doc_dir}/{filename}", content_type="image") for filename in os.listdir('images/content/') ]
    # print('Images loaded........................')
    # document_store.write_documents(images)
    # print('added to store........................',images)
    # #print(images)
    # retriever_text_to_image = MultiModalRetriever(
    # document_store=document_store,
    # #query_embedding_model="sentence-transformers/clip-ViT-L-14",
    # query_embedding_model="sentence-transformers/clip-ViT-B-32",
    # #query_embedding_model=sent_trans,
    # query_type="text",
    # #document_embedding_models={"image": "sentence-transformers/clip-ViT-L-14"},)
    # document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},)
    # print('Retriver ........................')
    # document_store.update_embeddings(retriever=retriever_text_to_image)
    # print('update embedding ........................')
    # pipeline = Pipeline()
    # pipeline.add_node(component=retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])
    pipeline = getImageSessionpipeline()
    results = pipeline.run(query=query, params={"retriever_text_to_image": {"top_k": 1}})
    results = sorted(results["documents"], key=lambda d: d.score, reverse=True)

    images_array = [doc.content for doc in results]
    scores = [doc.score for doc in results]

    # print(images_array)
    # print(scores)

    return images_array,scores


    
