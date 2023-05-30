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
import streamlit as st

def getReaderResult(doc_dir,modelSelected,user_message):
    st.write('inside getReaderResult..............')
    document_store = InMemoryDocumentStore(use_bm25=True)
    st.write('doc_strore ..............acquired')
    docs = convert_files_to_docs(dir_path=doc_dir,clean_func=clean_wiki_text,split_paragraphs=True)
    st.write('doc conversion ..............')
    document_store.write_documents(docs)
    st.write('doc conversion ..............')
    retriever = BM25Retriever(document_store=document_store)
    st.write('retriever ..............')
    reader = FARMReader(model_name_or_path=modelSelected, use_gpu=True)
    st.write('reader ..............')
    pipe = ExtractiveQAPipeline(reader, retriever)
    st.write('build pipeline ..............')
    results = pipe.run(query=user_message,params={"Retriever": {"top_k": 10},"Reader": {"top_k": 5}})
    st.write('completed reader ')
    return results
