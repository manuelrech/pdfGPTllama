import path
import sys

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from llama_index import Document

def abstract_parser(file_dict):
    abstract = file_dict['abstract']
    abstractDoc = Document(
                text = abstract,
                metadata = {
                 'filename': file_dict['file'],
                 'title': file_dict['title'],
                 'section': 'abstract',
                 'pub_date': file_dict['pub_date'],
                 'doi': file_dict['doi']
                 }, 
                excluded_llm_metadata_keys = ['filename', 'title', 'section', 'pub_date', 'doi']

                )

    return abstractDoc

def corpus_parser(file_dict):
    corpus = file_dict['corpus']

    corpus_doc_list = []
    for section in corpus:
        corpusDoc = Document(
                text = section['section_text'],
                metadata={
                    'filename': file_dict['file'],
                     'title': file_dict['title'],
                     'section': section['section_name'],
                     'pub_date': file_dict['pub_date'],
                     'doi': file_dict['doi'],
                     }, 
                excluded_llm_metadata_keys = ['filename', 'title', 'section', 'pub_date', 'doi']
                )
        corpus_doc_list.append(corpusDoc)

    return corpus_doc_list
