# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:39:28 2021

@author: jnadar4
"""
from loguru import logger
import numpy as np
from flask import Flask,jsonify,render_template
import pickle
import tensorflow as tf
import tensorflow_hub as t_hub
import nltk
import re
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd 
from random import randint
from time import sleep
from inscriptis import get_text
import urllib.request
import spacy
from spacy import displacy
import pandas as pd
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
from urllib.error import HTTPError
# importing model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
vec_model = pickle.load(open('vec_model.pkl','rb'))
model = pickle.load(open('supervised_model.pkl','rb'))
embed = t_hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4')

search_dict = ['CEO','COO','Manager','CFO','CTO','Chief Executive Officer','Chief Operating Officer','Chief Techinical Offer','CIO','Chief Information Officer','Clinical Manager,','Director']

find_dict = {
    0:'Administrative Services',
    1:'Consulting',
    2:'E-Learning',
    3:'Financial Services',
    4:'HealthCare',
    5:'IT',
    6:'Marketing',
    7:'Network',
    8:'Other',
    9:'Semiconductor'
}



def delay():
    sleep(randint(2,10))
    return None


# Alternate recursive link generator, Using BFS algorithm to traverse the tree
def get_link_rec(base,path,visited,max_depth=3,init_depth=0):
    final_lst=[]
    if init_depth<max_depth:
        try:
            soup = bs(requests.get(base+path).text,'html.parser')
            for i in soup.find_all("a"):
                link = i.get("href")
                if link!=None:
                    final_lst.append(link)
                if link not in visited:
                    if link!=None:
                        final_lst.append(link)
                    visited.add(link)
                    
                    if link.startswith("http"):
                        get_link_rec(base,"",visited,max_depth,init_depth+1)
                    else:
                        get_link_rec(base,link,visited,max_depth,init_depth+1)
                       
        except:
            pass
    
    return final_lst
                

# Scrapper Function
def scrapper(base_url):
    try:
        html = urllib.request.urlopen(base_url).read().decode('utf-8')
        text = get_text(html)
    except:
        return 'FOrbidden 403'
# =============================================================================
#     except HTTPError as e:
#         if e.code == 403:
#             return 'Forbidden 403'
# =============================================================================
    return text 

# Function to join strings found from one master link
# This function returns a string which contains all the content fetched from the website and its recursive links
def join_content(scrapper_lst):
    s = ', '.join(scrapper_lst)
    return s

# Function where we will be getting the scraped data along with the corresponding parent url
def main_working_func(parent_url):
    logger.debug('Scrapping the URL')
    di = dict({
        'url':[],
        'data':[]
    })
    sub_di = dict({
        'parent_url':[],
        'sub_url':[],
        'sub_data':[]
    })
    lst3=[] #temporary list to store the fetched data
    store_data = scrapper(parent_url)
    di['url'].append(parent_url)
    sub_di['parent_url'].append(parent_url)
    sub_di['sub_url'].append('-')
    sub_di['sub_data'].append(store_data)
    lst3.append(store_data)
    
    rec_lst = get_link_rec(parent_url,"",set([parent_url]))
    http_lst=[]
    normal = []
    final_data= []
    sent = []
    for i in rec_lst:
        if i.startswith('https') or i.startswith('http'):
            http_lst.append(i)
        if i.endswith('.html') or i.startswith('#'):
            if not(i.startswith('https') or i.startswith('http')):
                normal.append(i)

    for i in http_lst:
        if i.startswith('https') or i.startswith('http'):
            store_rec_data = scrapper(i)
            #di['data'].append(store_rec_data)
            #sub_di['parent_url'].append(parent_url)
            #sub_di['sub_url'].append(i)
            #sub_di['sub_data'].append(store_rec_data)
            final_data.append(store_rec_data)
            
        if i==None :
            continue    
    for i in normal:
        store_rec_data = scrapper(parent_url+i)
        #di['data'].append(store_rec_data)
        #sub_di['parent_url'].append(parent_url)
        #sub_di['sub_url'].append(i)
        #sub_di['sub_data'].append(store_rec_data)
        final_data.append(store_rec_data)

    #final_str = join_content(lst3)
    #di['data'].append(final_str)
    final_text = join_content(final_data)
    sent.append(final_text)
   
    
    return sent

def listtoString(s):
    str1=" "
    return (str1.join(s))

def key_people(text):
    doc = nlp(text)
    person = []
    # printing the named enitites
    for word in doc.ents:
        #print(word.text,word.label_)
        if word.label_ == 'PERSON':
            person.append(word.text)
    return person

def checkWord(text,lst):
    temp = {
        'name':[],
        'designation':[]
    }
    
    for person in lst:
        idx = text.find(person)
        for des in search_dict:
            for m in re.finditer(des,text):
                temp['name'].append(person)
                temp['designation'].append([des,abs(m.start()-idx)])
        
    return temp

def changeDict(store_dict):
    logger.debug('Getting associated People')
    n = len(store_dict['name'])
    for i in range(n):
        if i == n-1 or store_dict['name'][i] != store_dict['name'][i+1]:
           break   
    count_of_single_name = i+1
    number_of_name = n//count_of_single_name
    
    final_dict = dict()
    
    i=0
    j=0
    x = number_of_name
    
    while(x>0):
        temp = store_dict['designation'][i:i+count_of_single_name]
        temp.sort(key= lambda x:x[1])
        final_dict[store_dict['name'][j]] = str(temp[0][0])
        i+=count_of_single_name
        j+=count_of_single_name
        x-=1
    return final_dict


def cleanText(text):
    new_text = re.sub(r'\[[0-9]*\]',' ',text)
    new_text = re.sub(r'\s+',' ',new_text)
    new_text = new_text.lower()
    new_text = re.sub(r'\d',' ',new_text)
    new_text = re.sub(r'\s+',' ',new_text)
    return new_text

def paraToSent(text):
    new_text=cleanText(text)
    sent = nltk.tokenize.sent_tokenize(new_text)
    return sent


def get_pred(sent_list):
    embeddings = embed(sent_list)
    use_vec = np.array(embeddings)
    count_vec = np.array(vec_model.transform(sent_list).todense())
    combined = np.hstack((use_vec,count_vec))
    return model.predict(combined)

def prob_rfc(sent_list):
    print(sent_list)
    embeddings = embed(sent_list)
    use_vec = np.array(embeddings)
    count_vec = np.array(vec_model.transform(sent_list).todense())
    combined = np.hstack((use_vec,count_vec))
    print(count_vec)
    return model.predict_proba(combined)


def cluster_finder(prob_lst):
    logger.debug('Getting the domain')
    count =[0]*10
    avg =[0]*10
    for sent in range(len(prob_lst)):
        cls = prob_lst[sent].argmax()   
        val = prob_lst[sent].max()
        count[cls]+=1
        avg[cls]+=val

    for i in range(0,10):
        if count[i]!=0:
            avg[i]=avg[i]/count[i]
    
    idx = avg.index(max(avg))
    return find_dict[idx]
    
# =============================================================================
# def sentence_similarity(s1,s2,stopwords=None):
#     if stopwords is None:
#         stopwords = []
#     
#     s1=[i.lower() for i in s1]
#     s2=[i.lower() for i in s2]
#     
#     all_words = list(set(s1+s2))
#     
#     vec1 = [0]*len(all_words)
#     vec2 = [0]*len(all_words)
#     
#     for i in s1:
#         if i in stopwords:
#             continue
#         vec1[all_words.index(i)] += 1
#     
#     for i in s2:
#          if i in stopwords:
#             continue
#     
#          vec2[all_words.index(i)] += 1   
#     return 1-cosine_distance(vec1,vec2)
# 
# def generate_summary(text,top_n=5):
#     article = text.split(". ")
#     sentences = []
#     for sentence in article:
#         sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
#     stop_words = stopwords.words('english')
#     
#     similarity_matrix = np.zeros((len(sentences),len(sentences)))
#     
#     for id1 in range(len(sentences)):
#         for id2 in range(len(sentences)):
#             if id1 == id2:
#                 continue
#             similarity_matrix[id1][id2] = sentence_similarity(sentences[id1],sentences[id2],stop_words)
#             
#     sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
#     scores = nx.pagerank(sentence_similarity_graph)
#     ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
#     summarize_text = []
#     for i in range(top_n):
#         summarize_text.append(" ".join(ranked_sentence[i][1]))
#     return ('.'.join(summarize_text))
# 
# 
# 
# =============================================================================
# =============================================================================

def summary(text):
    return 'Summary to be added'
def main(url):
    text = main_working_func(url)
    list_to_str = listtoString(text)
    people_dict = checkWord(list_to_str,key_people(list_to_str))
    associated_ppl = changeDict(people_dict)
    sent=paraToSent(list_to_str)
    prob_lst = prob_rfc(sent)
    domain_info = cluster_finder(prob_lst)
    summary_part = summary(list_to_str)
    output = {'domain_info':domain_info ,'Associated_people':associated_ppl, 'Summary':summary_part}
    #output = jsonify(output)
    
    return output

print(main('https://www.uhcglobalclinicaljournal.com'))
    


# 