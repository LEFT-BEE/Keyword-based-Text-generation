import nltk
import numpy as np
nltk.download('all')
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize , word_tokenize , pos_tag
import requests

def get_translate(text):
    client_id = "3MzBqRVmWx6rKqNjJDtX" # <-- client_id 기입
    client_secret = "kGvyCB_uxy" # <-- client_secret 기입

    data = {'text' : text,
            'source' : 'ko',
            'target': 'en'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)

def swn_porlarity(sentence):
  sent_scores = []
  for word in sentence:
    sent_score = 0
    synsets = wn.synsets(word)
    synsets = synsets[:2]
    for synset in synsets:
      synset_name = synset.name()
      senti_synset = swn.senti_synset(synset_name)
      sent_score += senti_synset.pos_score() + senti_synset.neg_score()
    sent_scores.append(sent_score)

  return sent_scores  

def find_sent_word(kr_sentence):
  en_sentence = get_translate(kr_sentence)
  sparse_sentence = nltk.word_tokenize(en_sentence)
  print(sparse_sentence  , len(sparse_sentence))
  sent_score = swn_porlarity(sparse_sentence)
  print(sent_score ,  len(sent_score))
  sent_word = sparse_sentence[np.argmax(sent_score)]
  return sent_word