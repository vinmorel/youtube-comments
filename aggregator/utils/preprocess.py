import re
import spacy
import numpy as np
from langdetect import detect
from langdetect import DetectorFactory
import demoji


def check_dim(dim2_list: list) -> list:
    # Checks list dimension to make sure it is 2 dimension : [['foo']] 
    arr = np.array([dim2_list])
    if len(arr.shape) == 1:
        dim2_list = [dim2_list]
    return dim2_list

def check_lang(comment_list: list) -> list:
    DetectorFactory.seed = 0
    return [comment for comment in comment_list if detect(comment) == 'en']

def prune_encodings(comment: str) -> str:
    pruned_enc = comment.replace("\n"," ")
    pruned_enc = pruned_enc.replace("\r","")
    return pruned_enc

def make_lowercase(comment: str) -> str:
    return comment.lower()

def prune_links_emails(comment: str) -> str:
    pruned_links = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", comment)
    pruned_links = re.sub(r'\S*@\S*\s?', " ", pruned_links) # email addresses
    return pruned_links

def prune_punctuation(comment: str) -> str:
    bad_punctuations =  ['[()]', '[\[\]]', '[\{\}]', '[\*]','[“”]']
    for bad_punc in bad_punctuations:
        comment = re.sub(bad_punc, '', comment)
        
    bad_punctuations2 =  [r"\s+:\)*\s?", r"\s+:\(*\s?", r"\s+\(:*\s?", r"\s+:p*\s?", r"\s+:d*\s?", 
                            r"\s+:3*\s?", r"\s+\<3*\s?", r"\s+:o*\s?", r"\s+xd*\s?"]
    for bad_punc in bad_punctuations2:
        comment = re.sub(bad_punc, ' ', comment)

    bad_punctuations3 =  ['!','?','.',',',':',';', '$', '%', '`', '"', '_']
    for bad_punc in bad_punctuations3:
        comment = comment.replace(bad_punc, " ")
    
    bad_punctuations4 = ['=', '-', '+', '/', '^', '&', '|', '<', '>']
    for bad_punc in bad_punctuations4:
        comment = comment.replace(bad_punc, " ")    
    return comment

def prune_emojis(comment: str) -> str:
    try:
        comment = demoji.replace(comment, " ")
    except OSError as e:
        demoji.download_codes()
        comment = demoji.replace(comment, " ")
    return comment

def fix_spaces(comment:str) -> str:
    return " ".join(comment.split())

class lemmatizer():
    def __init__(self, model: str = 'en_core_web_sm', batch_size: int = 100):
        self.nlp = spacy.load(model)
        self.nlp.disable_pipes('ner')
        self.batch_size = batch_size

    def lemmatize(self, text: str):
        docs = self.nlp.pipe(text, batch_size=self.batch_size)
        lemmatized_comment_list = []
        for doc in docs:
            lemmas = ' '.join([token.lemma_ for token in doc])
            lemmatized_comment_list.append(lemmas)
        return lemmatized_comment_list


def preprocess(comment_list: list) -> list:
    # Applies all preprocessing operations to extracted list of comments
    l = lemmatizer()
    comment_list = check_dim(comment_list)

    preprocessed_comments = []
    for comment in comment_list:
        comment = prune_encodings(comment)
        comment = make_lowercase(comment)
        comment = prune_links_emails(comment)
        comment = prune_punctuation(comment)
        comment = prune_emojis(comment)
        comment = fix_spaces(comment)
        preprocessed_comments.append(comment)

    preprocessed_comments = check_lang(preprocessed_comments)
    print(preprocessed_comments)
    preprocessed_comments = l.lemmatize(preprocessed_comments)

    return preprocessed_comments

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    wdir = Path(__file__).resolve().parents[0]
    save_dir = wdir / "response.pickle"

    with open(save_dir, 'rb') as handle:
        data = pickle.load(handle)
        comments = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in data['items']]
    
        preprocessed_comments = preprocess(comments)
        print(preprocessed_comments)
