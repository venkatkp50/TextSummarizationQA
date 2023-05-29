from transformers import pipeline
from spacy import displacy
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

ner_colors = {
    "CC": 1, "CD": 2, "DT": 3, "EX": 4, "FW": 5, "IN": 6, "JJ": 7, "JJR": 8, "JJS": 9, "MD": 10, "NN": 11, "NNP": 12, 
    "NNPS": 13, "NNS": 14, "O": 0, "PDT": 15, "POS": 16, "PRP": 17, "RB": 18, "RBR": 19, "RBS": 20, "RP": 21, "SYM": 22,
    "TO": 23, "UH": 24, "VB": 25, "VBD": 26, "VBG": 27, "VBN": 28, "VBP": 29, "VBZ": 30, "WDT": 31, "WP": 32, "WRB": 33
}
cmap = plt.cm.get_cmap('rainbow', len(ner_colors))
ner_colors = {k:matplotlib.colors.rgb2hex(cmap(v-1)) for k,v in ner_colors.items()}
ner_1 = pipeline("ner", grouped_entities=True)
ner_2 = pipeline("ner", model="mrm8488/mobilebert-finetuned-pos", grouped_entities=True)

def convert_hf_to_displacy_format(hf_pred, _original_text, _title=None):
    """ Function to convert prediction to the displacy specific format """
    return [dict(
        text=_original_text, 
        ents=[{
            "start":ent["start"], 
            "end":ent["end"], 
            "label":ent["entity_group"], 
            "score":ent["score"]} for ent in hf_pred], 
        title=_title
    ),]



def getDispacy(text):
    original_text = text    
    ner_pred = ner_1(original_text)
    return displacy.render(convert_hf_to_displacy_format(ner_pred, original_text), style="ent", manual=True,jupyter=False)
