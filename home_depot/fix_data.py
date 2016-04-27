# -*- coding: utf-8 -*
"""correcting search terms with spellchecked versions"""

import pandas as pd
import numpy as np
import json
import re


def fix_text(s):
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
#     s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    s = s.replace("  "," ")
    
#     s = s.replace(",","") #could be number / segment later
#    s = s.replace("$"," ")
#    s = s.replace("?"," ")
    s = s.replace("-"," ")
    s = s.replace("//","/")
    s = s.replace("..",".")
    s = s.replace(" / "," ")
    s = s.replace(" \\ "," ")
    s = s.replace("."," . ")
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"(^\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)
#     s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
#     s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x "," xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*"," xbi ")
    s = s.replace(" by "," xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    s = s.replace(u"°"," degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v "," volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    s = s.replace("  "," ")
    s = s.replace(" . "," ")
    #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
#     s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
#     s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
#     s = s.lower()
    s = s.replace("toliet","toilet")
    s = s.replace("airconditioner","air conditioner")
    s = s.replace("vinal","vinyl")
    s = s.replace("vynal","vinyl")
    s = s.replace("skill","skil")
    s = s.replace("snowbl","snow bl")
    s = s.replace("plexigla","plexi gla")
    s = s.replace("rustoleum","rust-oleum")
    s = s.replace("whirpool","whirlpool")
    s = s.replace("whirlpoolga", "whirlpool ga")
    s = s.replace("whirlpoolstainless","whirlpool stainless")
    return s

with open("spell_check.tsv", "rb") as lines:
    spellcheck = dict(line.split("\t") for line in lines)


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train['search_term'] = train.search_term.map(lambda x: spellcheck.get(x, x)).map(fix_text)
test['search_term'] = test.search_term.map(lambda x: spellcheck.get(x, x)).map(fix_text)

train.to_csv("fixed_data/train.csv", index=False)
test.to_csv("fixed_data/test.csv", index=False)

df_pro_desc = pd.read_csv("data/product_descriptions.csv", encoding="ISO-8859-1")
attributes = pd.read_csv("data/attributes.csv", encoding="ISO-8859-1")


prod2bullets = {}
for prod, name, bullet in zip(attributes.product_uid, attributes.name, attributes.value):
    if np.isnan(prod) or type(bullet) != str or not name.startswith("Bullet"):
        continue
    prod2bullets.setdefault(prod, []).append(bullet)

from collections import Counter
import json
with open("dontsplit.txt", "rb") as lines:
    dontsplit = set(json.loads(line)[0] for line in lines)

def split_token(token):
    if token in dontsplit:
        return [token]
    for i, (c1, c2) in enumerate(zip(token, token[1:])):
        if (c1.islower() and c2.isupper()) or (c1 in ")%"):
            if i + 2 < len(token) and c1 + c2 + token[i+2] == "kWh":
                continue
            return [token[:(i + 1)], token[(i + 1):]]
    return [token]


def fix_description(prod_uid, desc):
    bullets = prod2bullets.get(prod_uid, [])
    for b in bullets:
        desc = desc.replace(b, "") + "\n" + b
    desc = desc.replace("Home Depot Protection Plan", " Home Depot Protection Plan")
    toktok = [[]]
    desc = fix_text(desc)
    for t in desc.split():
        split = split_token(t)
        if len(split) == 1:
            toktok[-1].append(t)
        else:
            a, b = split
            toktok[-1].append(a)
            toktok.append([b])
    return ("\n".join([" ".join(tok)  for tok in toktok]))

descriptions = [fix_description(pui, desc) for pui, desc in zip(df_pro_desc.product_uid, df_pro_desc.product_description)]

df_pro_desc['product_description'] = descriptions
df_pro_desc.to_csv("fixed_data/product_descriptions.csv", index=False, encoding="ISO-8859-1")