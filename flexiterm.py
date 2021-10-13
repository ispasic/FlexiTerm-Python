#!/usr/bin/env python
# coding: utf-8

# # --- FlexiTerm: multi-word term recognition




# --- dependencies ---

import csv
import jellyfish
import json
import math
import numpy as np
import os
import pprint
import random
import re
import spacy
import sqlite3
import sys
import time
from nltk.stem.porter import PorterStemmer
from pathlib import Path
from spacy.matcher import PhraseMatcher
from spacy import displacy


# # --- setting up




# --- database connection ---

schema = "./config/schema.sql"           # --- read database schema

try: 
    with open(Path(schema),'r') as file:
        sql_script = file.read()
        file.close()
        print(sql_script[0:100] + '...') # --- preview schema
except:
    print("ERROR: Schema file " + schema + " not found. Unable to create the tables.\n")
    quit()

# --- database connection
con = sqlite3.connect('flexiterm.sqlite')

# --- cursor (statement) objects to execute SQL queries
cur1 = con.cursor()
cur2 = con.cursor()
cur3 = con.cursor()

# --- create database tables
cur1.executescript(sql_script)
con.commit()





# --- default settings ---

default = {
   "pattern"  : "(((((NN|JJ) )*NN) IN (((NN|JJ) )*NN))|((NN|JJ )*NN POS (NN|JJ )*NN))|(((NN|JJ) )+NN( CD)?)",
   "stoplist" : "./config/stoplist.txt",
   "Smin"     : 0.962,
   "Amin"     : 5,
   "Fmin"     : 2,
   "Cmin"     : 1,
   "acronyms" : "explicit"
}





# --- load settings ---

settings_file = "./config/settings.json"

try: 
    with open(Path(settings_file),"r") as file:
        
        settings = json.load(file)
        file.close()
        
        if "pattern" in settings: 
            pattern = settings["pattern"]
            try: re.compile(pattern)
            except re.error: 
                print("WARNING: Invalid POS pattern: " + pattern)
                print("         Using the default instead.\n")
                pattern = default["pattern"]

        if "stoplist" in settings: 
            stoplist = settings["stoplist"]
            if not os.path.isfile(stoplist):
                print("WARNING: Stoplist file " + stoplist + " not found.")
                print("         Using the default instead.\n")
                stoplist = default["stoplist"]

        if "Smin" in settings:
            Smin = settings["Smin"]
            if not (0 < Smin and Smin < 1):
                print("WARNING: Invalid token similarity threshold:", Smin);
                print("         Using the default instead.\n")
                Smin = default["Smin"]

        if "Amin" in settings: 
            Amin = settings["Amin"]
            if type(Amin) != int:
                print("WARNING: Invalid acronym frequency threshold:", Amin);
                print("         Using the default instead.\n")
                Amin = default["Amin"]
        
        if "Fmin" in settings:
            Fmin = settings["Fmin"]
            if type(Fmin) != int:
                print("WARNING: Invalid term frequency threshold:", Fmin);
                print("         Using the default instead.\n")
                Fmin = default["Fmin"]
            
        if "Cmin" in settings:
            Cmin = settings["Cmin"]
            if Cmin < 0.7:
                print("WARNING: Invalid token C-value threshold:", Cmin);
                print("         Using the default instead.\n")
                Cmin = default["Cmin"]
                
        if "acronyms" in settings:
            acronyms = settings["acronyms"]
            if acronyms not in ["explicit", "implicit"]:
                print("WARNING: Invalid acronyms value:", acronyms);
                print("         Using the default instead.\n")
                acronyms = default["acronyms"]

except:
    print("WARNING: Settings file " + settings_file + " not found. Using the default values instead.\n")
    
    pattern = default["pattern"]
    stoplist = default["stoplist"]
    Smin = default["Smin"]
    Amin = default["Amin"]
    Fmin = default["Fmin"]
    Cmin = default["Cmin"]
    acronyms = default["acronyms"]

print("--- Settings ---")
print("* pattern  :", pattern)
print("* stoplist :", stoplist)
print("* Smin     :", Smin)
print("* Amin     :", Amin)
print("* Fmin     :", Fmin)
print("* Cmin     :", Cmin)
print("* acronyms :", acronyms)
print("----------------")





# --- load stoplist ---

print("Loading stoplist from " + stoplist + "...");

try: 
    # --- read a CSV file
    table = open(Path(stoplist),'r')
    rows = csv.reader(table)
    
    # --- insert rows from the CSV file
    cur1.executemany("INSERT INTO stopword (word) VALUES (?);", rows)
    con.commit()

    table.close()

except sqlite3.Error as error: print(error)
    
file = open(Path(stoplist), 'r')
stopwords = file.read().split('\n')
file.close()





# --- load language model from spacy ---
nlp = spacy.load('en_core_web_sm')
sentencizer = nlp.add_pipe('sentencizer')





# --- delete previous output files if any

folder = "./out"
filename = ["annotations.json", 
            "concordances.html", 
            "corpus.html", 
            "terminology.html", 
            "terminology.csv"]

for name in filename:
    file_path = os.path.join(folder, name)
    if os.path.exists(file_path): os.remove(file_path)


# # --- load & preprocess input documents




start_time = time.perf_counter()
timer = []





# --- fix potential tagging issues
def pretagging(txt):
    
    unit = ["meter",
          "metre",
          "mile",
          "centi",
          "milli",
          "kilo",
          "gram",
          "sec",
          "min",
          "hour",
          "hr",
          "day",
          "week",
          "month",
          "year",
          "liter",
          "litre"]

    abbr = ["m",
          "cm",
          "mm",
          "kg",
          "g",
          "mg",
          "s",
          "h",
          "am",
          "pm",
          "l",
          "ml"]
    
    # --- insert white space in front of a unit where necessary
    for u in unit:
        txt = re.sub("(\d)" + u, "\\1 " + u, txt)

    for a in abbr:
        txt = re.sub("(\d)" + a, "\\1 " + a, txt)

    # --- compress repetative punctuation into a single character
    txt = re.sub("\\!+", "!", txt)
    txt = re.sub("\\?+", "?", txt)
    txt = re.sub("\\.+", ".", txt)
    txt = re.sub("\\-+", "-", txt)
    txt = re.sub("_+", "_", txt)
    txt = re.sub("~+", "~", txt)
    txt = re.sub("kappaB", "kappa B", txt)
    txt = re.sub('([a-z0-9])/([a-z0-9])', '\\1 / \\2', txt, flags=re.IGNORECASE)
    txt = re.sub("\(", " ( ", txt, flags=re.IGNORECASE)
    txt = re.sub("\)", " ) ", txt, flags=re.IGNORECASE)
    
    # --- remove long gene sequences
    txt = re.sub("[ACGT ]{6,}", "", txt);

    # --- normalise white spaces
    txt = re.sub("\\s+", " ", txt)

    # --- normalise non-ASCII characters
    # ???: test with unicode characters
    #txt = Normalizer.normalize(txt, Normalizer.Form.NFD);
    #txt = txt.replaceAll("[^\\x00-\\x7F]", "");
   
    return txt





# --- remove a hyphen between 2 letters so that it does not mess up the tokenisation in spacy: -/HYPH
# --- NOTE: not part of pretagging, because the hyphen is only ignored, not removed

def hyphen(txt):
    txt = re.sub('([a-z])\\-([a-z])', '\\1 \\2', txt, flags=re.IGNORECASE)
    # --- repeat for overlapping matches, as only one gets replaced,
    #     e.g. glutathione-S-transferase -> glutathione S-transferase -> glutathione S transferase
    txt = re.sub('([a-z])\\-([a-z])', '\\1 \\2', txt, flags=re.IGNORECASE)
    return txt





# --- generalise tags to simplify patterns (regex) specified in the settings

def gtag(tag):

    if (len(tag) <= 1):         tag = "PUN"
    elif (tag == "PRP$"):       tag = "PRP"
    elif (tag == "WP$"):        tag = "WP"
    elif (tag.find("JJ") == 0): tag = "JJ"
    elif (tag.find("NN") == 0): tag = "NN";
    elif (tag.find("RB") == 0): tag = "RB";
    elif (tag.find("VB") == 0): tag = "VB";
    
    return tag





# --- prepare lemma for stemming
def prestem(lemma):
    
    if len(lemma) > 1: 
        if lemma[0:1] == '-': lemma = lemma[1:]    # --- strip of hyphen at the start

    if len(lemma) > 1:
        if lemma[-1:] == '-': lemma = lemma[:-1]   # --- strip of hyphen at the end
    
    lemma = re.sub('isation', 'ization', lemma)    # --- American spelling for consistent stemming
    
    return lemma





# --- load data

#####
cur1.execute("DELETE FROM data_document;")
cur1.execute("DELETE FROM data_sentence;")
cur1.execute("DELETE FROM data_token;")
#####

stemmer = PorterStemmer()

# --- read documents from the "text" folder
folder = "./text"
print("Loading data from " + folder + "...");
n = 0
for doc_id in os.listdir(folder):
    n += 1
    file_path = os.path.join(folder, doc_id)
    if os.path.isfile(file_path):
        print('.', end='')
        file = open(file_path, "r", encoding="utf8")
        verbatim = file.read()
        file.close()
        content = pretagging(verbatim)
        
        row = (doc_id, content, verbatim)
        cur1.execute("INSERT INTO data_document(id, document, verbatim) VALUES(?, ?, ?);", row)
        
        # --- split sentences
        s = 0
        doc = nlp(hyphen(content))
        for sent in doc.sents: # --- store sentences
            s+=1
            sentence = sent.text
            tokens = " ".join([token.text for token in sent])
            for token in sent: 
                token.tag_ = gtag(token.tag_) # --- generalise tag, e.g. JJR --> JJ
                # --- prevent tagging of symbols and abbreviations as NNs
                if token.text == '%': token.tag_ = 'SYM'
                elif token.text.lower() in ('et', 'al', 'etc'): token.tag_ = 'XX'
                elif token.text.lower() in ('related', 'based'): token.tag_ = 'JJ'
            tags = " ".join([token.tag_ for token in sent])
            tagged_sentence = " ".join([token.text+"/"+token.tag_ for token in sent])
            sentence_id = doc_id+"."+str(s)
            row = (sentence_id, doc_id, s, sentence, tagged_sentence, tags)
            cur1.execute("INSERT INTO data_sentence(id, doc_id, position, sentence, tagged_sentence, tags) VALUES(?, ?, ?, ?, ?, ?)", row)
            
            # --- tokenise sentences
            p = 0
            for token in sent: # --- store tokens
                p+=1
                lemma = token.lemma_.lower()    # --- lemmatise
                lemma = prestem(lemma)          # --- prepare lemma for stemming
                stem = stemmer.stem(lemma)      # --- stem lemma
                row = (sentence_id, p, token.text, stem, lemma, token.tag_)
                cur1.execute("INSERT INTO data_token(sentence_id, position, token, stem, lemma, gtag) VALUES(?, ?, ?, ?, ?, ?)", row)

if n == 0:
    con.close()
    sys.exit('No input data found. Check the text folder.')
                
con.commit()
    
print('\nData loaded.')





cur1.execute("CREATE INDEX idx01 ON data_document(id);")
cur1.execute("CREATE INDEX idx02 ON data_token(sentence_id, position);")





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Data loaded in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- extract term candidates




start_time = end_time





# --- extract NPs of a predefined structure (the pattern in the settings)

#####
cur1.execute("DELETE FROM term_phrase;")
#####

print("Extracting term candidates...");

regex = re.compile(pattern)

cur1.execute("SELECT id, tags FROM data_sentence WHERE length(sentence) > 30;") # --- extract POS tags
rows1 = cur1.fetchall()
total = len(rows1)
n = 0
for row1 in rows1:
    
    # --- progress bar
    n += 1
    sys.stdout.write('\r')
    p = int(100*n/total)
    sys.stdout.write("[%-100s] %d%%" % ('='*p, p))
    sys.stdout.flush()
    
    sentence_id = row1[0]
    tags = row1[1]
    # --- match patterns
    for chunk in re.finditer(regex, tags):
        start = tags[:chunk.span()[0]].count(' ')+1
        length = tags[chunk.span()[0]:chunk.span()[1]].count(' ')+1
        
        # --- extract the corresponding tokens
        cur2.execute("""SELECT token
                        FROM   data_token
                        WHERE  sentence_id = ?
                        AND    position >= ? 
                        AND    position < ?
                        ORDER BY position ASC;""", (sentence_id, start, start+length))
        rows2 = cur2.fetchall()
        
        # --- trim leading stopwords
        tokens = []
        for row2 in rows2: tokens.append(row2[0]) 
        i = 0
        while length > 1:
            if tokens[i].lower() in stopwords:
                start += 1
                length -= 1
                i+=1
            else: break
        
        tokens = tokens[i:]
        
        # --- trim trailing stopwords
        i = len(tokens) - 1
        while length > 1:
            if tokens[i].lower() in stopwords:
                length -= 1
                i-=1
            else: break

        tokens = tokens[:i+1]
        
        # --- join tokens into a phrase
        phrase = " ".join(tokens)
        phrase_id = sentence_id+"."+str(start)
        
        # --- if still multi-word phrase and not too long
        if 1 < length and length < 8:
            
            # --- strip off possible . at the end
            if phrase.endswith('.'): phrase = phrase[:-1]
                
            # --- ignore phrases that contain web concepts: email address, URL, #hashtag
            if not(phrase.find("@")>=0 or 
                   phrase.find("#")>=0 or 
                   phrase.lower().find("http")>=0 or 
                   phrase.lower().find("www")>=0):
                # --- normalise phrase by stemming
                cur2.execute("""SELECT DISTINCT stem
                                FROM   data_token
                                WHERE  sentence_id = ?
                                AND    ? <= position AND position < ?
                                EXCEPT SELECT word FROM stopword
                                ORDER BY stem ASC;""", (sentence_id, start, start+length))
                                ###AND    NOT (LOWER(token) = token AND LENGTH(token) < 3)
                rows2 = cur2.fetchall()
                stems = []
                for row2 in rows2: stems.append(row2[0])
                normalised = " ".join(stems)
                normalised = normalised.replace('.', '') # --- e.g. U.K., Dr., St. -> UK, Dr, St
                
                # --- store phrase as a MWT candidate
                cur2.execute("""INSERT INTO term_phrase(id, sentence_id, token_start, token_length, phrase, normalised)
                                VALUES (?,?,?,?,?,?);""", (phrase_id, sentence_id, start, length, phrase, normalised))

cur1.execute("UPDATE term_phrase SET flat = LOWER(REPLACE(phrase, ' ', ''));")
con.commit()





cur1.execute("CREATE INDEX idx03 ON term_phrase(flat);")
cur1.execute("CREATE INDEX idx04 ON term_phrase(LOWER(phrase));")





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Term candidates extracted in {run_time:0.4f} seconds")
timer.append(run_time)





start_time = end_time





# --- re-normalise term candidates that have different TOKENISATION,
#     e.g. posterolateral corner B vs. postero lateral corner
# --- keep the one with MORE tokens (e.g. postero lateral corner)

cur1.execute("DELETE FROM tmp_normalised;")

cur1.execute("""INSERT INTO tmp_normalised(changeto, changefrom)
                SELECT P1.normalised, P2.normalised
                FROM   term_phrase P1, term_phrase P2
                WHERE  P1.flat = P2.flat
                AND    P1.token_length > P2.token_length
                AND    P1.normalised <> P2.normalised;""")

cur1.execute("""SELECT DISTINCT changefrom, changeto FROM tmp_normalised;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    changefrom = row1[0]
    changeto   = row1[1]
    print(changefrom, "-->", changeto)
    cur2.execute("UPDATE term_phrase SET normalised = ? WHERE normalised = ?;", (changeto, changefrom))

con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Term candidates normalised in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- acronym recognition method




start_time = end_time





# --- assumption: acronyms are explicitly defined in text, e.g. 
#     ... blah blah retinoic acid receptor (RAR) blah blah ...
#                   ~~~~~~~~~~~~~~~~~~~~~~  ~~~

# --- based on this paper:
#     Schwartz A & Hearst M (2003) 
#     A simple algorithm for identifying abbreviation definitions in biomedical text, 
#     Pacific Symposium on Biocomputing 8:451-462 [http://biotext.berkeley.edu/software.html]





# --- alpha -> a: helps properly estimate the acronym length and simplifies matching against the long form

def pad(string):
    return " " + string + " "

def greek2english(string):
    
    letters = ["alpha", "beta", "gamma", "delta", "epsilon", "zata", "eta", "theta", "iota","kappa", "lambda", "mu", "nu", "xi", "omikron", "pi", "rho", "sigma", "tau", "upsilon", "phi","chi", "psi", "omega"]
    
    string = pad(string)
    
    for letter in letters:
        string = re.sub(pad(letter), pad(letter[0:1]), string, flags=re.IGNORECASE)

    return string.strip()





# --- checks if a string looks like an acronym

def isValidShortForm(string):

    string = greek2english(string)
    
    if len(string) < 2:                                                                       # --- acronym too short
        return False
    elif len(string) > 8:                                                                     # --- acronym too long
        return False
    elif not(any(char.isupper() for char in string)):                                         # --- no uppercase
        return False
    elif (sum([int(c.islower()) for c in string]) > sum([int(c.isupper()) for c in string])): # --- more lowercase than uppercase
        return False
    elif not(string[0].isalpha() or string[0].isdigit() or string[0] == '('):                 # --- invalid first character
        return False
    elif (len(re.sub('[a-z0-9\s\'/\-]', '', string.lower())) > 0):                            # --- invalid characters present
        return False
    elif string[1] == "'":                                                                    # --- 2nd character ' as in A'
        return False

    return True





# --- processes the context (i.e. definition) to extract the best long form for a given acronym

def bestLongForm(acronym, definition):
    
    # --- case-insensitive matching
    acronym = acronym.replace("-", "").lower()
    definition = definition.lower()
    d = len(definition) - 1

    # --- go through the acronym & definition character by character,
    #     FROM RIGHT TO LEFT looking for a match

    for a in range(len(acronym) - 1, -1, -1):
        
        c = acronym[a]

        if (c.isalpha() or c.isdigit()):    # --- match an alphanumeric character
            
            while ((d >= 0 and definition[d] != c) or (a == 0 and d > 0 and (definition[d-1].isalpha() or definition[d-1].isdigit()))):
                d -= 1                      # --- keep moving to the left

        if (d < 0):                         # --- match failed
            return None
        else:                               # --- match found
            d -= 1                          # --- skip the matching character and then continue matching

    d = definition.rfind(' ', 0, d+1) + 1   # --- complete the left-most word (up to the white space)

    definition = definition[d:].strip()     # --- delete the surplus text on the left

    if definition.startswith('an '):     definition = definition[3:]    # --- starts with a determiner?
    elif definition.startswith('a '):    definition = definition[2:]
    elif definition.startswith('the '):  definition = definition[4:]
    elif (definition.startswith('[') and definition.endswith(']')):     # --- [definition]
        definition = definition[1:-1]
    elif (definition.startswith("'") and definition.endswith("'")):     # --- 'definition'
        definition = definition[1:-1]

    return definition





# --- extracts all potential (acronym, definition) pairs from a given sentence

def extractPairs(sentence):
    
    pairs = []

    # --- remove double quotes
    sentence = sentence.replace('"', ' ')
    
    # --- normalise white spaces
    sentence = re.sub('\s+', ' ', sentence)
    
    acronym = ''
    definition = ''
    o = sentence.find(' (')   # --- find (
    c = -1                    # --- ) index
    tmp = -1

    while (1 == 1):

        if (o > -1):
            o +=1                       # --- skip white space, i.e. ' (' -> '('
            c = sentence.find(')', o)   # --- find closed parenthesis
            
            # --- extract candidates for (acronym, definition)
            if (c > -1):
                # --- find the start of the previous clause based on punctuation
                cutoff = max(sentence.rfind('. ', 0, o), sentence.rfind(', ', 0, o))
                if (cutoff == -1): cutoff = -2

                definition = sentence[cutoff + 2:o].strip()
                acronym = sentence[o + 1:c].strip()
        
        if (len(acronym) > 0 or len(definition) > 0): # --- candidates successfully instantiated above

            if (len(acronym) > 1 and len(definition) > 1):
                # --- look for parentheses nested within the candidate acronym
                nextc = sentence.find(')', c + 1)
                if (acronym.find('(') > -1 and nextc > -1):
                    acronym = sentence[o + 1:nextc]
                    c = nextc

                # --- if separator found within parentheses, then trim everything after it
                tmp = acronym.find(', ')
                if (tmp > -1): acronym = acronym[0:tmp]
                tmp = acronym.find('; ')
                if (tmp > -1): acronym = acronym[0:tmp]
                tmp = acronym.find(' or ')
                if (tmp > -1): acronym = acronym[0:tmp]
                if (tmp > -1): acronym = acronym[0:tmp]

                # --- (or ...) -> (...)
                tmp = acronym.find('or ')
                if (tmp == 0): acronym = acronym[3:]

                tokens = acronym.split()
                if (len(tokens) > 3 or len(acronym) > len(definition)):
                    # --- definition found within (...)
                
                    # --- extract the last token before "(" as a candidate for acronym
                    tmp = sentence.rfind(' ', 0, o - 2)
                    substr = sentence[tmp + 1:o - 1]
                
                    # --- swap acronym & definition
                    definition = acronym
                    acronym = substr
                    
                    # --- validate (... definition ...)
                    if (len(definition.replace('-', ' ').split(' ')) > len(acronym) + 2):
                        acronym = '' # --- delete acronym

            acronym = acronym.strip()
            definition = definition.strip()

            if (isValidShortForm(acronym)):
                blf = matchPair(acronym, definition)
                if blf != None: 

                    # --- NOTE: blf is already in lowercase
                    
                    pairs.append([acronym, blf])

            # --- prepare to process the rest of the sentence after ")"
            sentence = sentence[c + 1:]

        elif (o > -1): sentence = sentence[o + 1:] # --- process the rest of the sentence

        acronym = ''
        definition = ''
    
        o = sentence.find(' (')
        if o < 0: return pairs





# --- finds the best match for an acronym and checks if it looks like a valid long form

def matchPair(acronym, definition):
    
    # --- abort if acronym too short
    if (len(acronym) < 2): return None
    
    # --- find the long form
    blf = bestLongForm(acronym, definition)

    # --- abort if no long form found
    if (blf == None): return None

    # --- t = the number of tokens in the long form
    t = len(blf.replace('-', ' ').split(' '))
    
    # --- c = the number of alphanumeric characters in the acronym
    c = sum([int(char.isalpha() or char.isdigit()) for char in acronym])

    # --- case-insensitive matching; NOTE: blf is already in lowercase
    acronym = acronym.lower().replace(' ', '')
    
    # --- sanity check
    if len(blf) < 8:                           # --- long form too short
        return None
    elif len(blf) <= len(acronym):             # --- long form < short form
        return None
    elif blf.startswith(acronym + ' '):        # --- acronym nested in the long form
        return None
    elif blf.find(' ' + acronym + ' ') > -1:   # --- acronym nested in the long form
        return None
    elif blf.endswith(' ' + acronym):          # --- acronym nested in the long form
        return None
    elif acronym[0:1] != blf[0:1]:             # --- they don't start with the same letter
        return None
    elif t > 2*c or t > c+5:                   # --- too many tokens in the long form
        return None
    elif blf.find('[') >= 0 or blf.find(']') >= 0:
        return None
    else:                                       # --- no match in the last two tokens
        tokens = blf.split()
        if len(tokens) > 2:
            last2 = " ".join(tokens[-2:])
            if last2 == last2.replace(acronym[-1], ""):
                return None

        # --- delete all other letters from the definition: a token with no match will disappear
        remainder = re.sub("[^ "+acronym.replace('-', '')+"]", "", blf)
        tokens = len(remainder.split())
        #if len(acronym) - tokens >= 2:          # --- at least two unmatched tokens
        if len(blf.split()) - tokens >= 2:       # --- at least two unmatched tokens
            return None
           
    return blf


# # --- explicit acronym recognition




# --- compare two acronym definitions and return the preferred one

def preferred(acronym, definition1, definition2):
    # --- lemmatise and lowercase both definitions
    def1 = " ".join([token.lemma_.lower() for token in nlp(definition1.replace('-', ' '))])
    def2 = " ".join([token.lemma_.lower() for token in nlp(definition2.replace('-', ' '))])
    
    def1_def2 = pad(def1)
    for token in def2.split(): def1_def2 = re.sub(pad(token), " ", def1_def2)
    def1_def2 = def1_def2.strip()
    
    def2_def1 = pad(def2)
    for token in def1.split(): def2_def1 = re.sub(pad(token), " ", def2_def1)
    def2_def1 = def2_def1.strip()

    if def1_def2 == "":                                     # --- nuclear factor kappa B vs nuclear REGULATORY factor kappa B
        if len(acronym) == len(def2.split()):               # --- prefer potential initialism
            return definition2
        else: 
            return definition1
    elif def2_def1 == "":                                   # --- nuclear REGULATORY factor kappa B vs nuclear factor kappa B
        if len(acronym) == len(def1.split()):               # --- prefer potential initialism
            return definition1
        else: 
            return definition2
    elif bestLongForm(def1_def2, def2_def1) == def2_def1:   # --- GC receptor vs glucocorticoid receptor
        return definition2
    elif bestLongForm(def2_def1, def1_def2) == def1_def2:   # --- glucocorticoid receptor vs GC receptor
        return definition1
    else:
        sim = jellyfish.jaro_winkler_similarity(def1, def2)
        if sim < 0.7:                                       # --- ambiguous acronym
            return 'xxx'
        elif len(def1_def2) < len(def2_def1):               # --- keep the shorter one
            return definition1
        else:
            return definition2





def explicit_acronyms():
    ###
    cur1.execute("DELETE FROM term_acronym;")
    ###

    dictionary = {} # --- create a JSON dictionary of short/long forms

    # --- extract sentences that contain a pair of parentheses, e.g.
    #     ... blah blah ( blah blah ) blah blah ...

    cur1.execute("SELECT sentence FROM data_sentence WHERE tags LIKE '%-LRB- % -RRB-%';")
    rows1 = cur1.fetchall()
    for row1 in rows1:
        sentence = row1[0]

        # --- extract all acronym definitions
        pairs = extractPairs(sentence)
    
        for i in range(len(pairs)):
            # --- parse definition by spacy so that it is comparable to previously extracted MWT candidates
            definition = nlp(pairs[i][1])
            # --- store definition to the dictionary
            acronym = pairs[i][0]
            value = " ".join([token.text for token in definition])
            cur2.execute("INSERT INTO tmp_acronym(acronym, phrase) VALUES(?,?);", (acronym, value)) # --- for debugging
            if acronym in dictionary.keys():
                dictionary[acronym] = preferred(acronym, value, dictionary[acronym])
            else:
                dictionary[acronym] = value

    # --- print dictionary to log
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dictionary)

    # --- store acronyms as MWT candidates
    for key in dictionary.keys():
        phrase = dictionary[key]
        if phrase != 'xxx': # --- ignore ambiguous acronyms
            cur1.execute("""INSERT INTO term_acronym(acronym, phrase, normalised)
                            SELECT DISTINCT ?, ?, normalised
                            FROM   term_phrase
                            WHERE  LOWER(?) = LOWER(phrase);""", (key, phrase, phrase))
    return


# # --- implicit acronym recognition




# --- assumptions: 
#     (1) acronyms are frequently used
#     (2) expanded form also used in the corpus, but
#        these two are probably not linked explicitly
#        e.g. blah ACL blah blah ACL blah blah anterior cruciate ligament blah blah 
#                  ~~~           ~~~           ~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- find tokens that are potential acronyms:
#     (1) must contain an UPPERCASE letter, but no lowercase letters
#     (2) must not start with - (avoids e.g. -LRB-)
#     (3) must not end with . (avoids MR. so and so)
#     (4) has to be at least 3 characters long as shorter ones are 
#         likely to introduce false positive expanded forms as they 
#         are more likely to match a random phrase as an expanded form 
#         candidate
#     (5) acronyms are frequently used, so a threshold is set to >MIN times

def implicit_acronyms():
    cur1.execute("DELETE FROM tmp_acronym;")
    cur1.execute("DELETE FROM term_acronym;")

    # --- find tokens that look like acronyms
    cur1.execute("""SELECT token, COUNT(*)
                    FROM   data_token
                    WHERE  UPPER(token) = token
                    AND    LENGTH(token) < 6
                    AND    token GLOB '[A-Z][A-Z]*[A-Z]'
                    GROUP BY token
                    HAVING COUNT(*) > ?;""", (Amin,))
    rows1 = cur1.fetchall()
    for row1 in rows1:
        acronym = row1[0]
        length  = len(acronym)
        pattern = ""
    
        # --- create a LIKE pattern to retrieve matching phrases
        for i in range (0, length): pattern += acronym[i] + "% "
        pattern = pattern.strip()

        # --- extract potential expanded forms
        cur2.execute("""INSERT INTO tmp_acronym(acronym, normalised)
                        SELECT DISTINCT ?, normalised
                        FROM   term_phrase
                        WHERE  phrase LIKE ?
                        AND    LENGTH(phrase) - LENGTH(REPLACE(phrase, ' ', '')) = ? - 1;""", (acronym, pattern, length))

    # --- check number of senses
    cur1.execute("SELECT acronym, COUNT(*) FROM tmp_acronym GROUP BY acronym;")
    rows1 = cur1.fetchall()
    for row1 in rows1:
        acronym = row1[0]
        senses = row1[1]
        
        cosine = {} # --- calculate cosine similarity based on verbs that co-occur in the same sentence

        # --- sqrt(sum(verb:count^2)) for the acronym
        cur2.execute("DELETE FROM v1;")
        cur2.execute("""INSERT INTO v1(lemma, value)
                        SELECT lemma, COUNT(*)
                        FROM   data_token
                        WHERE  sentence_id IN (SELECT sentence_id FROM data_token WHERE token=?)
                        AND    gtag = 'VB' AND lemma NOT IN ('be', 'have', 'do') GROUP BY lemma;""", (acronym,))
        cur2.execute("SELECT SUM(value*value) FROM v1;")
        norm1 = cur2.fetchone()[0]
        if norm1 != None: norm1 = math.sqrt(norm1)
        else:             norm1 = 0
        
        if norm1 > 0:
            # --- for each sense
            cur2.execute("SELECT normalised FROM tmp_acronym WHERE acronym = ?;", (acronym,))
            rows2 = cur2.fetchall()
            for row2 in rows2:
                
                # --- get sense
                normalised = row2[0]
                
                # --- frequency of occurrence
                cur3.execute("SELECT COUNT(*) FROM term_phrase WHERE normalised = ?;", (normalised,))
                f = cur3.fetchone()[0]
            
                if f > 1: # --- ignore single occurrences (outliers)
                    
                    # --- sqrt(sum(verb:count^2)) for the sense
                    cur3.execute("DELETE FROM v2;")
                    cur3.execute("""INSERT INTO v2(lemma, value)
                                    SELECT lemma, COUNT(*)
                                    FROM   data_token
                                    WHERE  sentence_id IN (SELECT sentence_id FROM term_phrase WHERE normalised = ?)
                                    AND    gtag = 'VB' AND lemma NOT IN ('be', 'have', 'do') GROUP BY lemma;""", (normalised,))
                    cur3.execute("SELECT SUM(value*value) FROM v2;")
                    norm2 = cur3.fetchone()[0]
                    if norm2 != None: norm2 = math.sqrt(norm2)
                    else:             norm2 = 0
                    
                    if norm2 > 0:
                        # --- scalar product: sum(verb:count_acronym*verb:count_sense)
                        cur3.execute("""SELECT SUM(v1.value * v2.value)
                                        FROM   v1, v2
                                        WHERE  v1.lemma = v2.lemma;""")
                        product = cur3.fetchone()[0]
                        if product != None: cosine[normalised] = product / (norm1*norm2)
                        else:               cosine[normalised] = 0

        if cosine:
            # --- find the most similar sense
            normalised = max(cosine, key=lambda k: cosine[k])
            similarity = max(cosine.values())
        
            # --- find the most common phrase for the given normalised form
            cur2.execute("""SELECT LOWER(phrase), COUNT(*) AS C
                            FROM   term_phrase
                            WHERE  normalised = ?
                            ORDER BY C DESC;""", (normalised,))
            phrase = cur2.fetchone()[0]

            print(acronym, '\t', phrase, '\t', normalised, '\t', similarity)
        
            # --- store acronym definition
            cur2.execute("""INSERT INTO term_acronym(acronym, phrase, normalised) VALUES(?,?,?);""", (acronym, phrase, normalised))
        
    return


# # --- extract acronyms




if acronyms == "explicit": 
    print("Extracting explicit acronyms...")
    explicit_acronyms()
else:
    print("Extracting implicit acronyms...")
    implicit_acronyms()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Acronyms extracted in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- integrate acronyms




start_time = end_time





# --- expand definitions that contain other acronyms, e.g. 
#     NIK = NF kappa B inducing kinase -> nuclear factor kappa B inducing kinase
cur1.execute("DELETE FROM tmp_normalised;")

cur1.execute("""SELECT DISTINCT LOWER(A.acronym), A.normalised, LOWER(P.phrase), P.normalised
                FROM   term_acronym A, term_acronym P
                WHERE  ' ' || P.phrase || ' ' LIKE '% ' || A.acronym || ' %';""")
rows1 = cur1.fetchall()
for row1 in rows1:
    acronym = row1[0].split()
    definition = row1[1].split()
    phrase = row1[2]
    normalised = row1[3].split()
    normalised = np.setdiff1d(normalised, acronym)
    normalised = np.union1d(normalised, definition)
    renormalised = " ".join(np.sort(normalised))

    cur2.execute("INSERT INTO tmp_normalised(changefrom, changeto) VALUES(?, ?);", (phrase, renormalised))

cur1.execute("SELECT changefrom, changeto FROM tmp_normalised;")
rows1 = cur1.fetchall()
for row1 in rows1:
    phrase = row1[0]
    normalised = row1[1]
    cur2.execute("UPDATE term_acronym SET normalised = ? WHERE LOWER(phrase) = ?;", (normalised, phrase))

    
# --- treat acronyms that are NOT already NESTED within multi-word term candidates
#     as stand-alone MWT candidates
# --- insert mentions of such acronyms into the term_phrase table    

cur1.execute("""INSERT INTO term_phrase(id, sentence_id, token_start, token_length, phrase, normalised)
                SELECT sentence_id || '.' || position, sentence_id, position, 1, acronym, normalised
                FROM   data_token T, term_acronym A
                WHERE  T.token = A.acronym
                AND    T.gtag != 'IN'
                EXCEPT
                SELECT T.sentence_id || '.' || T.position, T.sentence_id, T.position, 1, A.acronym, A.normalised
                FROM   data_token T, term_acronym A, term_phrase P
                WHERE  T.token = A.acronym
                AND    T.sentence_id = P.sentence_id
                AND    P.token_start <= T.position
                AND    T.position < P.token_start + P.token_length;""")


        
# --- now replace NESTED mentions of acronyms with their EXPANDED FORMS
cur1.execute("DELETE FROM tmp_normalised;")

cur1.execute("""SELECT DISTINCT LOWER(P.phrase), P.normalised
                FROM   term_acronym A, term_phrase P
                WHERE  P.normalised <> A.normalised
                AND    ' ' || P.phrase || ' ' LIKE '% ' || LOWER(A.acronym) || ' %';""")
rows1 = cur1.fetchall()
for row1 in rows1:
    phrase = row1[0]
    normalised = row1[1].split()
    cur2.execute("""SELECT LOWER(acronym) AS acr, normalised
                    FROM   term_acronym
                    WHERE  ' ' || ? || ' ' LIKE '% ' || acr || ' %'
                    ORDER BY LENGTH(acronym) DESC;""", (phrase,))
    rows2 = cur2.fetchall()
    for row2 in rows2:
        acronym = row2[0].split()
        definition = row2[1].split()
        normalised = np.setdiff1d(normalised, acronym)
        normalised = np.union1d(normalised, definition)

    renormalised = " ".join(np.sort(normalised))

    cur2.execute("INSERT INTO tmp_normalised(changefrom, changeto) VALUES(?, ?);", (phrase, renormalised))

cur1.execute("SELECT changefrom, changeto FROM tmp_normalised;")
rows1 = cur1.fetchall()
for row1 in rows1:
    phrase = row1[0]
    normalised = row1[1]
    cur2.execute("UPDATE term_phrase SET normalised = ? WHERE LOWER(phrase) = ?;", (normalised, phrase))

    
# --- update multi-word acronyms, which were previously picked up as MWT candidates
cur1.execute("SELECT LOWER(acronym), normalised FROM term_acronym WHERE acronym LIKE '% %';")
rows1 = cur1.fetchall()
for row1 in rows1:
    acronym = row1[0]
    token_length = len(acronym.split())
    normalised = row1[1]
    cur2.execute("UPDATE term_phrase SET normalised = ? WHERE LOWER(phrase) = ?;", (normalised, acronym))
    
# --- add previously missed MWT candidates
    extra = 0
    cur2.execute("""SELECT COUNT(*)
                    FROM   data_sentence 
                    WHERE  ' ' || sentence || ' ' LIKE '% '|| ? ||' %';""", (acronym,))
    extra += cur2.fetchone()[0]
    
    cur2.execute("""SELECT COUNT(*)
                    FROM   term_phrase
                    WHERE  ' ' || phrase || ' ' LIKE '% '|| ? ||' %';""", (acronym,))
    extra -= cur2.fetchone()[0]

    while extra > 0:
        cur2.execute("""INSERT INTO term_phrase(id, sentence_id, token_start, token_length, phrase, normalised)
                        VALUES(?,0,0,?,?,?);""", (acronym+'.'+str(extra),token_length,acronym,normalised))
        extra -= 1

con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Acronyms integrated in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- normalise MWT candidates




start_time = end_time





cur1.execute("DELETE FROM tmp_normalised;")

cur1.execute("""INSERT INTO tmp_normalised(changefrom, changeto)
                SELECT DISTINCT P1.normalised, P2.normalised
                FROM   term_phrase P1, term_phrase P2
                WHERE  P1.flat LIKE '%-%'
                AND    REPLACE(LOWER(P1.phrase),'-',' ') = LOWER(P2.phrase);""")

cur1.execute("""SELECT DISTINCT changefrom, changeto FROM tmp_normalised;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    changefrom = row1[0]
    changeto   = row1[1]
    print(changefrom, "-->", changeto)
    cur2.execute("UPDATE term_phrase SET normalised = ? WHERE normalised = ?;", (changeto, changefrom))

cur1.execute("DELETE FROM tmp_normalised;")

cur1.execute("""INSERT INTO tmp_normalised(changeto, changefrom)
                SELECT DISTINCT P1.normalised, P2.normalised
                FROM   term_phrase P1, term_phrase P2
                WHERE  P1.flat LIKE '%-%'
                AND    REPLACE(LOWER(P1.phrase),'-','') = LOWER(P2.phrase)
                AND    REPLACE(P1.normalised,'-','') = P2.normalised;""")

cur1.execute("""SELECT DISTINCT changefrom, changeto FROM tmp_normalised;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    changefrom = row1[0]
    changeto   = row1[1]
    print(changefrom, "-->", changeto)
    cur2.execute("UPDATE term_phrase SET normalised = ? WHERE normalised = ?;", (changeto, changefrom))
    
con.commit()





cur1.execute("DELETE FROM term_normalised;")
cur1.execute("DELETE FROM term_bag;")
cur1.execute("DELETE FROM token;")
cur1.execute("DELETE FROM token_similarity;")

# --- select normalised MWT candidates
cur1.execute("""INSERT INTO term_normalised(normalised)
                SELECT normalised FROM (
                SELECT normalised, COUNT(*) AS t
                FROM   term_phrase
                WHERE  LENGTH(normalised) > 5
                AND    normalised GLOB '[a-z0-9]*'
                AND    normalised LIKE '% %'
                GROUP BY normalised
                HAVING t > 1);""")

# --- tokenise normalised MWT candidates
cur1.execute("SELECT rowid, normalised FROM term_normalised;")
rows1 = cur1.fetchall()
for row1 in rows1:
    id = row1[0]
    normalised = row1[1]
    
    # --- tokenise normalised form
    tokens = normalised.split()
    
    # --- store tokens as a bag of words
    for token in tokens:
        cur2.execute("INSERT INTO term_bag(id, token) VALUES(?,?);", (id, token))

    cur2.execute("UPDATE term_normalised SET len = ? WHERE rowid = ?;", (len(tokens), id))
        
con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Term candidates re-normalised in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- normalise tokens




start_time = end_time





# --- extract vocabulary of MWT candidates, i.e. select distinct tokens
cur1.execute("INSERT INTO token(token) SELECT DISTINCT token FROM term_bag;")

# --- index tokens for faster retrieval
cur1.execute("CREATE INDEX idx05 ON token(token);")

# --- compare tokens so that similar ones can be normalised
# --- NOTE: for efficiency, only tokens of similar length that start with 
#           the same letter or potential ligature (ae, oe) are compared
cur1.execute("""SELECT T1.token AS t1, T2.token AS t2
                FROM   token T1, token T2
                WHERE  t1 < t2
                AND    (SUBSTR(t1,1,1) = SUBSTR(t2,1,1) OR (SUBSTR(t1,1,1) = 'e' AND SUBSTR(t2,1,1) IN ('a', 'o')))
                AND    ABS(LENGTH(t1) - LENGTH(t2)) < 2;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    t1 = row1[0]
    t2 = row1[1]
    if not any(c.isdigit() for c in t1+t2): # --- ignore tokens that contain digits as these may be significant
        
        # --- calculate token similarity
        sim = jellyfish.jaro_winkler_similarity(t1, t2)
        if sim > Smin: # --- token similarity threshold
            cur2.execute("INSERT INTO token_similarity(token1, token2) VALUES(?,?)",(t1,t2))

# --- A -> B, B -> C, A -> C, then ignore B -> C and use A to normalise both B and C
cur1.execute("""SELECT token1 AS t1, token2 AS t2
                FROM   token_similarity
                EXCEPT
                SELECT S2.token1 AS t1, S2.token2 AS t2
                FROM   token_similarity S1, token_similarity S2
                WHERE  S1.token2 = S2.token1;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    changeto   = row1[0]
    changefrom = row1[1]
    print(changefrom, "\t-->", changeto)
    cur2.execute("UPDATE term_bag SET token = ? WHERE token = ?", (changeto, changefrom))

con.commit()





# --- speed up searching through the bags of words
cur1.execute("CREATE INDEX idx06 ON term_bag(id);")
cur1.execute("CREATE INDEX idx07 ON term_bag(id, token);")
con.commit()





# --- re-normalise the MWT candidates using similar tokens
total = 0
cur1.execute("SELECT MAX(rowid) FROM term_normalised;")
row1 = cur1.fetchone()
total = row1[0]
if total == None: total = 0
for i in range(1, total+1):
    tokens = []
    cur2.execute("SELECT token FROM term_bag WHERE id = ? ORDER BY token;""", (i,))
    rows2 = cur2.fetchall()
    for row2 in rows2: tokens.append(row2[0])
    expanded = " ".join(tokens)
    cur2.execute("UPDATE term_normalised SET expanded = ? WHERE rowid = ?;", (expanded, i))
        
con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Tokens normalised in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- identify nested MWTs




start_time = end_time





# --- speed up searching through the phrases
cur1.execute("CREATE INDEX idx08 ON term_phrase(normalised);")
cur1.execute("CREATE INDEX idx09 ON term_normalised(normalised);")
cur1.execute("CREATE INDEX idx10 ON term_normalised(expanded);")
con.commit()





###
cur1.execute("DELETE FROM term_nested_aux;")
cur1.execute("DELETE FROM term_nested;")
###

# --- select candidate MWT pairs to check for nestedness
for i in range(1, total):

    # --- progress bar
    sys.stdout.write('\r')
    p = int(100*i/total)+1
    sys.stdout.write("[%-100s] %d%%" % ('='*p, p))
    sys.stdout.flush()
    
    cur1.execute("SELECT token FROM term_bag WHERE id = ?;", (i,))
    tokens = ""
    rows1 = cur1.fetchall()
    for row1 in rows1: 
        tokens += "','" + row1[0].replace("'", "''")
    tokens = tokens[2:] + "'"
   
    cur1.execute("SELECT DISTINCT id FROM term_bag WHERE id > ? AND token IN ("+tokens+");", (i,))
    rows1 = cur1.fetchall()
    for row1 in rows1:
        
        j = row1[0]
        
        # --- term_i - term_j = 0 ?
        cur2.execute("""SELECT token FROM term_bag WHERE id = ?
                        EXCEPT
                        SELECT token FROM term_bag WHERE id = ?;""", (i,j))
        row2 = cur2.fetchone()
        if row2 == None: # --- term_i subset of term_j
            cur3.execute("INSERT INTO term_nested_aux(parent, child) VALUES(?,?)", (j,i))
        else:
            # --- term_j - term_i = 0 ?
            cur2.execute("""SELECT token FROM term_bag WHERE id = ?
                            EXCEPT
                            SELECT token FROM term_bag WHERE id = ?;""", (j,i))
            row2 = cur2.fetchone()
            if row2 == None: # --- term_j subset of term_i
                cur3.execute("INSERT INTO term_nested_aux(parent, child) VALUES(?,?)", (i,j))

# --- select unique nested MWT pairs
cur1.execute("""INSERT INTO term_nested(parent, child)
                SELECT DISTINCT N1.expanded, N2.expanded
                FROM   term_normalised N1, term_normalised N2, term_nested_aux A
                WHERE  N1.rowid = A.parent
                AND    N2.rowid = A.child
                AND    N1.expanded <> N2.expanded;""") # --- proper subsets only
con.commit()





cur1.execute("CREATE INDEX idx11 ON term_nested(parent);")
cur1.execute("CREATE INDEX idx12 ON term_nested(child);")
con.commit()


# # --- calculate termhood




# --- C-value (collocation)
def cValue(length, f, s, nf):
    c = f
    if s > 0: c -= nf*1.0 / s
    c = c * math.log(length)
    return c

# --- inverse document frequency (discriminativeness)
def idf(n, df):
    return math.log10(n*1.0 / df)





###
cur1.execute("DELETE FROM term_termhood;")
cur1.execute("DELETE FROM term_output;")
###

cur1.execute("""INSERT INTO term_termhood(expanded, len, s, nf)
                SELECT DISTINCT expanded, len, 0, 0 FROM term_normalised;""")

# --- calculate frequency of standalone occurrence
cur1.execute("""SELECT N.expanded, COUNT(*)
                FROM   term_normalised N, term_phrase P
                WHERE  N.normalised = P.normalised
                GROUP BY N.expanded;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    expanded = row1[0]
    f        = row1[1]
    cur2.execute("UPDATE term_termhood SET f = ? WHERE expanded = ?;", (f, expanded))

# --- calculate the number of parent (superset) MWTs
cur1.execute("SELECT child, COUNT(*) FROM term_nested GROUP BY child;")
rows1 = cur1.fetchall()
for row1 in rows1:
    child = row1[0]
    s     = row1[1]

    cur2.execute("UPDATE term_termhood SET s = ? WHERE expanded = ?;", (s, child))

# --- calculate the frequency of nested occurrence
cur1.execute("""SELECT child, COUNT(*)
                FROM   term_nested N, term_normalised C, term_phrase P
                WHERE  N.parent = C.expanded
                AND    C.normalised = P.normalised
                GROUP BY child;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    child = row1[0]
    nf    = row1[1]

    cur2.execute("UPDATE term_termhood SET nf = ? WHERE expanded = ?;", (nf, child))

# --- add up frequencies (both nested and standalone): f(t)
cur1.execute("UPDATE term_termhood SET f = f + nf;")

# --- calculate C-value
cur1.execute("SELECT expanded, len, f, s, nf FROM term_termhood;")
rows1 = cur1.fetchall()
for row1 in rows1:
    expanded = row1[0]
    length   = row1[1]
    f        = row1[2]
    s        = row1[3]
    nf       = row1[4]
    c        = cValue(length, f, s, nf) # --- NOTE: no ln(x) in sqlite, so have to calculate C-value externally
    
#    if c > 1:
    cur2.execute("UPDATE term_termhood SET c = ? WHERE expanded = ?;", (c, expanded))

# --- store term list
cur1.execute("""INSERT INTO term_output(id, variant, c, f)
                SELECT T.rowid, LOWER(P.phrase) as variant, T.c, COUNT(*)
                FROM   term_termhood T, term_normalised N, term_phrase P
                WHERE  T.expanded = N.expanded
                AND    N.normalised = P.normalised
                AND    T.f > ?
                AND    T.c > ?
                GROUP BY T.rowid, variant, T.c;""", (Fmin, Cmin))

# --- delete outliers: highly ranked terms that have a single variant with frequency of 1
#     (e.g. kappa b), which is ranked highly only because of nested frequency
cur1.execute("""SELECT id FROM term_output O WHERE f <= ?
                AND    1 = (SELECT COUNT(*) FROM term_output I WHERE O.id = I.id);""", (Fmin,))
rows1 = cur1.fetchall()
for row1 in rows1: 
    cur2.execute("DELETE FROM term_output WHERE id = ?;", (row1[0],))

# --- n = total number of documents (to calculate IDF later on)
cur1.execute("SELECT COUNT(*) FROM data_document;")
n = cur1.fetchone()[0]
con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Termhood calculated in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- find term occurrences in text




start_time = end_time





cur1.execute("DELETE FROM output_label;")

common = ['all', 
          'on', 
          'in', 
          'at', 
          'to', 
          'by', 
          'of', 
          'off', 
          'so', 
          'or', 
          'as', 
          'and', 
          'ie',
          'eg',
          'dr',
          'mr',
          'mrs',
          'ms',
          'km',
          'mm',
          'old',
          'no', 
          'not', 
          'pre',
          'be', 
          'is', 
          'are', 
          'am', 
          'can', 
          'for', 
          'up',
          'has',
          'had',
          'who']

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

print("Retrieving terms to match...")
cur1.execute("SELECT id, variant FROM term_output;")
rows1 = cur1.fetchall()
total = len(rows1)
i = 0
for row1 in rows1:

    matcher.add(str(row1[0]), None, nlp(row1[1]))
    
    # --- progress bar
    i += 1
    sys.stdout.write('\r')
    p = int(100*i/total)
    sys.stdout.write("[%-100s] %d%%" % ('='*p, p))
    sys.stdout.flush()

print("\nLooking up terms in documents...")
cur1.execute("SELECT id, document FROM data_document;")
rows1 = cur1.fetchall()
total = len(rows1)
i = 0
for row1 in rows1:
    doc_id= row1[0]
    doc = nlp(hyphen(row1[1]))
    matches = matcher(doc)

    # --- progress bar
    i += 1
    sys.stdout.write('\r')
    p = int(100*i/total)
    sys.stdout.write("[%-100s] %d%%" % ('='*p, p))
    sys.stdout.flush()
    
    for match_id, start, end in matches:
        term_id = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        o = len(span)
        s = len(doc[0:end].text) - o
        if (span.lower() not in common) or span.upper() == span: # --- making sure that short acronyms such as OR are uppercased to avoid FPs
            cur2.execute("INSERT INTO output_label(doc_id, start, offset, label) VALUES (?,?,?,?);", (doc_id, s, o, term_id))

# --- update document frequency
cur1.execute("""SELECT label, COUNT(DISTINCT doc_id) as df
                FROM   output_label
                GROUP BY label;""")
rows1 = cur1.fetchall()
for row1 in rows1:
    label = row1[0]
    df    = row1[1]
    cur2.execute("UPDATE term_output SET df=?, idf=? WHERE id = ?;", (df, idf(n,df), label))

# --- delete nested labels
cur1.execute("""DELETE FROM output_label WHERE rowid IN (
                SELECT T2.rowid FROM output_label T1, output_label T2
                WHERE  T1.doc_id = T2.doc_id
                AND    T1.start <= T2.start
                AND    T2.start + T2.offset <= T1.start + T1.offset
                AND    (T1.start != T2.start OR T2.start + T2.offset != T1.start + T1.offset));""")

# --- delete overlapping labels
cur1.execute("""DELETE FROM output_label WHERE rowid IN (
                SELECT T2.rowid FROM output_label T1, output_label T2
                WHERE  T1.doc_id = T2.doc_id
                AND    T2.start <= T1.start + T1.offset
                AND    T1.start + T1.offset <= T2.start + T2.offset
                AND    (T1.start != T2.start OR T2.start + T2.offset != T1.start + T1.offset));""")

# --- delete terms that have no occurrences (it may happen 
#     when they are nested in a term, which was mistagged)
cur1.execute("DELETE FROM term_output WHERE id NOT IN (SELECT label FROM output_label);")

con.commit()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Term occurrences annotated in {run_time:0.4f} seconds")
timer.append(run_time)





cur1.execute("CREATE INDEX idx13 ON term_output(id);")
cur1.execute("CREATE INDEX idx14 ON term_output(c, id, f);")
cur1.execute("CREATE INDEX idx15 ON output_label(doc_id);")
cur1.execute("CREATE INDEX idx16 ON output_label(label);")
cur1.execute("CREATE INDEX idx17 ON output_label(label, doc_id);")
con.commit()


# # --- annotate term occurrences in text




# --- color scaling
def transition(value, maximum, start_point, end_point):
    return start_point + (end_point - start_point)*value/maximum

def transition3(value, maximum):
    r1= transition(value, maximum, 37, 211)
    r2= transition(value, maximum, 150, 234)
    r3= transition(value, maximum, 190, 242)
    return "#%02x%02x%02x" % (int(r1), int(r2), int(r3))

# --- random color picking
def color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('9ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    return color

# --- top C-value score
cur1.execute("SELECT MAX(c) FROM term_output;")
top = cur1.fetchone()[0]

random_colors = True

# --- asign colors to terms
entities = []
colors = {"ENT":"#E8DAEF"}

cur1.execute("SELECT DISTINCT id, c FROM term_output ORDER BY c DESC;")
rows1 = cur1.fetchall()
for row1 in rows1:
    id = row1[0]
    c  = row1[1]
    entities.append(str(id))
    color = transition3(top-c, top)
    colors[str(id)] = color

if random_colors:
    color = color_generator(len(entities))
    for i in range(len(entities)): colors[entities[i]] = color[i]

# --- coloring options for spacy's PhraseMatcher
options = {"ents": entities, "colors": colors}

# --- spacy-formatted entity annotations
annotations = []

# --- for each document
cur1.execute("SELECT id, document FROM data_document;")
rows1 = cur1.fetchall()
for row1 in rows1:
    doc_id = row1[0]
    doc = row1[1]

    # --- retrieve previously stored PhraseMatcher labels
    ents = []
    cur2.execute("SELECT start, offset, label FROM output_label WHERE doc_id = ?;", (doc_id,))
    rows2 = cur2.fetchall()
    for row2 in rows2:
        start = row2[0]
        end = row2[0] + row2[1]
        label = str(row2[2])
        ents.append({"start": start, "end": end, "label": label})

    annotations.append({"text": doc, "ents": ents, "title": doc_id, "settings": {}})

# --- export spacy-formatted entity annotations
with open(Path("./out/annotations.json"), "w") as file:
    json.dump(annotations, file, indent=4)
    file.close()

# --- visualise annotations
html = displacy.render(annotations, style="ent", manual=True, options=options, page=True, jupyter=False)
html = re.sub('>([^<]+)</h2>', ' id="D\\1">\\1</h2>', html, flags=re.IGNORECASE)

# --- export HTML visualisation/annotation
with open(Path("./out/corpus.html"), "w", encoding="utf8") as file: 
    file.write(html)
    file.close()





def header(title):
    return """<!DOCTYPE html>
<html lang="en">
<head>
<title>""" + title + """</title>
<style></style>
</head>
<body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">"""


# # --- extract concordances




start_time = end_time





def concordance(id, doc_id, left, term, right): 
    return """
    <tr>
        <td><a href='corpus.html#D"""+ doc_id +"""' target='_blank'>""" + doc_id + """</a></td>
        <td style='text-align:right'>""" + left + """</td>
        <td style='text-align:center;width:1px;white-space:nowrap;'>
        <mark class="entity" style="background: """ + colors[str(id)] + """; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">""" + term + """</mark>
        </td>
        <td>""" + right  + """</td>
    </tr>"""

# --- start an HTML document
output = header("Concordances")
output += "<h1>Term concordances</h1>"

cur1.execute("SELECT DISTINCT id FROM term_output ORDER BY c DESC;")
rows1 = cur1.fetchall()
for row1 in rows1:
    id = row1[0]

    output += "\n<br/><br/>\n<h2 style='margin:0' id='T" + str(id) + "'>Term ID: <a href='terminology.html#L"+ str(id) +"' target='_blank'>"+ str(id) +"</a></h2><br/><table border='0'>"
    
    cur2.execute("""SELECT doc_id,
                           SUBSTR(D.document, MAX(start+1-80, 1), MIN(start, 80)), 
                           SUBSTR(D.document, start+1, offset),
                           SUBSTR(D.document, start+1+offset, 80)
                    FROM   output_label L, data_document D
                    WHERE  label = ?
                    AND    L.doc_id = D.id
                    ORDER BY doc_id, start;""", (id,))
    rows2 = cur2.fetchall()
    for row2 in rows2: output += concordance(id, row2[0], row2[1], row2[2], row2[3])
    
    # --- close the table
    output += "\n</table>"

# --- end the HTML document
output += "\n</html>"

# --- write HTML to file
with open(Path("./out/concordances.html"), "w", encoding="utf8") as file: 
    file.write(output)
    file.close()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Concordances extracted in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- export terminology (lexicon)




start_time = end_time





# --- export terminology into a CSV file

cur1.execute("""SELECT id, variant, c, f, df, ROUND(c*idf, 3) AS c_idf
                FROM   term_output
                ORDER BY c DESC, id ASC, f DESC;""")
rows1 = cur1.fetchall()
   
with open(Path("./out/terminology.csv"), "w", encoding="utf8") as file:
    csv_writer = csv.writer(file, delimiter="\t")
    csv_writer.writerow([i[0] for i in cur1.description])
    csv_writer.writerows(rows1)
    file.close()





# --- export terminology into an HTML file

def firstrow(id, c, variant, f): 
    id = str(id)
    c = str(round(c, 3))
    f = str(f)
    return """
    <tr>
        <td rowspan='xxxxx' style='text-align:center'><a href='concordances.html#T""" + id + """' target='_blank'>""" + id + """</a></td>
        <td rowspan='xxxxx' style='text-align:center'>""" + c  + """</td>
        <td id='L"""+ str(id) +"""' bgcolor='""" + colors[str(id)] + """'>""" + variant + """</td>
        <td style='text-align:center'>""" + f + """</td>
    </tr>"""

def nextrow(variant, f): 
    f = str(f)
    return """
    <tr>
        <td bgcolor='""" + colors[str(id)] + """'>""" + variant + """</td>
        <td style='text-align:center'>""" + f + """</td>
    </tr>"""

# --- start an HTML document
output = header("Terminology")
output += """
<h1>Terminology</h1>
<br><br>
<table>
    <tr>
        <th>Term ID</th>
        <th>Termhood</th>
        <th>Term variant</th>
        <th>Term variant frequency</th>
    </tr>"""

cur1.execute("""SELECT id, variant, c, f
                FROM   term_output
                ORDER BY c DESC, id ASC, f DESC;""")
rows1 = cur1.fetchall()
pre = -1     # --- previous term ID
tr = ""      # --- current table row
rowspan = 0  # --- total of variants per ID
for row1 in rows1:
    id = row1[0]
    if id != pre: # --- next term
        output += tr.replace("xxxxx", str(rowspan))
        tr = firstrow(id, row1[2], row1[1], row1[3])
        pre = id
        rowspan = 1
    else:         # --- append to the current term
        tr += nextrow(row1[1], row1[3])
        rowspan += 1

# --- don't forget to add the last term ???
if tr != "": output += tr.replace("xxxxx", str(rowspan))

# --- end the HTML document
output += "\n</table>\n</html>"
output = re.sub('<style></style>', '<style>td, th {border: 1px solid #999; padding: 0.5rem;}</style>', output, flags=re.IGNORECASE)

# --- write HTML to file
with open(Path("./out/terminology.html"), "w", encoding="utf8") as file: 
    file.write(output)
    file.close()





end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Terminology exported in {run_time:0.4f} seconds")
timer.append(run_time)


# # --- close the database




con.close()


# # --- the end




n = 0
i = 0
timer[2] += timer[5]
for t in timer:
    if i != 5: print(f"{t:0.3f}")
    i += 1


