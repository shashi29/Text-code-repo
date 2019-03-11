import pandas as pd
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords


train = pd.read_csv("train.csv",sep='~')
test = pd.read_csv("test.csv",sep='~')

d1 = sns.countplot(train['Is_Response'])
d2 = sns.countplot(train['Device_Used'])


train['Device_Used'].value_counts()

browser = ['Firefox','Edge','Google Chrome','InternetExplorer','Mozilla Firefox',
           'Mozilla','IE','Chrome','Internet Explorer','Safari','Opera']
#Device = ['Desktop','Mobile','Tablet']
#c1 = sns.countplot(train['Is_Response'])
#c2 = sns.countplot(train['Device_Used'])
#c3 = sns.countplot(train['Browser_Used'],order=browser)
#c4 = sns.countplot(x='Device_Used',hue='Is_Response',data=train,order=['Desktop','Mobile','Tablet'])
#c5 = sns.countplot(hue='Device_Used',x='Is_Response',data=train)
#c6 = sns.countplot(hue='Device_Used',x='Browser_Used',data=train,order=browser)
#c7 = sns.countplot(x='Device_Used',hue='Browser_Used',data=train,order=Device)

#No missing values
train.isna().sum()

#Unique categories in each variable
train['Device_Used'].unique()
train['Is_Response'].unique()
train['Browser_Used'].value_counts()

#Encode the target varible
stars_dict = {'Good':1,'Bad':0}
train["Is_Response"] = train['Is_Response'].replace(stars_dict,regex=True)


#Pre-processing of the doc
stop_words = stopwords.words('english')
remove_words = ['aaa']

def preprocess_text(text):
    text = text.str.replace("[^a-zA-Z]", " ")
    text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    text = [remove_stopwords(r.split()) for r in text]
    text = [r.lower() for r in text]
    return text

def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

train['Description'] = preprocess_text(train['Description'])

#adding additional information about the data
train['word_count'] = train['Description'].apply(lambda x: len(str(x).split(" ")))
train['char_count'] = train['Description'].str.len() ## this also includes spaces

freq = pd.Series(' '.join(train['Description']).split()).value_counts()[-18000:]
freq = pd.DataFrame(freq).reset_index()
#d1 = sns.barplot(x=freq.index,y=freq['count'],data=freq)
#Remove most common word
Most_common_words = ['the','hotel','room']
train['Description'] = train['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in Most_common_words))
train['Description'] = train['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

from textblob import TextBlob
train['Description'] = train['Description'].apply(lambda x: str(TextBlob(x).correct()))

#Lemmitization 
#Convert word into its root format
from textblob import Word
train['Description'] = train['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
