import pandas as pd
import gc
import string
import numpy as np
from utils import clean, remove_URL, remove_emoji, remove_html
from nltk.corpus import stopwords 
stopWords = set(stopwords.words('english'))

# Read data
df_train = pd.read_csv('data/train.csv', dtype={'id': np.int16, 'target': np.int8})
traindex = df_train.index

df_test = pd.read_csv('data/test.csv', dtype={'id': np.int16})
testdex = df_test.index

# concat train, test
y = df_train.target.values
df_train.drop(['target'],axis=1, inplace=True)
df = pd.concat([df_train, df_test],axis=0)

# to_drop = ['id', 'location']
# df.drop(to_drop, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
del df_train, df_test
gc.collect()

# Meta features
def feature_count(df, feat_name, cond):
    df = df.copy()
    # df.text = df.text.str.lower().str.split()
    df[feat_name] = df.text.apply(lambda t: len([w for w in str(t).lower().split() if w in cond]))
    return df

def feature_coincide(df, feat_name, cond):
    df = df.copy()
    df[feat_name] = df.text.apply(lambda t: len([c for c in str(t) if c == cond]))
    return df


def clean_text(df):
    df = df.copy()
    df['text'] = df.text.apply(clean)
    return df


print("Start preprocessing") 
df = (df
      .assign(word_count = lambda x: x.text.str.split().apply(len))
      .assign(char_count = lambda x: x.text.apply(len))
      .pipe(feature_count,
            'stop_word_count',
            stopWords)
      .pipe(feature_count,
            'punctuation_count',
            string.punctuation)
      .pipe(feature_coincide,
            'mention_count',
            "@")
      .pipe(feature_coincide,
            "hashtag_count",
            "#")
      .assign(keyword=lambda x: x.keyword.fillna('no_keyword'))
      .assign(location=lambda x: x.location.fillna('no_location'))
      .assign(text = lambda x: x.text.apply(remove_URL))
      .assign(text = lambda x: x.text.apply(remove_html))
      .assign(text = lambda x: x.text.apply(remove_emoji))
      # .pipe(clean_text)
    )

print("Finished preproc")

train_df = df.loc[traindex,:].reset_index(drop=True)
train_df['target'] = y
train_df.to_csv("data/prepared_df_train.csv", index=False)

df.loc[testdex,:].reset_index(drop=True).to_csv("data/prepared_df_test.csv", index=False)

print("Start train_val split")
from Dataset import train_val_split
train_val_split()

