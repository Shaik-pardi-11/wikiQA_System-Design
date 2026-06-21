import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep='\t')
    print(df.head())
    return df

def clean_data(df):
    df = df[['Question', 'Sentence', 'Label']].dropna()
    df['Question'] = df['Question'].str.lower()
    df['Sentence'] = df['Sentence'].str.lower()
    return df