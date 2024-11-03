import pandas as pd

def get_data():
    df = pd.read_csv('./data/Corona_NLP_train.csv', encoding='latin-1')
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: x.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore'))
    return df[['OriginalTweet', 'Sentiment']]
