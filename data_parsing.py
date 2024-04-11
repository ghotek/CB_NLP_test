import pandas as pd

import requests
import re
from bs4 import BeautifulSoup

from datetime import datetime
from dateutil import parser

def open_data(path='data/dataset.csv'):
    data = pd.read_csv(path)
    return data

def open_fit_data(path='data/dataset.csv'):
    data = pd.read_csv(path)
    return data.dropna()

def get_last_data(url='https://www.cbr.ru/press/keypr/'):
    responce = requests.get(url)
    tree = BeautifulSoup(responce.text, 'html.parser')
    
    # Date of last release
    date = tree.find_all('div', {'class': 'col-md-6 col-12 news-info-line_date'})
    date = date[0].text

    # Text of last release
    tree_text = tree.get_text()
    tree_text.replace('\n','').replace('\t','').replace('\xa0',' ').replace('\r',' ').replace('  ','')
    text_str = tree.find_all('div', attrs={'class': 'landing-text'})[0].get_text(separator=' ')
    text_str = text_str.replace('\n','').replace('\t','').replace('\xa0',' ').replace('\r',' ').strip()
    text_str = ''.join([i for i in text_str if not i.isdigit()])
    text_str = re.sub(r'[^\w\s]', '', text_str)

    return date, text_str, url

def update_data(path='data/dataset.csv', url='https://www.cbr.ru/press/keypr/'):
    df1 = open_data()

    # Date of last release with valid format
    responce = requests.get(url)
    tree = BeautifulSoup(responce.text, 'html.parser')
    actual_release_date = tree.find_all('meta', {'name': 'zoom:last-modified'})

    # last_release_date - date of last release in csv table
    # actual_release_date - date of actual release from url

    last_release_date = df1.loc[0, 'date']
    actual_release_date = actual_release_date[0]['content']
    
    dt_last_release_date = datetime.strptime(last_release_date, '%Y-%m-%d')
    dt_actual_release_date = parser.parse(actual_release_date)
    dt_actual_release_date = datetime(dt_actual_release_date.year,
                                    dt_actual_release_date.month,
                                    dt_actual_release_date.day)
    
    if dt_last_release_date == dt_actual_release_date:
        return 0
    
    # Find key rate and target of last release of csv table
    actual_key_rate = tree.find_all('title')[0].text
    actual_key_rate = ''.join([c if c.isdigit() else ' ' for c in actual_key_rate]).split()
    actual_key_rate = float('.'.join(actual_key_rate))

    df1.loc[0, 'future_key_rate'] = actual_key_rate
    if df1.loc[0, 'future_key_rate'] > df1.loc[0, 'key_rate']:
        df1.loc[0, 'target'] = 1
    if df1.loc[0, 'future_key_rate'] == df1.loc[0, 'key_rate']:
        df1.loc[0, 'target'] = 0
    if df1.loc[0, 'future_key_rate'] < df1.loc[0, 'key_rate']:
        df1.loc[0, 'target'] = -1

    # Fill new data DataFrame
    _, new_text, _ = get_last_data()

    df2 = pd.DataFrame(columns=df1.columns)
    
    df2.loc[0, 'text'] = new_text
    df2.loc[0, 'date'] = datetime.strftime(dt_actual_release_date, '%Y-%m-%d')
    df2.loc[0, 'key_rate'] = actual_key_rate
    
    df = pd.concat([df2, df1], ignore_index=True)
    df.to_csv(path, index=False)
    
    return 1


if __name__ == "__main__":
    update_data()