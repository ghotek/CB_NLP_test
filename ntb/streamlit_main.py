import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

URL = 'https://www.cbr.ru/press/keypr/'
data = pd.read_csv('target.csv')

class CatModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(min_df=0.3, max_df=0.8, ngram_range=(3, 3))
        self.cat_model = CatBoostClassifier()

    @staticmethod
    def clean_text(text):
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from pymorphy3 import MorphAnalyzer

        morph = MorphAnalyzer(lang='ru')
        stop_words = set(stopwords.words('russian'))
        stop_words.update(
            ['совет', 'директоры', 'банк', 'россия', 'другой', 'это',
             'ключевой ставка', 'директор'])
        text = re.sub('[^а-яёА-ЯЁ]', ' ', text)  # оставляем только кириллицу
        text = word_tokenize(
            text.lower(),
            language='russian')  # приводим к нижнему регистру и токенизируем по словам
        # приводим токены к нормальной форме, удаляем стоп-слова и короткие токены

        text_cleaned = []
        for token in text:
            token = token.lower().strip()
            t_clean = morph.normal_forms(token)[0]
            if len(t_clean) > 2 and (t_clean not in stop_words):
                text_cleaned.append(t_clean)
        text = " ".join(text_cleaned)  # возвращаем строку

        return text

    def fit(self, x: pd.Series, y: pd.Series) -> None:
        import pickle

        x = x.apply(self.clean_text)
        self.tfidf: TfidfVectorizer = self.tfidf.fit(x)
        x = self.tfidf.transform(x)
        self.cat_model = self.cat_model.fit(x, y)

        pickle.dump(self.tfidf, open('tfidf.pkl', 'wb'))
        pickle.dump(self.cat_model, open('model.pkl', 'wb'))

    def predict(self, x):
        import pickle

        x = x.apply(self.clean_text)
        self.tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        x = self.tfidf.transform(x)
        self.cat_model = pickle.load(open('model.pkl', 'rb'))

        return self.cat_model.predict(x)

def getTextDate():
    import requests
    import re
    from bs4 import BeautifulSoup

    responce = requests.get(URL)
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

    return date, text_str


if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    
    import time

    model = CatModel()

    # заголовок приложения
    st.title('Анализ ключевой ставки по пресс-релизам ЦБ')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['Ставка']))
    st.plotly_chart(fig, use_container_width=True)

    user_text = st.text_input('Введите текст пресс-релиза для анализа дальнейшей ставки')

    # проверяем нажата кнопка или нет
    if (st.button('Выдать прогноз')):
        input_series = pd.Series([user_text])
        prediction = model.predict(input_series)[0]
        print(prediction)
        formulation = ''
        st.write(prediction)
        if prediction == 1:
            formulation = 'Ставка пойдет наверх'
        elif prediction == 0:
            formulation = 'Ставка сохранится'
        elif prediction == -1:
            formulation = 'Ставка пойдет вниз'
        st.write('Данный релиз говорит о том, что ' + formulation)

    live_pred_place = st.empty()
    while True:
        date, text = getTextDate()
        
        input_series = pd.Series([text])
        live_pred = model.predict(input_series)[0]

        if live_pred == 1:
            formulation = 'Ставка пойдет наверх'
        elif live_pred == 0:
            formulation = 'Ставка сохранится'
        elif live_pred == -1:
            formulation = 'Ставка пойдет вниз'
        
        with live_pred_place.container():
            live_pred_title = 'Предсказание ключевой ставки по [последнему пресс-релизу ЦБ](' + URL + ') от ' + date
            st.write(live_pred_title)
            st.write(live_pred)
            st.write('Данный релиз говорит о том, что ' + formulation)

            time.sleep(100)



