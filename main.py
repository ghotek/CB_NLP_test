import pandas as pd
import streamlit as st
import plotly.graph_objs as go

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import time

from model import CatModel
from CB_NLP_test.data_parsing import open_data, open_fit_data, get_last_data, update_data

SLEEP_TIME = 5 * 24 * 60 * 60

def main_page():
    print('running...')

    model = CatModel()

    st.title('Анализ ключевой ставки по пресс-релизам ЦБ')
    live_graph = st.empty()
    live_pred  = st.empty()
    
    st.divider()
    
    user_text = st.text_input('Введите текст пресс-релиза для анализа дальнейшей ставки')
    user_pred = st.button('Выдать прогноз')

    if (user_pred):
        input_series = pd.Series([user_text])
        
        prediction = model.predict(input_series)[0]
        
        formulation = ''
        if prediction == 1:
            formulation = 'Ставка пойдет наверх'
        elif prediction == 0:
            formulation = 'Ставка сохранится'
        elif prediction == -1:
            formulation = 'Ставка пойдет вниз'
        
        st.write('По данному релизу модель предсказывает, что ' + formulation)

    while True:
        
        with live_graph.container():
            plot_data = open_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data['key_rate']))
            st.plotly_chart(fig, use_container_width=True)

        date, text, url = get_last_data()
        
        input_series = pd.Series([text])
        live_pred_val = model.predict(input_series)[0]

        if live_pred_val == 1:
            formulation = 'Ставка пойдет наверх'
        elif live_pred_val == 0:
            formulation = 'Ставка сохранится'
        elif live_pred_val == -1:
            formulation = 'Ставка пойдет вниз'
        
        with live_pred.container():
            live_pred_title = f'Предсказание ключевой ставки по [последнему пресс-релизу ЦБ]({url}) от {date}'
            st.write(live_pred_title)
            st.write('Данный релиз говорит о том, что ' + formulation)

        if update_data():
            fit_data = open_fit_data()
            model.fit(fit_data['text'], fit_data['target'])
        
        time.sleep(SLEEP_TIME)
            

if __name__ == "__main__":
    main_page()
