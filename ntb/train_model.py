import pandas as pd

from streamlit_main import CatModel

data = pd.read_csv('target.csv')
data = data.dropna()
Cat_Model = CatModel()
Cat_Model.fit(data['text'], data['target'])
