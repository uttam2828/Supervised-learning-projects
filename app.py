

import streamlit as st 
import pandas as pd 
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote
user_name = 'uttam'
database = 'project'
your_password = 'password'
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))



label = joblib.load("label_salary")
preprocess = joblib.load("Data26")
model = joblib.load("Data27")
outlier = joblib.load("winsor")


def nb_model(data):
    
    #data = pd.read_csv(data)
    df = data.drop_duplicates()
    columns = ['num__age', 'num__educationno', 'num__hoursperweek']
    df1 = pd.DataFrame(preprocess.transform(df), columns=preprocess.get_feature_names_out())
    df1[columns] = outlier.transform(df1[columns])
    predict = pd.DataFrame(model.predict(df1), columns = ['Predictions'])
    predict['Predictions'] = label.inverse_transform(predict['Predictions'])
    final = pd.concat([data, predict], axis=1)
    final.to_sql('prediction_Salary', con = engine, if_exists = 'replace', index = False)
    return final

def main():
    st.title("Fire area prediction")
    st.sidebar.title("Fire area prediction")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Fire area prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    uploadeddata = st.sidebar.file_uploader("Choose a data", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadeddata is not None :
        try:

            data = pd.read_csv(uploadeddata)
        except:
                try:
                    data = pd.read_excel(uploadeddata)
                except:      
                    data = pd.DataFrame()
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel data.")
    
    result = ""
    st.text('')
    if st.button("Predict"):
        result = nb_model(data)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()

    
    
    