#!/usr/bin/env python3
import itertools
import joblib
import matplotlib
import time
import io

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def main():
  matplotlib.use('Agg')
  
  with open('data/hungarian.data', encoding='Latin1') as stream:
    lines = [line.strip() for line in stream]

  data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
  )

  df = pd.DataFrame.from_records(data)

  df = df.iloc[:, :-1]
  df = df.drop(df.columns[0], axis=1)
  df = df.astype(float)

  df.replace(-9.0, np.NaN, inplace=True)

  df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

  column_mapping = {
    2: 'age',
    3: 'sex',
    8: 'cp',
    9: 'trestbps',
    11: 'chol',
    15: 'fbs',
    18: 'restecg',
    31: 'thalach',
    37: 'exang',
    39: 'oldpeak',
    40: 'slope',
    43: 'ca',
    50: 'thal',
    57: 'target'
  }
  
  df_selected = pd.DataFrame(df_selected.values, columns=list(column_mapping.values()))
  
  columns_to_drop = ['ca', 'slope','thal']
  df_selected = df_selected.drop(columns_to_drop, axis=1)

  mean_tbps = df_selected['trestbps'].dropna()
  mean_chol = df_selected['chol'].dropna()
  mean_fbs = df_selected['fbs'].dropna()
  mean_restcg = df_selected['restecg'].dropna()
  mean_thalach = df_selected['thalach'].dropna()
  mean_exang = df_selected['exang'].dropna()

  mean_tbps = mean_tbps.astype(float)
  mean_chol = mean_chol.astype(float)
  mean_fbs = mean_fbs.astype(float)
  mean_thalach = mean_thalach.astype(float)
  mean_exang = mean_exang.astype(float)
  mean_restcg = mean_restcg.astype(float)

  mean_tbps = round(mean_tbps.mean())
  mean_chol = round(mean_chol.mean())
  mean_fbs = round(mean_fbs.mean())
  mean_thalach = round(mean_thalach.mean())
  mean_exang = round(mean_exang.mean())
  mean_restcg = round(mean_restcg.mean())

  fill_values = {
    'trestbps': mean_tbps,
    'chol': mean_chol,
    'fbs': mean_fbs,
    'thalach':mean_thalach,
    'exang':mean_exang,
    'restecg':mean_restcg
  }

  df_clean = df_selected.fillna(value=fill_values)
  df_clean.drop_duplicates(inplace=True)

  x = df_clean.drop('target', axis=1)
  y = df_clean['target']

  smote = SMOTE(random_state=4)
  x, y = smote.fit_resample(x, y)
  
  df_final = x
  
  scaler = MinMaxScaler()
  x = scaler.fit_transform(x)

  df_final['target'] = y
  
  with open('models/knn-model-tun-norm.jbl', 'rb') as stream:
    knn_model_tun_norm = joblib.load(stream)
  
  with open('models/rf-model-tun-norm.jbl', 'rb') as stream:
    rf_model_tun_norm = joblib.load(stream)
    
  with open('models/xgb-model-tun-norm.jbl', 'rb') as stream:
    xgb_model_tun_norm = joblib.load(stream)
  
  y_pred_knn = knn_model_tun_norm.predict(x)
  accuracy_knn = accuracy_score(y, y_pred_knn)
  accuracy_knn = round((accuracy_knn * 100), 2)
  
  y_pred_rf = rf_model_tun_norm.predict(x)
  accuracy_rf = accuracy_score(y, y_pred_rf)
  accuracy_rf = round((accuracy_rf * 100), 2)
  
  y_pred_xgb = xgb_model_tun_norm.predict(x)
  accuracy_xgb = accuracy_score(y, y_pred_xgb)
  accuracy_xgb = round((accuracy_xgb * 100), 2)
  
  fig, ax = plt.subplots()
  bars = plt.bar([
      'KNN',
      'Random Forest',
      'XGBOOST',
    ], [
      accuracy_knn,
      accuracy_rf,
      accuracy_xgb,
    ], 
    color=sns.palettes.mpl_palette('Dark2'),
  )
  
  plt.xlabel('Model')
  plt.ylabel('Accuracy (%)')
  plt.title('Tunning + Oversample + Normalization')
  plt.xticks(rotation=45, ha='right')

  for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

  # save as data buffer!
  buffer = io.BytesIO()
  plt.savefig(buffer)
  plt.close()
  
  # convert data buffer into image class!
  buffer.seek(0)
  image = Image.open(buffer)
    
  st.set_page_config(
  page_title = 'Hungarian Heart Disease',
  page_icon = ':heart:'
)

  st.title('Hungarian Heart Disease')
  st.write(f'Models KNN :green[**{accuracy_knn}**]% + Random Forest :green[**{accuracy_rf}**]% + XGBOOST :green[**{accuracy_xgb}**]% (Tunning + Oversample + Normalization)')
  st.write('')
    
  st.image(image, caption='model tunning normalization graph accuracy')
  
  single_predict_tab, multi_predict_tab = st.tabs(['Single Predict', 'Multi Predict'])
  
  with single_predict_tab:
    st.sidebar.header('User Input With Sidebar')

    age_min = df_final['age'].min()
    age_max = df_final['age'].max()
    age = st.sidebar.number_input(label=':violet[**Age**]', min_value=age_min, max_value=age_max)
    
    st.sidebar.write(f':orange[Min] value: :orange[**{age_min}**], :red[Max] value: :red[**{age_max}**]')
    st.sidebar.write('')

    sex_sb = st.sidebar.selectbox(label=':violet[**Sex**]', options=['Male', 'Female'])
    st.sidebar.write('')
    st.sidebar.write('')
      
    sex = [
      'Female', 
      'Male', 
    ].index(sex_sb)

    cp_sb = st.sidebar.selectbox(label=':violet[**Chest pain type**]', options=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
    st.sidebar.write('')
    st.sidebar.write('')
      
    cp = [
      'Typical angina', 
      'Atypical angina', 
      'Non-anginal pain', 
      'Asymptomatic', 
    ].index(cp_sb) + 1
    
    trestbps_min = df_final['trestbps'].min()
    trestbps_max = df_final['trestbps'].max()
    
    trestbps = st.sidebar.number_input(label=':violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]', 
                                       min_value=trestbps_min, max_value=trestbps_max, )
    
    st.sidebar.write(f':orange[Min] value: :orange[**{trestbps_min}**], :red[Max] value: :red[**{trestbps_max}**]')
    st.sidebar.write('')

    chol_min = df_final['chol'].min()
    chol_max = df_final['chol'].max()

    chol = st.sidebar.number_input(label=':violet[**Serum cholestoral** (in mg/dl)]', min_value=chol_min, max_value=chol_max)
    
    st.sidebar.write(f':orange[Min] value: :orange[**{chol_min}**], :red[Max] value: :red[**{chol_max}**]')
    st.sidebar.write('')

    fbs_sb = st.sidebar.selectbox(label=':violet[**Fasting blood sugar > 120 mg/dl?**]', options=['False', 'True'])
    st.sidebar.write('')
    st.sidebar.write('')
      
    fbs = [
      'False', 
      'True',
    ].index(fbs_sb)

    restecg_sb = st.sidebar.selectbox(label=':violet[**Resting electrocardiographic results**]', options=['Normal', 'Having ST-T wave abnormality', 'Showing left ventricular hypertrophy'])
    st.sidebar.write('')
    st.sidebar.write('')
    
    restecg = [
      'Normal',
      'Having ST-T wave abnormality',
      'Showing left ventricular hypertrophy',
    ].index(restecg_sb)
    
    thalach_min = df_final['thalach'].min()
    thalach_max = df_final['thalach'].max()

    thalach = st.sidebar.number_input(label=':violet[**Maximum heart rate achieved**]', 
                                      min_value=thalach_min, max_value=thalach_max, )
    
    st.sidebar.write(f':orange[Min] value: :orange[**{thalach_min}**], :red[Max] value: :red[**{thalach_max}**]')
    st.sidebar.write('')

    exang_sb = st.sidebar.selectbox(label=':violet[**Exercise induced angina?**]', options=['No', 'Yes'])
    st.sidebar.write('')
    st.sidebar.write('')
    
    exang = [
      'No',
      'Yes',
    ].index(exang_sb)

    oldpeak_min = df_final['oldpeak'].min()
    oldpeak_max = df_final['oldpeak'].max()

    oldpeak = st.sidebar.number_input(label=':violet[**ST depression induced by exercise relative to rest**]', 
                                      min_value=oldpeak_min, max_value=oldpeak_max, )
    
    st.sidebar.write(f':orange[Min] value: :orange[**{oldpeak_min}**], :red[Max] value: :red[**{oldpeak_max}**]')
    st.sidebar.write('')

    data = {
      'Age': age,
      'Sex': sex_sb,
      'Chest pain type': cp_sb,
      'RPB': f'{trestbps} mm Hg',
      'Serum Cholestoral': f'{chol} mg/dl',
      'FBS > 120 mg/dl?': fbs_sb,
      'Resting ECG': restecg_sb,
      'Maximum heart rate': thalach,
      'Exercise induced angina?': exang_sb,
      'ST depression': oldpeak,
    }

    preview_df = pd.DataFrame(data, index=['input'])

    st.header('User Input as DataFrame')
    st.write('')
    st.dataframe(preview_df.iloc[:, :6])
    st.write('')
    st.dataframe(preview_df.iloc[:, 6:])
    st.write('')

    result = ':violet[-]'

    predict_btn = st.button('**Predict**', type='primary')

    st.write('')
    if predict_btn:
      inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
      
      
      def test(name, model, x):
        
        # model prediction
        prediction = model.predict(x)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
          status_text.text(f'{i}% complete')
          bar.progress(i)
          time.sleep(0.01)
          
          if i == 100:
            time.sleep(1)
            status_text.empty()
            bar.empty()
        
        score = round(prediction)
        status = [
          ':green[**Healthy**]',
          ':orange[**Heart Disease Level 1**]',
          ':orange[**Heart Disease Level 2**]',
          ':red[**Heart Disease Level 3**]',
          ':red[**Heart Disease Level 4**]',
        ][score]

        st.write('')
        st.subheader(f'Prediction ({name}): {status}')
        
      test('KNN', knn_model_tun_norm, inputs)
      test('Random Forest', rf_model_tun_norm, inputs)
      test('XGBOOST', xgb_model_tun_norm, inputs)
  
  with multi_predict_tab:
    st.header('Predict multiple data:')

    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

    st.write('')
    st.download_button('Download CSV Example', data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

    st.write('')
    st.write('')
    file_uploaded = st.file_uploader('Upload a CSV file', type='csv')

    if file_uploaded:
      uploaded_df = pd.read_csv(file_uploaded)
      
      def test(model, x):
        
        # normalization
        x = scaler.fit_transform(x)
        
        # model prediction
        prediction_arr = model.predict(x)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
          status_text.text(f'{i}% complete')
          bar.progress(i)
          time.sleep(0.01)

        result = []

        for prediction in prediction_arr:
          score = round(prediction)
          status = [
            'Healthy',
            'Heart Disease Level 1',
            'Heart Disease Level 2',
            'Heart Disease Level 3',
            'Heart Disease Level 4',
          ][score]
          
          result.append(status)
          
        for i in range(70, 101):
          status_text.text(f'{i}% complete')
          bar.progress(i)
          time.sleep(0.01)
          
          if i == 100:
            time.sleep(1)
            status_text.empty()
            bar.empty()
        
        return result

      inputs = uploaded_df.values
      uploaded_result = pd.DataFrame({
        'Prediction (KNN)': test(knn_model_tun_norm, inputs),
        'Prediction (Random Forest)': test(rf_model_tun_norm, inputs),
        'Prediction (XGBOOST)': test(xgb_model_tun_norm, inputs),
      })

      col1, col2 = st.columns([1, 2])

      with col1:
        st.dataframe(uploaded_result)
        
      with col2:
        st.dataframe(uploaded_df)


if str(__name__).upper() in ('__MAIN__',):
  main() 