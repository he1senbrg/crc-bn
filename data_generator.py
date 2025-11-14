import numpy as np
import pandas as pd
from config import YEARS, N_PATIENTS

def generate_synthetic_crc_data(n_patients=N_PATIENTS, years=YEARS):
    ages = np.random.choice(['<50','50-64','>=65'], size=n_patients, p=[0.6,0.25,0.15])
    sex = np.random.choice(['M','F'], size=n_patients, p=[0.48,0.52])
    smoking = np.random.choice(['No','Yes'], size=n_patients, p=[0.85,0.15])
    alcohol = np.random.choice(['None','Moderate','High'], size=n_patients, p=[0.6,0.3,0.1])
    diabetes = np.random.choice(['No','Yes'], size=n_patients, p=[0.92,0.08])
    hypertension = np.random.choice(['No','Yes'], size=n_patients, p=[0.9,0.1])
    bmi = np.random.choice(['Normal','Overweight','Obese'], size=n_patients, p=[0.5,0.35,0.15])
    year = np.random.choice(years, size=n_patients)
    
    base = np.where(ages=='<50', 0.0002, np.where(ages=='50-64', 0.001, 0.005))
    base += (smoking=='Yes') * 0.002
    base += (alcohol=='High') * 0.002
    base += (diabetes=='Yes') * 0.0015
    base += (bmi=='Obese') * 0.001
    base += (hypertension=='Yes') * 0.0008
    p_crc = np.clip(base, 0, 0.05)
    crc = np.where(np.random.rand(n_patients) < p_crc, 'Yes', 'No')
    
    df = pd.DataFrame({
        'Year': year,
        'Age': ages,
        'Sex': sex,
        'Smoking': smoking,
        'Alcohol': alcohol,
        'Diabetes': diabetes,
        'Hypertension': hypertension,
        'BMI': bmi,
        'CRC': crc
    })
    df['PatientID'] = range(len(df))
    return df[['PatientID','Year','Age','Sex','Smoking','Alcohol','Diabetes','Hypertension','BMI','CRC']]