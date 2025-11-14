import numpy as np

np.random.seed(42)

VARIABLES = ['Age', 'Sex', 'Smoking', 'Alcohol', 'Diabetes', 'Hypertension', 'BMI', 'CRC']

WHITELIST = [
    ('Age', 'CRC'), ('Smoking', 'CRC'), ('Alcohol', 'CRC'), ('Diabetes', 'CRC'),
    ('BMI', 'Diabetes'), ('BMI', 'Hypertension'), ('Age', 'Diabetes')
]

BLACKLIST = [
    ('CRC', 'Age'), ('CRC', 'Smoking'), ('CRC', 'Alcohol'), ('CRC', 'BMI')
]

YEARS = (2012, 2013, 2014, 2015, 2016)
N_PATIENTS = 25000