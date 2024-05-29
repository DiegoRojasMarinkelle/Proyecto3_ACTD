#pip install dash
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

import pandas as pd
file_path = 'Dataset_Proyecto3_ACTD.csv'
df = pd.read_csv(file_path)

# Convert punt_global to numeric to ensure proper categorization
df['punt_global'] = pd.to_numeric(df['punt_global'], errors='coerce')

# Categorize punt_global
def categorize_punt_global(punt_global):
    if 0 <= punt_global <= 199:
        return 'Malo'
    elif 200 <= punt_global <= 299:
        return 'Regular'
    elif 300 <= punt_global <= 500:
        return 'Bueno'
    else:
        return 'Desconocido'

df['punt_global_categoria'] = df['punt_global'].apply(categorize_punt_global)


df = df[df['cole_calendario'] != 'OTRO']
df = df[~df['estu_tipodocumento'].isin(['PC', 'PV', 'V'])]
df = df.dropna()

# Example categorical and numerical variables
categorical_variables = ['estu_tipodocumento', 'cole_area_ubicacion', 'cole_bilingue',
                        'cole_calendario', 'cole_caracter', 'cole_jornada', 'estu_genero',
                        'estu_mcpio_presentacion', 'fami_estratovivienda',
                        'fami_tieneautomovil', 'fami_tienecomputador', 'fami_tieneinternet',
                        'fami_tienelavadora', 'desemp_ingles',]
numerical_variables = ['punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas', 'punt_c_naturales',
                        'punt_lectura_critica']


# Load your pre-trained machine learning model
with open('model_over_final.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the preprocessor from the .joblib file
preprocessor = joblib.load('preprocessor.joblib')


# Crear el gráfico de caja con estilos especiales para 'A-' y 'B+'
fig1 = px.box(df, x='desemp_ingles', y='punt_global', title='Puntaje global por desempeño en inglés',
              labels={'desemp_ingles': 'Desempeño en inglés', 'punt_global': 'Puntaje global'})



fig2 = px.box(df, x='fami_estratovivienda', y='punt_global', title='Puntaje global por estrato socioeconómico',
             labels={'fami_estratovivienda': 'Estrato socioeconómico', 'punt_global': 'Puntaje global'})


# Initialize the Dash app with a title
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] # External stylesheet for corporate style
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Proyecto-3 Pruebas Saber 11")

# Define company's color palette
colors = {
    'background': '#f9f9f9', # Light background color
    'text': '#333333',  # Dark text color
    'company_color': '#0072ce'  # Company's primary color
}

# Crear campos de entrada y gráfico
input_fields = []
for var in categorical_variables:
    clases = df[var].unique()
    default_value = clases[0] 

    dropdown_options = [{'label': x, 'value': x} for x in clases if x is not None and x != 'null']
    dropdown_options.insert(0, {'label': default_value, 'value': default_value}) 

    input_fields.append(html.Div([
        html.Div(var, style={'color': 'black'}),  
        dcc.Dropdown(id=f'input-{var}', 
                     options=dropdown_options,
                     value=default_value, 
                     placeholder='Select an option')
    ]))

for var in numerical_variables:
    input_fields.append(html.Div([
        html.Div(var, style={'color': 'black'}),
        dcc.Input(id=f'input-{var}', type='number', placeholder='Enter a number', value=50)  
    ]))

# Crear un diccionario para almacenar cada LabelEncoder
label_encoders = {}
for var in categorical_variables:
    le = LabelEncoder()
    df[var] = le.fit_transform(df[var])
    label_encoders[var] = le

# Callback para actualizar la predicción
@app.callback(
    Output('output-prediction', 'children'),
    [Input(f'input-{var}', 'value') for var in categorical_variables + numerical_variables]
)
def update_prediction(*args):
    titles = [var for var in categorical_variables + numerical_variables]

    data_dict = dict(zip(titles, args))    

    df_inputs = pd.DataFrame([data_dict], columns=data_dict.keys())

    for k in categorical_variables:
        selected_label_encoder = label_encoders[k]
        df_inputs[k] = selected_label_encoder.transform(df_inputs[k])

    df_preprocessed = preprocessor.transform(df_inputs)
    preprocessed = pd.DataFrame(df_preprocessed, columns=titles)

    prediction = model.predict(preprocessed)
    prediction_result = np.argmax(prediction, axis=1)

    class_mapping = {0: 'Aceptable', 1: 'Bueno', 2: 'Malo'}
    predicted_classes = [class_mapping[pred] for pred in prediction_result] 

    return f"Puntaje obtenido: {predicted_classes[0]}"




# Define layout combinado
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    html.H1("Puntajes Saber 11", style={'color': colors['company_color']}),

    html.H2(id='output-prediction'),

    html.Div(style={'display': 'flex'}, children=[
        html.Div(style={'padding': '20px', 'width': '50%'}, children=[  # Izquierda
            html.H2("Entrada de Datos"),
            *input_fields
        ]),

        html.Div(style={'padding': '20px', 'width': '50%'}, children=[  # Derecha
            html.H2("Visualizaciones"),
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)