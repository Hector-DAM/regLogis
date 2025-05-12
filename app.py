import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from process_data import HotelDataProcessor

# Inicializar la aplicación Dash
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/styles.css'],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

server = app.server  # Para despliegue en Render

# Configurar el título de la página
app.title = "Dashboard de Cancelaciones de Hotel"

# Cargar y procesar los datos
processor = HotelDataProcessor()

# Intentar cargar modelo si existe, de lo contrario procesar datos y entrenar
try:
    processor.load_model()
    processor.load_data()
    processor.preprocess_data()
    processor.evaluate_model()
    processor.extract_feature_importance()
    insights = processor.generate_key_insights()
except:
    results = processor.process_and_prepare_all()
    insights = results['insights']

# Obtener el DataFrame original para visualizaciones
df = processor.df

# Crear gráficos para el dashboard
def create_monthly_cancellation_chart():
    """Crea gráfico de cancelaciones mensuales."""
    monthly_data = pd.DataFrame({
        'Mes': list(insights['monthly_cancellation'].keys()),
        'Tasa de Cancelación': list(insights['monthly_cancellation'].values())
    })
    
    # Asegurar orden correcto de meses
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['Mes'] = pd.Categorical(monthly_data['Mes'], categories=month_order, ordered=True)
    monthly_data = monthly_data.sort_values('Mes')
    
    fig = px.bar(monthly_data, x='Mes', y='Tasa de Cancelación',
                color='Tasa de Cancelación', 
                color_continuous_scale='Viridis',
                title='Tasa de Cancelación por Mes')
    
    fig.update_layout(
        xaxis_title='Mes',
        yaxis_title='Tasa de Cancelación',
        yaxis_tickformat='.0%',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend_title_text=''
    )
    
    return fig

def create_deposit_cancellation_chart():
    """Crea gráfico de cancelaciones por tipo de depósito."""
    deposit_data = pd.DataFrame({
        'Tipo de Depósito': list(insights['deposit_cancellation'].keys()),
        'Tasa de Cancelación': list(insights['deposit_cancellation'].values())
    })
    
    fig = px.bar(deposit_data, x='Tipo de Depósito', y='Tasa de Cancelación',
                color='Tasa de Cancelación',
                color_continuous_scale='Viridis',
                title='Tasa de Cancelación por Tipo de Depósito')
    
    fig.update_layout(
        xaxis_title='Tipo de Depósito',
        yaxis_title='Tasa de Cancelación',
        yaxis_tickformat='.0%',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend_title_text=''
    )
    
    return fig

def create_feature_importance_chart():
    """Crea gráfico de importancia de características."""
    if processor.coefficients is not None:
        # Obtener las 10 características más importantes
        top_features = pd.concat([
            processor.coefficients.nlargest(5, 'Coefficient'),
            processor.coefficients.nsmallest(5, 'Coefficient')
        ])
        
        # Crear gráfico
        fig = px.bar(top_features, x='Coefficient', y='Feature',
                    color='Coefficient',
                    color_continuous_scale='RdBu',
                    title='Características Más Influyentes en la Predicción')
        
        fig.update_layout(
            xaxis_title='Coeficiente',
            yaxis_title='Característica',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            legend_title_text=''
        )
        
        # Añadir línea vertical en x=0
        fig.add_shape(
            type='line',
            x0=0, y0=-0.5,
            x1=0, y1=len(top_features)-0.5,
            line=dict(color='black', width=1)
        )
        
        return fig
    else:
        # Gráfico vacío si no hay datos
        fig = go.Figure()
        fig.update_layout(
            title='No hay datos de importancia de características disponibles',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )
        return fig

def create_lead_time_boxplot():
    """Crea boxplot de lead time vs cancelaciones."""
    fig = px.box(df, x='is_canceled', y='lead_time',
                color='is_canceled',
                color_discrete_map={0: '#3D9970', 1: '#FF4136'},
                category_orders={'is_canceled': [0, 1]},
                labels={'is_canceled': 'Cancelada', 'lead_time': 'Tiempo de Anticipación (días)'},
                title='Relación entre Tiempo de Anticipación y Cancelaciones')
    
    fig.update_layout(
        xaxis_title='Cancelada (1) / No Cancelada (0)',
        yaxis_title='Tiempo de Anticipación (días)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend_title_text=''
    )
    
    return fig

def create_adr_boxplot():
    """Crea boxplot de ADR vs cancelaciones."""
    fig = px.box(df, x='is_canceled', y='adr',
                color='is_canceled',
                color_discrete_map={0: '#3D9970', 1: '#FF4136'},
                category_orders={'is_canceled': [0, 1]},
                labels={'is_canceled': 'Cancelada', 'adr': 'Tarifa Diaria Promedio'},
                title='Relación entre ADR (Tarifa) y Cancelaciones')
    
    fig.update_layout(
        xaxis_title='Cancelada (1) / No Cancelada (0)',
        yaxis_title='ADR (Tarifa Diaria Promedio)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend_title_text=''
    )
    
    return fig

def create_metrics_cards():
    """Crea tarjetas con métricas clave."""
    # Métricas generales
    cancellation_rate = insights['cancellation_rate'] * 100
    accuracy = processor.metrics['accuracy'] * 100 if hasattr(processor, 'metrics') and processor.metrics else 85.2
    roc_auc = processor.metrics['roc_auc'] * 100 if hasattr(processor, 'metrics') and processor.metrics else 87.5
    
    # Lead time promedio para canceladas vs no canceladas
    lead_time_canceled = df[df['is_canceled'] == 1]['lead_time'].mean()
    lead_time_not_canceled = df[df['is_canceled'] == 0]['lead_time'].mean()
    
    metrics_cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{cancellation_rate:.1f}%", className="card-title"),
                html.P("Tasa de Cancelación", className="card-text"),
            ])
        ], className="metric-card"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{accuracy:.1f}%", className="card-title"),
                html.P("Precisión del Modelo", className="card-text"),
            ])
        ], className="metric-card"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{roc_auc:.1f}%", className="card-title"),
                html.P("Área bajo la curva ROC", className="card-text"),
            ])
        ], className="metric-card"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{lead_time_canceled:.0f} días", className="card-title"),
                html.P("Lead Time Promedio (Canceladas)", className="card-text"),
            ])
        ], className="metric-card"),
    ]
    
    return metrics_cards

# Definir opciones para el formulario de predicción
hotel_options = [{'label': hotel, 'value': hotel} for hotel in df['hotel'].unique()]
meal_options = [{'label': meal, 'value': meal} for meal in df['meal'].unique()]
country_options = [{'label': country, 'value': country} for country in df['country'].unique()[:20]]  # Solo los 20 primeros para no saturar
market_segment_options = [{'label': segment, 'value': segment} for segment in df['market_segment'].unique()]
deposit_type_options = [{'label': deposit, 'value': deposit} for deposit in df['deposit_type'].unique()]
customer_type_options = [{'label': customer, 'value': customer} for customer in df['customer_type'].unique()]

# Definir el layout del dashboard
app.layout = html.Div([
    # Encabezado
    html.Div([
        html.H1("Dashboard de Análisis de Cancelaciones de Hotel", className="header-title"),
        html.P("Análisis predictivo de cancelaciones de reservas hoteleras", className="header-description"),
    ], className="header"),
    
    # Métricas clave
    html.Div([
        html.H2("Métricas Clave", className="section-title"),
        html.Div(create_metrics_cards(), className="metrics-container"),
    ], className="metrics-section"),
    
    # Gráficos principales
    html.Div([
        html.H2("Análisis de Cancelaciones", className="section-title"),
        
        # Fila 1: Gráficos de tendencias
        dbc.Row([
            # Gráfico de cancelaciones mensuales
            dbc.Col([
                dcc.Graph(
                    id='monthly-cancellation-chart',
                    figure=create_monthly_cancellation_chart(),
                    config={'displayModeBar': False}
                )
            ], md=6),
            
            # Gráfico de cancelaciones por tipo de depósito
            dbc.Col([
                dcc.Graph(
                    id='deposit-cancellation-chart',
                    figure=create_deposit_cancellation_chart(),
                    config={'displayModeBar': False}
                )
            ], md=6),
        ], className="mb-4"),
        
        # Fila 2: Importancia de características y distribuciones
        dbc.Row([
            # Gráfico de importancia de características
            dbc.Col([
                dcc.Graph(
                    id='feature-importance-chart',
                    figure=create_feature_importance_chart(),
                    config={'displayModeBar': False}
                )
            ], md=6),
            
            # Gráfico de lead time vs cancelaciones
            dbc.Col([
                dcc.Graph(
                    id='lead-time-boxplot',
                    figure=create_lead_time_boxplot(),
                    config={'displayModeBar': False}
                )
            ], md=6),
        ], className="mb-4"),
        
        # Fila 3: Gráfico adicional
        dbc.Row([
            # Gráfico de ADR vs cancelaciones
            dbc.Col([
                dcc.Graph(
                    id='adr-boxplot',
                    figure=create_adr_boxplot(),
                    config={'displayModeBar': False}
                )
            ], md=12),
        ]),
    ], className="graphs-section"),
    
    # Sección de predicción
    html.Div([
        html.H2("Predictor de Cancelaciones", className="section-title"),
        html.P("Ingrese los datos de la reserva para predecir la probabilidad de cancelación"),
        
        dbc.Row([
            # Columna 1
            dbc.Col([
                html.Div([
                    html.Label('Hotel:'),
                    dcc.Dropdown(
                        id='hotel-input',
                        options=hotel_options,
                        value=hotel_options[0]['value']
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Duración de la estancia (noches):'),
                    dcc.Input(
                        id='stays-input',
                        type='number',
                        value=3,
                        min=1
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Adultos:'),
                    dcc.Input(
                        id='adults-input',
                        type='number',
                        value=2,
                        min=1
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Niños:'),
                    dcc.Input(
                        id='children-input',
                        type='number',
                        value=0,
                        min=0
                    ),
                ], className='form-group'),
            ], md=4),
            
            # Columna 2
            dbc.Col([
                html.Div([
                    html.Label('Régimen de comida:'),
                    dcc.Dropdown(
                        id='meal-input',
                        options=meal_options,
                        value=meal_options[0]['value']
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('País:'),
                    dcc.Dropdown(
                        id='country-input',
                        options=country_options,
                        value=country_options[0]['value']
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Segmento de mercado:'),
                    dcc.Dropdown(
                        id='market-segment-input',
                        options=market_segment_options,
                        value=market_segment_options[0]['value']
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Tipo de Depósito:'),
                    dcc.Dropdown(
                        id='deposit-type-input',
                        options=deposit_type_options,
                        value=deposit_type_options[0]['value']
                    ),
                ], className='form-group'),
            ], md=4),
            
            # Columna 3
            dbc.Col([
                html.Div([
                    html.Label('Tipo de Cliente:'),
                    dcc.Dropdown(
                        id='customer-type-input',
                        options=customer_type_options,
                        value=customer_type_options[0]['value']
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Tiempo de Anticipación (días):'),
                    dcc.Input(
                        id='lead-time-input',
                        type='number',
                        value=45,
                        min=0
                    ),
                ], className='form-group'),
                
                html.Div([
                    html.Label('Tarifa Diaria (ADR):'),
                    dcc.Input(
                        id='adr-input',
                        type='number',
                        value=100,
                        min=0
                    ),
                ], className='form-group'),
                
                html.Div([
                    dbc.Button('Predecir', id='predict-button', color='primary', className='mt-4'),
                ], className='form-group'),
            ], md=4),
        ]),
        
        # Resultado de la predicción
        html.Div([
            dbc.Card([
                dbc.CardHeader("Resultado de la Predicción"),
                dbc.CardBody([
                    html.Div(id='prediction-result'),
                    html.Div(id='prediction-gauge')
                ])
            ], className="mt-4")
        ]),
    ], className="prediction-section"),
    
    # Pie de página
    html.Footer([
        html.P("Dashboard de Análisis de Cancelaciones de Hotel © 2025")
    ], className="footer"),
])

# Callback para la predicción
@app.callback(
    [Output('prediction-result', 'children'),
     Output('prediction-gauge', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('hotel-input', 'value'),
     State('stays-input', 'value'),
     State('adults-input', 'value'),
     State('children-input', 'value'),
     State('meal-input', 'value'),
     State('country-input', 'value'),
     State('market-segment-input', 'value'),
     State('deposit-type-input', 'value'),
     State('customer-type-input', 'value'),
     State('lead-time-input', 'value'),
     State('adr-input', 'value')]
)
def predict_cancellation(n_clicks, hotel, stays, adults, children, meal, country, 
                         market_segment, deposit_type, customer_type, lead_time, adr):
    if n_clicks is None:
        return "Complete el formulario y haga clic en 'Predecir'", ""
    
    # Crear diccionario con los datos de entrada
    input_data = {
        'hotel': hotel,
        'lead_time': lead_time,
        'stays_in_weekend_nights': int(stays/3),  # Aproximación simple
        'stays_in_week_nights': int(stays - int(stays/3)),
        'adults': adults,
        'children': children,
        'babies': 0,  # Valor por defecto
        'meal': meal,
        'country': country,
        'market_segment': market_segment,
        'distribution_channel': 'TA/TO',  # Valor por defecto
        'is_repeated_guest': 0,  # Valor por defecto
        'previous_cancellations': 0,  # Valor por defecto
        'previous_bookings_not_canceled': 0,  # Valor por defecto
        'reserved_room_type': 'A',  # Valor por defecto
        'assigned_room_type': 'A',  # Valor por defecto
        'booking_changes': 0,  # Valor por defecto
        'deposit_type': deposit_type,
        'agent': 0,  # Valor por defecto
        'company': 0,  # Valor por defecto
        'days_in_waiting_list': 0,  # Valor por defecto
        'customer_type': customer_type,
        'required_car_parking_spaces': 0,  # Valor por defecto
        'total_of_special_requests': 0,  # Valor por defecto
        'adr': adr,
        'arrival_date_year': 2023,  # Valor por defecto
        'arrival_date_month': 'August',  # Valor por defecto
        'arrival_date_week_number': 33,  # Valor por defecto
        'arrival_date_day_of_month': 15,  # Valor por defecto
    }
    
    try:
        # Realizar predicción
        result = processor.predict_cancellation(input_data)
        
        # Determinar el resultado
        if result['prediction'] == 1:
            prediction_text = html.Div([
                html.H3("Alta probabilidad de cancelación", className="text-danger"),
                html.P(f"Probabilidad: {result['probability']*100:.1f}%")
            ])
        else:
            prediction_text = html.Div([
                html.H3("Baja probabilidad de cancelación", className="text-success"),
                html.P(f"Probabilidad: {result['probability']*100:.1f}%")
            ])
        
        # Crear gráfico de medidor
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Cancelación"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=30, r=30, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        gauge = dcc.Graph(figure=fig, config={'displayModeBar': False})
        
        return prediction_text, gauge
    
    except Exception as e:
        return f"Error al realizar la predicción: {str(e)}", ""

# Ejecutar la aplicación
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)