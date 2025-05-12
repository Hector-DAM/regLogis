# Dashboard de Análisis de Cancelaciones de Hotel

Esta aplicación web proporciona un dashboard interactivo para el análisis predictivo de cancelaciones de reservas hoteleras. Utiliza un modelo de regresión logística para predecir la probabilidad de que una reserva sea cancelada, basándose en diversos factores.

## Características

- **Análisis exploratorio visual** de los datos de reservas
- **Modelo predictivo** de cancelaciones basado en regresión logística
- **Dashboard interactivo** con métricas clave y visualizaciones
- **Herramienta de predicción** en tiempo real para nuevas reservas
- **Diseño responsive** adaptado a diferentes dispositivos

## Estructura del Proyecto

```
hotel-cancellation-dashboard/
├── app.py                  # Aplicación principal Dash
├── process_data.py         # Módulo para procesar datos y entrenar modelo
├── assets/                 # Recursos estáticos
│   └── styles.css          # Estilos CSS
├── templates/              # Plantillas HTML
│   └── index.html          # Plantilla base
├── model/                  # Carpeta para almacenar modelo entrenado
│   └── hotel_cancellation_model.pkl
├── requirements.txt        # Dependencias
└── README.md               # Documentación
```

## Requisitos

- Python 3.8+
- Dash
- Plotly
- Pandas
- NumPy
- Scikit-learn
- Flask
- Gunicorn (para despliegue)

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tuusuario/hotel-cancellation-dashboard.git
cd hotel-cancellation-dashboard
```

2. Crea un entorno virtual e instala las dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Coloca el archivo de datos `hotel_bookings.csv` en la raíz del proyecto.

## Ejecución Local

```bash
python app.py
```

Abre tu navegador en `http://localhost:8050`

## Despliegue en Render

Esta aplicación está configurada para ser desplegada en Render usando el archivo `render.yaml`. Para desplegarla:

1. Crea una cuenta en [Render](https://render.com/)
2. Conecta tu repositorio de GitHub
3. Render detectará automáticamente la configuración y desplegará la aplicación

## Modelo de Predicción

El modelo utiliza las siguientes características para predecir cancelaciones:

- Tipo de hotel
- Tiempo de anticipación (lead time)
- Duración de la estancia
- Número de adultos y niños
- Tipo de comida
- País de origen
- Segmento de mercado
- Tipo de depósito
- Tipo de cliente
- Tarifa diaria (ADR)

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto

Para preguntas y soporte, por favor contacta a través de [tuemail@example.com](mailto:tuemail@example.com).