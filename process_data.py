import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import joblib
import os

class HotelDataProcessor:
    def __init__(self, data_path='hotel_bookings.csv'):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.coefficients = None
        self.accuracy = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.metrics = {}
        
    def load_data(self):
        """Carga los datos del archivo CSV."""
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess_data(self):
        """Realiza el preprocesamiento de los datos."""
        if self.df is None:
            self.load_data()
        
        # Manejar valores faltantes
        self.df['children'] = self.df['children'].fillna(0)
        self.df['country'] = self.df['country'].fillna('unknown')
        self.df['agent'] = self.df['agent'].fillna(0)
        self.df['company'] = self.df['company'].fillna(0)
        
        # Eliminar columnas no útiles
        if 'reservation_status' in self.df.columns:
            self.df = self.df.drop(['reservation_status', 'reservation_status_date'], axis=1)
        
        # Separar características y objetivo
        features = self.df.drop(['is_canceled'], axis=1)
        target = self.df['is_canceled']
        
        # Identificar columnas categóricas y numéricas
        self.categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.3, random_state=42
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_and_train_model(self):
        """Construye y entrena el modelo de regresión logística."""
        if self.X_train is None:
            self.preprocess_data()
        
        # Preprocesamiento para columnas numéricas
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocesamiento para columnas categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinar preprocesadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        # Crear pipeline con preprocesador y modelo
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        # Entrenar el modelo
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def evaluate_model(self):
        """Evalúa el modelo y calcula métricas."""
        if self.model is None:
            self.build_and_train_model()
        
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Precisión del modelo
        self.accuracy = accuracy_score(self.y_test, y_pred)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Reporte de clasificación
        class_report = classification_report(self.y_test, y_pred, 
                                            target_names=['No Cancelada', 'Cancelada'],
                                            output_dict=True)
        
        # Guardar métricas
        self.metrics = {
            'accuracy': self.accuracy,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'classification_report': class_report
        }
        
        return self.metrics
    
    def extract_feature_importance(self):
        """Extrae la importancia de las características del modelo."""
        if self.model is None:
            self.build_and_train_model()
        
        # Crear lista de nombres de características
        feature_names = []
        
        # Obtener nombres de características para columnas numéricas
        for col in self.numerical_cols:
            feature_names.append(col)
        
        # Obtener nombres de características para columnas categóricas (one-hot encoding)
        categorical_features = self.model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(self.categorical_cols)
        for cat_feature in categorical_features:
            feature_names.append(cat_feature)
        
        # Verificar si las dimensiones coinciden
        if len(feature_names) == self.model['classifier'].coef_.shape[1]:
            self.feature_names = feature_names
            self.coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': self.model['classifier'].coef_[0]
            })
            
            # Ordenar por importancia absoluta
            self.coefficients = self.coefficients.reindex(
                self.coefficients['Coefficient'].abs().sort_values(ascending=False).index
            )
            
            return self.coefficients
        else:
            return None
    
    def generate_key_insights(self):
        """Genera insights clave sobre los datos."""
        if self.df is None:
            self.load_data()
        
        insights = {}
        
        # Tasa de cancelación general
        insights['cancellation_rate'] = self.df['is_canceled'].mean()
        
        # Tasas de cancelación por mes
        cancellation_by_month = self.df.groupby('arrival_date_month')['is_canceled'].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        cancellation_by_month = cancellation_by_month.reindex(month_order)
        insights['monthly_cancellation'] = cancellation_by_month.to_dict()
        
        # Tasas de cancelación por tipo de depósito
        insights['deposit_cancellation'] = self.df.groupby('deposit_type')['is_canceled'].mean().to_dict()
        
        # Estadísticas de lead time
        lead_time_stats = self.df.groupby('is_canceled')['lead_time'].describe()
        insights['lead_time_stats'] = lead_time_stats.to_dict()
        
        # Estadísticas de ADR (tarifa)
        adr_stats = self.df.groupby('is_canceled')['adr'].describe()
        insights['adr_stats'] = adr_stats.to_dict()
        
        return insights
    
    def save_model(self, filepath='model/hotel_cancellation_model.pkl'):
        """Guarda el modelo entrenado en un archivo."""
        if self.model is None:
            self.build_and_train_model()
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el modelo
        joblib.dump(self.model, filepath)
        return filepath
    
    def load_model(self, filepath='model/hotel_cancellation_model.pkl'):
        """Carga un modelo previamente entrenado."""
        self.model = joblib.load(filepath)
        return self.model
    
    def predict_cancellation(self, input_data):
        """Realiza predicciones con el modelo entrenado."""
        if self.model is None:
            try:
                self.load_model()
            except:
                self.build_and_train_model()
        
        # Convertir a DataFrame si es un diccionario
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Realizar predicción
        prediction = self.model.predict(input_data)
        probability = self.model.predict_proba(input_data)[:, 1]
        
        return {'prediction': prediction[0], 'probability': probability[0]}
    
    def process_and_prepare_all(self):
        """Ejecuta todo el pipeline y devuelve resultados relevantes."""
        self.load_data()
        self.preprocess_data()
        self.build_and_train_model()
        metrics = self.evaluate_model()
        feature_importance = self.extract_feature_importance()
        insights = self.generate_key_insights()
        
        # Guardar modelo
        model_path = self.save_model()
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'insights': insights,
            'model_path': model_path
        }


# Ejecutar si se llama como script principal
if __name__ == "__main__":
    processor = HotelDataProcessor()
    results = processor.process_and_prepare_all()
    print(f"Precisión del modelo: {results['metrics']['accuracy']:.4f}")
    print(f"Modelo guardado en: {results['model_path']}")