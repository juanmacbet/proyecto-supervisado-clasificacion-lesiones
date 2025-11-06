import pickle
import numpy as np
import gradio as gr
import gdown

# --- Descargar modelo desde Google Drive ---
url = "https://drive.google.com/uc?export=download&id=1gBbozuuzCA81QbieOlYazhEOESnrFoRw"
output = "svm_model_lesiones.pkl"
gdown.download(url, output, quiet=False)

# --- Cargar modelo y preprocesadores ---
with open(output, "rb") as f:
    data_cargada = pickle.load(f)

modelo_cargado = data_cargada["modelo"]
ohe_cargado = data_cargada["encoder"]
scaler_cargado = data_cargada["scaler"]

# --- Columnas ---
categorical_cols = ["Position"]
numeric_cols = [
    "Age", "Height_cm", "Weight_kg", "Training_Hours_Per_Week", 
    "Matches_Played_Past_Season", "Previous_Injury_Count", "Knee_Strength_Score", 
    "Hamstring_Flexibility", "Reaction_Time_ms", "Balance_Test_Score", 
    "Sprint_Speed_10m_s", "Agility_Score", "Sleep_Hours_Per_Night", 
    "Stress_Level_Score", "Nutrition_Quality_Score", 
    "Warmup_Routine_Adherence", "BMI"
]

# --- Función de predicción ---
def predecir_lesion(
    Position,
    Age, Height_cm, Weight_kg, Training_Hours_Per_Week,
    Matches_Played_Past_Season, Previous_Injury_Count, Knee_Strength_Score,
    Hamstring_Flexibility, Reaction_Time_ms, Balance_Test_Score,
    Sprint_Speed_10m_s, Agility_Score, Sleep_Hours_Per_Night,
    Stress_Level_Score, Nutrition_Quality_Score, Warmup_Routine_Adherence, BMI
):
    # Construir arrays
    X_num = np.array([[Age, Height_cm, Weight_kg, Training_Hours_Per_Week,
                       Matches_Played_Past_Season, Previous_Injury_Count, Knee_Strength_Score,
                       Hamstring_Flexibility, Reaction_Time_ms, Balance_Test_Score,
                       Sprint_Speed_10m_s, Agility_Score, Sleep_Hours_Per_Night,
                       Stress_Level_Score, Nutrition_Quality_Score,
                       Warmup_Routine_Adherence, BMI]])
    X_cat = np.array([[Position]])
    
    # Transformar categóricas y escalar
    X_cat_enc = ohe_cargado.transform(X_cat)
    X_final = np.hstack([X_num, X_cat_enc])
    X_scaled = scaler_cargado.transform(X_final)
    
    # Predicción
    pred = modelo_cargado.predict(X_scaled)[0]
    
    # Resultado legible
    return "Lesión probable la próxima temporada" if pred == 1 else "Sin riesgo significativo de lesión"


demo = gr.Interface(
    fn=predecir_lesion,
    inputs=[
        gr.Dropdown(["Defender", "Midfielder", "Forward", "Goalkeeper"], label="Position"),
        gr.Number(label="Age", minimum=0, maximum=1000),
        gr.Number(label="Height (cm)", minimum=0, maximum=1000),
        gr.Number(label="Weight (kg)", minimum=0, maximum=1000),
        gr.Number(label="Training Hours Per Week", minimum=0, maximum=168),
        gr.Number(label="Matches Played Past Season", minimum=0, maximum=1000),
        gr.Number(label="Previous Injury Count", minimum=0, maximum=1000),
        gr.Number(label="Knee Strength Score", minimum=0, maximum=100),
        gr.Number(label="Hamstring Flexibility", minimum=0, maximum=100),
        gr.Number(label="Reaction Time (ms)", minimum=0, maximum=1000),
        gr.Number(label="Balance Test Score", minimum=0, maximum=100),
        gr.Number(label="Sprint Speed 10m (s)", minimum=0, maximum=1000),
        gr.Number(label="Agility Score", minimum=0, maximum=100),
        gr.Number(label="Sleep Hours Per Night", minimum=0, maximum=24),
        gr.Number(label="Stress Level Score", minimum=0, maximum=100),
        gr.Number(label="Nutrition Quality Score", minimum=0, maximum=100),
        gr.Number(label="Warmup Routine Adherence", minimum=0, maximum=1),
        gr.Number(label="BMI", minimum=0, maximum=100),
    ],
    outputs=gr.Textbox(label="Predicción del modelo"),
    title="Predicción de Lesiones Deportivas (SVM)",
    description="Introduce los datos del jugador y el modelo estimará si habrá lesión en la próxima temporada."
)

demo.launch()
