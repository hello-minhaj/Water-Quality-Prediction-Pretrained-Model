import gradio as gr
import pandas as pd
import pickle

# Load the trained model
with open("water_quality.pkl", "rb") as file:
    model = pickle.load(file)

# Prediction function
def water_quality_predict(
    ph, Hardness, Solids, Chloramines,
    Sulfate, Conductivity, Organic_carbon,
    Trihalomethanes, Turbidity
):
    # Validation
    if not (0 <= ph <= 14):
        return "⚠️ Invalid input: pH must be between 0 and 14"

    if Turbidity < 0:
        return "⚠️ Invalid input: Turbidity cannot be negative"

    input_df = pd.DataFrame([[
    ph, Hardness, Solids, Chloramines,
    Sulfate, Conductivity, Organic_carbon,
    Trihalomethanes, Turbidity
    ]], columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines',
        'Sulfate', 'Conductivity', 'Organic_carbon', 
        'Trihalomethanes', 'Turbidity'
    ])

    prediction = model.predict(input_df)[0]

    return "🟢 Drinkable" if prediction == 1 else "🔴 Not Drinkable"

# Gradio Interface
app = gr.Interface(
    fn=water_quality_predict,
    inputs=[
        gr.Number(label="pH", value=7, minimum=0, maximum=14, info="Physical range: 0–14 | WHO recommended: 6.5–8.5"),
        gr.Number(label="Hardness (mg/L)", value=236, info="Calcium & magnesium salts | Practical upper bound ≈ 2000 mg/L (very hard water)"),
        gr.Number(label="Total Dissolved Solids – TDS (mg/L)", value=500, info="Mineral content in water | Desirable ≤ 500 mg/L | Practical upper bound ≈ 2000 mg/L"),
        gr.Number(label="Chloramines (mg/L)", value=2, info="Disinfectant level | Safe limit ≤ 4 mg/L | Values >10 mg/L are unrealistic"),
        gr.Number(label="Sulfate (mg/L)", value=250, info="Typical freshwater < 500 mg/L | Upper bound ≈ 2000 mg/L"),
        gr.Number(label="Conductivity (µS/cm)", value=400, info="Electrical conductivity | WHO guideline ≤ 400 µS/cm | Practical upper bound ≈ 5000 µS/cm"),
        gr.Number(label="Organic Carbon – TOC (mg/L)", value=2, info="Drinking water usually < 5 mg/L | Upper bound ≈ 20 mg/L"),
        gr.Number(label="Trihalomethanes – THMs (µg/L)", value=80, info="EPA limit ≤ 80 µg/L | Practical upper bound ≈ 200 µg/L"),
        gr.Number(label="Turbidity (NTU)", value=1, info="Cloudiness of water | WHO limit ≤ 5 NTU | Values >100 NTU indicate extreme contamination")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="💧 Water Quality Prediction App",
    description= "Enter standard water quality parameters based on WHO/EPA guidelines. "
        "The model predicts whether the water is safe for drinking."

)

# Launch the app
app.launch()



