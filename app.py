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

    return "Drinkable" if prediction == 1 else "Not Drinkable"

# Gradio Interface
app = gr.Interface(
    fn = water_quality_predict,
    inputs=[
        gr.Number(label="PH", value= 7),
        gr.Number(label="Hardness", value = 236),
        gr.Number(label="Solids", value = 14245),
        gr.Number(label="Chloramines", value = 6),
        gr.Number(label="Sulfate", value = 373),
        gr.Number(label="Conductivity", value = 416),
        gr.Number(label="Organic Carbon", value = 10),
        gr.Number(label="Trihalomethanes", value = 85),
        gr.Number(label="Turbidity", value = 2)
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Water Quality Prediction App",
    description="Enter water-related components to predict the outcome using our trained model."
)

# Launch the app
app.launch()