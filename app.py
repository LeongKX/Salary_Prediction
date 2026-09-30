"""
Gradio app for the salary predictor.

Loads the model + metadata saved by train.py and serves a simple GUI.
The dropdown choices come straight from the training data, and the inputs
are fed to the model as a DataFrame with the exact same columns used in
training (so there is no risk of columns being in the wrong order).

Run:  python app.py     (then open the printed local URL)
First run train.py once to create salary_model.joblib.
"""

import os

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = "salary_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise SystemExit("salary_model.joblib not found. Run `python train.py` first.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]
choices = artifact["choices"]
metrics = artifact["metrics"]

def predict_salary(
    job_title,
    experience_level,
    expertise_level,
    employment_type,
    company_size,
    company_location,
    employee_residence,
    year,
):
    # Build a one-row DataFrame keyed by feature name -> no column-order bugs.
    row = {
        "Job Title": job_title,
        "Employment Type": employment_type,
        "Experience Level": experience_level,
        "Expertise Level": expertise_level,
        "Company Location": company_location,
        "Company Size": company_size,
        "Employee Residence": employee_residence,
        "Year": int(year),
    }
    input_df = pd.DataFrame([row])[features]

    usd = float(model.predict(input_df)[0])

    # A prediction interval is more honest than a single number: the model's
    # typical error (MAE) gives a sensible +/- band.
    mae = metrics["mae"]
    low, high = max(0.0, usd - mae), usd + mae

    lines = [
        "Estimated annual salary (USD)",
        "",
        f"  ${usd:,.0f}",
        f"  Likely range: ${low:,.0f} - ${high:,.0f}",
        "",
        "Note: this dataset is mostly US jobs priced in USD, so figures reflect",
        f"global (US-weighted) pay, not local wages. Typical error is ~${mae:,.0f}.",
    ]
    return "\n".join(lines)


with gr.Blocks(title="Data Science Salary Predictor") as demo:
    gr.Markdown("# Data Science Salary Predictor")
    gr.Markdown(
        f"Trained on the Latest Data Science Salaries dataset. "
        f"Test R² = {metrics['r2']:.2f}, average error ≈ ${metrics['mae']:,.0f}."
    )

    with gr.Row():
        with gr.Column():
            job_title = gr.Dropdown(choices["Job Title"], label="Job Title", value="Data Scientist")
            experience_level = gr.Radio(
                choices["Experience Level"], label="Experience Level", value="Senior"
            )
            expertise_level = gr.Radio(
                choices["Expertise Level"], label="Expertise Level", value="Expert"
            )
            employment_type = gr.Radio(
                choices["Employment Type"], label="Employment Type", value="Full-Time"
            )
        with gr.Column():
            company_size = gr.Radio(choices["Company Size"], label="Company Size", value="Medium")
            company_location = gr.Dropdown(
                choices["Company Location"], label="Company Location", value="United States"
            )
            employee_residence = gr.Dropdown(
                choices["Employee Residence"], label="Employee Residence", value="United States"
            )
            year = gr.Dropdown(
                choices["Year"], label="Year", value=max(choices["Year"])
            )

    predict_button = gr.Button("Predict Salary", variant="primary")
    result_output = gr.Textbox(label="Prediction", lines=8)

    predict_button.click(
        fn=predict_salary,
        inputs=[
            job_title,
            experience_level,
            expertise_level,
            employment_type,
            company_size,
            company_location,
            employee_residence,
            year,
        ],
        outputs=result_output,
    )


if __name__ == "__main__":
    demo.launch()
