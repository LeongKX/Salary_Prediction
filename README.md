# Data Science Salary Predictor

Predicts a data scientist's annual salary (in USD) from job and company details,
with a simple web app built using Gradio.

## What it does
You pick things like job title, experience level, company size and location, and
the model estimates the salary along with a realistic range.

## Files
| File | What it is |
|------|------------|
| `train.py` | Trains the model and saves `salary_model.joblib` |
| `app.py` | Runs the Gradio web app |
| `salary_prediction.ipynb` | Notebook version (data exploration + model) |
| `Latest_Data_Science_Salaries.csv` | The dataset |
| `salary_model.joblib` | The trained model |
| `requirements.txt` | Python packages needed |

## How to run
```bash
pip install -r requirements.txt
python train.py     # trains and saves the model
python app.py       # starts the web app
```
Then open the link it prints (http://127.0.0.1:7860) in your browser.

## The model
- **Inputs:** Job Title, Employment Type, Experience Level, Expertise Level,
  Company Location, Company Size, Employee Residence, Year
- **Encoding:** One-Hot encoding inside a scikit-learn Pipeline
- **Algorithm:** Random Forest
- **Accuracy:** R² ≈ 0.45, average error ≈ $38k (median salary ≈ $136k)

## Note on the numbers
The dataset is mostly US jobs priced in USD, so predictions reflect US-level pay,
not local salaries. Salary is also naturally noisy, so the app shows a **range**
rather than a single exact figure.
