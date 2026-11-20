# SentinelAI 🛡️

### Real-Time Threat Detection and Indirect Prompt Injection Mitigation in Autonomous AI Agents

SentinelAI is an AI-powered security application designed to identify potentially malicious prompts and detect prompt injection attempts targeting AI agents.

It uses a fine-tuned **RoBERTa transformer model** to classify input text as benign or malicious. The application combines a machine learning backend with a React-based dashboard to provide real-time threat analysis, classification, and monitoring.

---

## Features

- **AI-Powered Threat Detection:** Uses a fine-tuned RoBERTa model for binary text classification.
- **Real-Time Analysis:** Analyzes submitted text through a FastAPI backend.
- **Threat Classification:** Identifies inputs as `BENIGN` or `MALICIOUS`.
- **Probability Scores:** Displays benign and malicious confidence scores.
- **Security Dashboard:** Provides a visual interface for monitoring analysis results.
- **Analysis History:** Stores analysis records using SQLite.
- **Configurable Classification Threshold:** Supports threshold-based classification.
- **Research and Evaluation:** Includes model evaluation, threshold analysis, false-positive analysis, and augmentation experiments.

---

## Technology Stack

### Machine Learning
- Python
- PyTorch
- Hugging Face Transformers
- RoBERTa
- Scikit-learn
- Pandas
- NumPy

### Backend
- FastAPI
- Uvicorn
- SQLite
- Pydantic

### Frontend
- React
- Vite
- JavaScript
- CSS
- Lucide React
- jsPDF

---

## Project Structure

```text
SentinelAI/
│
├── backend/
│   ├── app/
│   │   ├── database.py
│   │   ├── detector.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── security.py
│   │
│   ├── ml/
│   │   ├── train.py
│   │   ├── roberta_model.py
│   │   ├── evaluate_roberta.py
│   │   ├── analyze_thresholds.py
│   │   └── ...
│   │
│   ├── data/
│   │   ├── augmentation/
│   │   ├── evaluation/
│   │   └── processed/
│   │
│   ├── models/
│   ├── collect_dataset.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── research/
│   ├── augmentation_results/
│   ├── dataset_analysis/
│   ├── evaluation_results/
│   ├── model_comparison/
│   └── transformer_results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

SentinelAI uses the **Prompt Injection Detection Dataset** published by S-Labs on Hugging Face.

Dataset: <https://huggingface.co/datasets/S-Labs/prompt-injection-dataset>

### Dataset Information

| Property | Details |
|---|---|
| Publisher | S-Labs |
| Platform | Hugging Face |
| Dataset ID | `S-Labs/prompt-injection-dataset` |
| License | MIT (as declared by the publisher) |
| Input Feature | `text` |
| Target Feature | `label` |
| Benign Label | 0 |
| Injection Label | 1 |

### Dataset Splits

| Split | Samples |
|---|---:|
| Training | 11,089 |
| Validation | 2,101 |
| Testing | 2,101 |
| **Total** | **15,291** |

The original dataset is not included in this repository.

To download the dataset, run the collection script from the backend directory:

```powershell
cd backend
python collect_dataset.py
```

The downloaded CSV files will be saved under `backend/data/raw/`.

The project also includes separately developed synthetic augmentation and evaluation samples.

### Dataset Citation

S-Labs. (2026). *Prompt Injection Detection Dataset*. Hugging Face.

<https://huggingface.co/datasets/S-Labs/prompt-injection-dataset>

---

## Model Performance

The primary RoBERTa model was evaluated on a held-out test dataset containing 2,101 samples.

The classification threshold of **0.20** was selected through validation-based threshold analysis and subsequently evaluated on the test dataset.

### Primary RoBERTa Results

| Metric | Result |
|---|---:|
| Accuracy | 96.76% |
| Precision | 97.12% |
| Recall | 96.38% |
| F1-Score | 96.75% |

### Model Improvement Experiments

An augmented RoBERTa model was also developed using additional hard-negative training samples.

The experiments investigated the trade-off between reducing false positives and maintaining malicious prompt detection recall.

The augmented model reduced false positives on the original test dataset but also increased false negatives.

Therefore, the original RoBERTa model remains the primary model used by SentinelAI.

**Note:** These results are based on the project's evaluation datasets and should not be interpreted as guaranteed real-world detection performance.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SentinelAI.git
cd SentinelAI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Python Environment

```powershell
python -m venv .venv
```

Activate it using Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 4. Obtain the Trained Model

The trained RoBERTa model weights are not included in this repository because of their large file size.

Place the compatible trained model and tokenizer files inside:

```text
backend/models/roberta/
```

The directory must contain the model configuration, tokenizer files, and trained model weights required by the detector.

### 5. Start the Backend

From the `backend` directory:

```powershell
python -m uvicorn app.main:app --reload
```

Backend API:

<http://127.0.0.1:8000>

Interactive API documentation:

<http://127.0.0.1:8000/docs>

### 6. Start the Frontend

Open a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend will normally be available at:

<http://localhost:5173>

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend health |
| POST | `/analyze` | Analyzes submitted text |
| GET | `/logs/{analysis_id}` | Retrieves a stored analysis record |

---

## Research and Evaluation

The `research/` directory contains experimental outputs and evaluation artifacts, including:

- Dataset exploratory data analysis.
- Model performance comparisons.
- RoBERTa and DistilBERT evaluation results.
- Confusion matrices.
- Classification threshold analysis.
- Contextual false-positive analysis.
- Hard-negative augmentation experiments.
- SHAP explainability results.

These experiments support the development and evaluation of the SentinelAI detection pipeline.

---

## Limitations

- Detection performance depends on the quality and diversity of training data.
- The model performs text classification and does not guarantee detection of every prompt injection attack.
- Evaluation results may not fully represent real-world autonomous agent environments.
- The trained model weights must be obtained separately.
- Text classification alone cannot provide complete protection against indirect prompt injection.
- This project is a research and demonstration system, not a replacement for comprehensive AI agent security controls.

---

## Future Improvements

- Expand the dataset with more diverse indirect prompt injection examples.
- Evaluate against additional real-world attack patterns.
- Integrate contextual security analysis.
- Improve detection of obfuscated and multi-stage attacks.
- Explore integration with autonomous AI agent workflows.
- Investigate additional defense mechanisms beyond text classification.

---

## Disclaimer

SentinelAI is developed for academic research, experimentation, and educational purposes.

It is not intended to provide guaranteed protection against all prompt injection attacks or replace comprehensive security mechanisms.

---

## Author

Developed as an AI and Machine Learning security project.