# Dissertation Project - Enhancing Business Strategies Using LLM Based AI Model

This project implements a dual-component AI system combining sentiment analysis using a fine-tuned DistilGPT-2 model with LoRA and sales forecasting using a multi-store LSTM model. Results are presented in a user-friendly Streamlit dashboard, enabling real-time strategic decision-making in retail settings.

# Project Overview

Objective: To integrate unstructured customer feedback and structured sales data for proactive, data-driven business decisions.
Core Components:

  Sentiment Classification using DistilGPT-2 + LoRA
  Sales Forecasting using LSTM for multi-store data
  Interactive Dashboard built with Streamlit

# Tools and Technologies

| Technology               | Purpose                            |
| ------------------------ | ---------------------------------- |
| Python 3.9               | Core scripting                     |
| Google Colab             | Model training with T4 GPU         |
| PyTorch                  | Deep learning framework            |
| HuggingFace Transformers | Pre-trained LLM integration        |
| `peft`                   | LoRA fine-tuning                   |
| Scikit-learn             | Preprocessing, metrics             |
| Plotly                   | Interactive graphs                 |
| Streamlit                | Dashboard deployment               |
| Ngrok                    | Public URL tunneling for dashboard |


# Project Structure


├── app.py                      # Sentiment analysis pipeline
├── sales.py                    # LSTM forecasting pipeline
├── dashboard.py                # Streamlit UI for both models
├── model.py                    # LoRA-enhanced DistilGPT-2
├── final_model/                # Saved DistilGPT-2 model
├── sales_forecaster_multi.pt   # Saved LSTM model
├── processed_data_large.csv    # Preprocessed sentiment dataset
├── sales_data.csv              # Structured sales + economic data
├── your_reviews.csv            # Synthetic test reviews
├── README.md                   # Project documentation


# How to Run the Project

Step 1: Environment Setup (Google Colab Recommended)

Open [Google Colab](https://colab.research.google.com)
Install required libraries:

!pip install transformers datasets peft
!pip install transformers datasets peft streamlit plotly scikit-learn --quiet


Step 2: Sentiment Analysis (app.py)

Input: `amazon.csv` → millions of customer reviews
Steps:

  * Clean and preprocess text (remove URLs, special characters)
  * Convert star ratings to sentiment labels (positive/neutral/negative)
  * Output: `processed_data_large.csv`

Step 3: Fine-Tune DistilGPT-2 with LoRA

python
from transformers import Trainer, TrainingArguments
from peft import get_peft_model

 -- Load pre-trained DistilGPT-2 and attach LoRA adapters

 -- Train using HuggingFace Trainer API with sentiment-labeled dataset

 -- Save model to `final_model/`

Step 4: Sales Forecasting (sales.py)

 -- Input: 'sales_data.csv' (weekly sales, CPI, fuel, etc.)

    Steps:

  * One-hot encode store IDs
  * Normalize numerical features using MinMaxScaler
  * Window sales data into 4-week input sequences
  * Train LSTM for 5 epochs using MSE loss

 -- Save model as `sales_forecaster_multi.pt`

Step 5: Run Dashboard (dashboard.py)
-------
from pyngrok import ngrok
import subprocess
import nest_asyncio
import time

# Required to allow nested asyncio loops (for Google Colab, etc.)
nest_asyncio.apply()

# Authenticate ngrok
!ngrok authtoken 2wDplgF0KhYmJcS2dOI83dynPHE_5nDg7ek2tS7MiCQMHw8YJ

# Kill any previous tunnels
ngrok.kill()

# Start the Streamlit app (non-blocking)
streamlit_cmd = "streamlit run /content/drive/MyDrive/BuisinessStrat/dashboard.py --server.port 8501"
process = subprocess.Popen(streamlit_cmd, shell=True)

# Wait a few seconds to ensure the server starts
time.sleep(10)

# Start ngrok tunnel after Streamlit is running
public_url = ngrok.connect(8501, proto="http", bind_tls=True)
print(f"Streamlit app is live at: {public_url}")

---------

 -- Upload or use default datasets (`processed_data_large.csv` and `sales_data.csv`)
    Explore:

    📊 Sentiment distributions (positive, neutral, negative)
    📈 Sales forecast per store
    🔄 Correlation insights (sentiment vs. sales trends)


📈 Evaluation Metrics

| Model                | Metric              | Description                           |
| -------------------- | ------------------- | ------------------------------------- |
| Sentiment Classifier | Accuracy / F1 Score | Over 85% accuracy observed            |
| LSTM Forecasting     | RMSE / MAE          | Captures seasonal trends              |
| Combined Insight     | Pearson Correlation | \~0.2–0.3 sentiment-sales correlation |


✅ Key Features

  Real-time, web-based dashboard
  Rule-based business recommendations (e.g., inventory prompts, sentiment alerts)
  Parameter-efficient fine-tuning (LoRA)
  Multimodal insight: qualitative + quantitative

🔐 Ethics & Data Handling

  Weak supervision used (heuristic sentiment labeling)
  No PII used; all datasets are public or anonymized
  Conforms to ethical AI practices and reproducibility standards

📦 Sample Data Sources

  [Amazon Reviews Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
  [Walmart Store Sales Data (Kaggle)](https://www.kaggle.com/code/msjahid/walmart-sales-exploration/input)


📌 Future Improvements

  Real-time streaming support
  More granular sentiment categories
  Transformer-based time-series forecasters (e.g., Informer, TFT)
  Integration with CRM or ERP systems
