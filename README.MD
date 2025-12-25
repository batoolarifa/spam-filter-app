# 🚨 Spam Email Classification System (ML + Gradio)

An end-to-end **Spam Email Classification** project built using **Machine Learning**, following a **modular, production-ready architecture**, and deployed with an interactive **Gradio UI**.

This system classifies emails as **Spam** or **Not Spam** using **TF-IDF feature extraction** and a **Support Vector Machine (SVM)** classifier, prioritizing **high precision** to reduce false positives.



## 📌 Project Overview

Spam emails often contain promotions, scams, or malicious content. Manual filtering is inefficient and error-prone.  
This project automates spam detection by leveraging **Natural Language Processing (NLP)** and **Machine Learning**, providing a reliable and scalable solution.



## 🎯 Objectives

- Clean and preprocess raw email text
- Extract meaningful textual features
- Train and compare multiple ML models
- Evaluate performance using standard classification metrics
- Select the best-performing model
- Deploy the model with a user-friendly web interface



## 📂 Dataset

- **Source:** Kaggle – Spam Email Dataset  
  https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data

- **Columns:**
  - `text` → Email content
  - `spam` → Target label  
    - `1` = Spam  
    - `0` = Not Spam

The dataset contains a mix of promotional, scam, and legitimate emails.


## 🔄 Project Workflow

### 1️⃣ Data Understanding
- Loaded and inspected dataset structure
- Checked shape, missing values, and duplicates
- Reviewed sample emails for context



### 2️⃣ Text Preprocessing
Applied NLP techniques to clean and normalize text:
- Lowercasing
- Removing special characters and punctuation
- Tokenization
- Stopword removal
- Lemmatization

This ensured consistent and noise-free input for modeling.



### 3️⃣ Exploratory Data Analysis (EDA)
- Analyzed class distribution (Spam vs Not Spam)
- Studied email length (words & characters)
- Identified frequent words in spam and non-spam emails
- Visualized patterns to understand data behavior



### 4️⃣ Feature Engineering
- Generated numerical features:
  - Word count
  - Character count
- Compared feature distributions between spam and ham emails



### 5️⃣ Model Building
Text was vectorized using:
- **Bag of Words (BoW)**
- **TF-IDF**
- **TF-IDF (1–2 grams)**

Models trained and evaluated:
- Naive Bayes (Multinomial, Bernoulli, Gaussian)
- Random Forest
- Extra Trees
- **Linear Support Vector Machine (SVM)**

Dense conversion was applied where required.



### 6️⃣ Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

📌 **Precision was prioritized** to minimize false spam detection (false positives).



### 7️⃣ Final Model Selection
- **TF-IDF + Linear SVM** delivered the best balance of performance and reliability
- Final model and vectorizer were saved using `pickle`



### 8️⃣ Prediction on New Emails
- New email text goes through the same preprocessing pipeline
- TF-IDF vectorization is applied
- Model predicts:
  - **Spam**
  - **Not Spam**



## 🧠 Project Architecture (Modular Design)

```

spam-filter-app/
│
├── app.py                  # Gradio application
├── utils/
│   ├── model_loader.py     # Loads trained model & vectorizer
│   ├── preprocessing.py   # Text cleaning & NLP pipeline
│   └── predict.py          # Prediction logic
│
├── saved_models/
│   ├── vectorizer_TF-IDF.pkl
│   └── SVM_TF-IDF.pkl
│
├── notebook/
│   └── spam_classification.ipynb  # Complete ML workflow
│
├── requirements.txt
└── README.md

```

✔ Clean separation of concerns  
✔ Reusable utility modules  
✔ Production-friendly structure  



## 🖥️ Web Application (Gradio)

- Interactive UI for email classification
- Input full email content
- One-click prediction
- Example emails included
- Clean, minimal interface

### ▶ Demo
[Live Demo 🔗](https://huggingface.co/spaces/arifa-batool/spam-filter-app)


![Spam Email](https://github.com/user-attachments/assets/1defefcc-c8fc-4576-94bf-ce5d3aa093dc)



![Not Spam Email](https://github.com/user-attachments/assets/152ea3af-bb33-47ed-9c6c-74c1a983b04e)




## ⚙️ Technologies Used

- **Python**
- **Scikit-learn**
- **NLTK**
- **Gradio**
- **Pandas & NumPy**
- **Pickle**
- **Jupyter Notebook**



## 📈 Results & Conclusion

- Successfully built a robust spam classification system
- Achieved strong precision, reducing false spam flags
- Modular architecture supports easy scaling and reuse
- UI enables real-world usability and testing

This project demonstrates **end-to-end ML development**, from data exploration to deployment.



## 🚀 Future Improvements

- Support batch email classification
- Deploy on cloud (Hugging Face / AWS / GCP)
- Add confidence scores for predictions



## 👤 Author

**Syeda Arifa Batool**  
SE @ Karachi University | AI & ML Practitioner | Applying technology to create real-world value 📈



## 🔗 Connect with Me

- **LinkedIn:** https://www.linkedin.com/in/arifa-batool/  
- **Kaggle:** https://www.linkedin.com/in/arifa-batool/  
- **Email:** thearifabatool@gmail.com


⭐ If you find this project useful, feel free to star the repository!
