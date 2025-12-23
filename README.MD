
# 🚨 Spam Email Classifier
🔗 Live Demo: [Add your deployed app link here]

---

## 📌 Description
A web-based application to classify emails as **Spam** or **Not Spam (Ham)** using **TF-IDF vectorization** and a **Support Vector Machine (SVM)** model. Users can quickly detect spam emails with a clean, interactive interface.

---

![Spam Email Classifier UI](images/spam_email_demo.gif)
*Example: Classifying Spam and Not Spam emails*



## 🚀 Features
- 📊 **Data Preprocessing**: Tokenization, stopword removal, punctuation removal, lemmatization using NLTK.
- 🤖 **Machine Learning Model**: TF-IDF vectorization + SVM.
- 🌐 **Interactive Web App**: Built with Gradio for a user-friendly interface.
- 🕙 **Real-Time Predictions**: Paste emails and get instant results.
- 🖼 **Visual Feedback**: UI shows predictions for Spam and Not Spam emails.

---

## 📂 Dataset
- **Type**: Email samples labeled as Spam or Not Spam.
- **Contents**: Spam emails (ads, phishing, promotions) and normal emails (personal/professional emails).
- **Preprocessing**: Lowercasing, punctuation removal, tokenization, stopword removal, lemmatization.

---

## 🛠 Models Used
- ✅ Support Vector Machine (SVM) with TF-IDF vectorization

---

## 📊 Results & Evaluation
- Predicts whether a given email is Spam or Not Spam.
- Including **subject + body** improves accuracy.

---

## 🖼️ Additional UI Screenshots

**Spam Email Example**  
![Spam Email Prediction](images/spam_email.png)  
*Prediction: Spam*

**Not Spam Email Example**  
![Not Spam Email Prediction](images/not_spam_email.png)  
*Prediction: Not Spam*


---

## ⚡ Usage
1️⃣ **Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```


2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```


3️⃣ **Launch the Gradio app**

```bash
pip install -r requirements.txt
```


1️⃣ **Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```
