# 📧 Spam Detection using Machine Learning

This project aims to build a **Spam Detection System** using multiple **Machine Learning models**. It classifies messages as either **Spam** or **Not Spam (Ham)** based on text features. The models are trained and evaluated using a combination of different datasets and vectorization techniques to maximize accuracy and performance.

---

## 📋 Project Overview

In this project:

* We **trained, tested, and compared** several Machine Learning algorithms for the spam classification task.
* **TF-IDF Vectorizer** was used to transform the raw text data into numerical features.
* Models were trained and saved as `.pkl` files for easy deployment in any production environment.

---

## 📂 Project Structure

| File/Folder                     | Description                                                                                                             |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------- |
| **Training\_data.csv**          | The main dataset used for training different ML models.                                                                 |
| **Independent\_dataset.csv**    | Separate dataset used for testing models independently to evaluate generalization.                                      |
| **Training\_the\_models.ipynb** | Jupyter Notebook where multiple ML models (SVM, Random Forest, Decision Tree, Naive Bayes, KNN) were trained and saved. |
| **Independent\_testing.ipynb**  | Notebook to test the saved models against the independent dataset and check their performance.                          |
| **decision\_tree\_model.pkl**   | Pre-trained Decision Tree model saved using pickle for deployment.                                                      |
| **knn\_model.pkl**              | Pre-trained K-Nearest Neighbors (KNN) model.                                                                            |
| **naive\_bayes\_model.pkl**     | Pre-trained Naive Bayes model for spam detection.                                                                       |
| **rf\_model.pkl**               | Pre-trained Random Forest model.                                                                                        |
| **svm\_model.pkl**              | Pre-trained Support Vector Machine (SVM) model, usually the best-performing model for text classification tasks.        |
| **tfidf\_vectorizer.pkl**       | Saved TF-IDF Vectorizer, used to transform incoming text into model-acceptable input during deployment.                 |
| **README.md**                   | Documentation describing the project, approach, and structure.                                                          |
| **\_Store**                     | Meta or checkpoint storage (GitHub internal system folder, usually can be ignored).                                     |

---

## 🚀 Key Features

* Multiple ML models trained and compared.
* Separate independent dataset testing.
* Ready-to-deploy `.pkl` model files.
* TF-IDF feature extraction used for text preprocessing.
* Easy integration into any backend or web application for real-time spam detection.

---

## 📈 Models Used

* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Naive Bayes
* K-Nearest Neighbors (KNN)

Each model was trained on labeled text data, optimized, and saved for future inference.

---

## 🛠️ Future Improvements

* Add Deep Learning models like LSTM for better performance.
* Create a Flask or FastAPI app for real-time spam prediction.
* Automate model selection based on accuracy during inference.

---

Would you also like me to give a small **"How to Run"** guide if someone wants to use your `.pkl` models easily? 🚀
(Example: how to load `svm_model.pkl` and `tfidf_vectorizer.pkl` and predict spam?)
I can add that if you want!
