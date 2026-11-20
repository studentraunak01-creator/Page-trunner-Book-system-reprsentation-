# 📚 PageTurnner – Book Recommendation System

> Discover your next favourite book with intelligent book recommendations.

## 📖 About the Project

**PageTurnner** is a book recommendation system developed using Python and Machine Learning techniques. It helps users discover books similar to the books they already enjoy.

The system uses **Content-Based Filtering**, **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity** to analyse book information and recommend similar books.

This project aims to make book discovery easier, more interactive and enjoyable.

## ✨ Features

- 📚 Book recommendation based on book similarity.
- 🔍 Discover books similar to a selected book.
- 🤖 Content-Based Recommendation approach.
- 🧠 TF-IDF for text feature representation.
- 📊 Cosine Similarity for comparing books.
- 🖥️ Interactive web application built with Streamlit.
- 📁 Utilises the Goodbooks-10k dataset.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Pandas | Data handling and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and similarity calculations |
| TF-IDF | Converts text information into numerical features |
| Cosine Similarity | Measures similarity between book representations |

## 🧠 Recommendation Methodology

PageTurnner uses a Content-Based Filtering approach.

### 1. Data Preparation
Book information is loaded and prepared for the recommendation process.

### 2. Feature Extraction
TF-IDF converts relevant text information about books into numerical vectors.

### 3. Similarity Calculation
Cosine Similarity compares the numerical representations of books to measure how similar they are.

### 4. Recommendation
The system identifies books with similar content features and presents recommendations to the user.

## 🏗️ Project Structure

```text
PageTurnner/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── recommender.py
│
├── data/
│   ├── raw/
│   └── processed/
```

## 📊 Dataset

This project uses the **Goodbooks-10k dataset**, which contains information about approximately 10,000 books.

The dataset provides book-related information used for analysing and recommending similar books.

Dataset source:

[Goodbooks-10k Dataset – GitHub](https://github.com/zygmuntz/goodbooks-10k)

The original dataset files are not included in this repository.

## 🚀 Installation and Setup

Follow these steps to run PageTurnner on your computer.

### Step 1: Clone the repository

```bash
git clone https://github.com/TheDataSprint/PageTurnner-Book-Recommendation-System
```

### Step 2: Navigate to the project folder

```bash
cd PageTurnner
```

Use the actual repository folder name if it differs.

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

**Linux / Ubuntu:**

```bash
source .venv/bin/activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Download and prepare the dataset

Download the required Goodbooks-10k dataset files from the dataset source and place the required files in the appropriate `data/raw/` directory.

Follow the file requirements specified in the project's data loader and preprocessing code.

### Step 7: Run the application

```bash
python -m streamlit run app.py
```

### Step 8: Open the application

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use PageTurnner.

## 🔮 Future Scope

- Personalised recommendations based on user preferences.
- Hybrid recommendation systems combining different approaches.
- Improved search and filtering functionality.
- Integration of additional book datasets.
- Enhanced recommendation diversity.

## 🎓 Project Information

- **Project Name:** PageTurnner – Book Recommendation System
- **Domain:** Machine Learning / Data Science
- **Programming Language:** Python
- **Application Framework:** Streamlit
- **Recommendation Technique:** Content-Based Filtering

Developed as an academic project for Bachelor of Technology in Computer Science and Engineering.

## 📄 License

This project is distributed under the license included in this repository.

---

<p align="center">
  Made with ❤️ and Python for book lovers.
</p>
