# 🎬 Netflix Movie Recommendation System

An AI-powered desktop application that recommends similar movies or TV shows based on user input. The system uses TF-IDF vectorization and cosine similarity to identify content similarity from the Netflix dataset, and displays movie posters fetched from the TMDb or OMDB APIs — all inside a polished Tkinter GUI.

---

## 🚀 Features

### ✅ **AI-Powered Movie Recommendation**

* Uses a **TF-IDF vectorizer** to convert movie descriptions, genres, cast, directors, and type into numerical vectors.
* Calculates similarity using **cosine similarity**.
* Returns the **top 5 most similar movies/shows**.

### 🎨 **Modern Tkinter GUI**

* Attractive Netflix-style UI.
* Search bar with placeholder behavior.
* Scrollable horizontal results section with movie cards.
* Poster display for each recommendation.

### 🖼️ **Poster Fetching (TMDb/OMDB)**

* Fetches high-quality posters from **TMDb API** (recommended).
* Fallback to **OMDB API** for quick testing.
* Auto-generated placeholder poster if none are found.

### 📦 **Dataset**

* Uses the **Netflix Titles dataset** from Kaggle.
* Script automatically loads and processes:

  * Title
  * Description
  * Cast
  * Director
  * Category (Series/Movie)
  * Genre

### ⚠️ **Smart Error Handling**

* Invalid movie? Shows warnings.
* Missing dataset? Shows clear error messages.
* Handles network issues gracefully.

---

## 🏗️ How It Works

### **1. Load the Dataset**

The Netflix dataset is read using Pandas, missing fields are filled, and a new field `combined_features` is constructed from relevant metadata.

### **2. Build the ML Model**

A **TfidfVectorizer** creates feature vectors from the combined text.
A similarity matrix is generated using **cosine_similarity**.

### **3. Recommend Movies**

Given a movie title:

* If exact match doesn’t exist, fuzzy matching is used.
* Top similar movies are selected.
* Posters are fetched from TMDb/OMDB.

### **4. Display Results**

Results appear as horizontal scrollable cards with:

* Poster
* Title
* Highlighted “⭐ Recommended” tag

---

## 🛠️ Dependencies

### Python Libraries

* tkinter
* PIL (Pillow)
* pandas
* numpy
* scikit-learn
* requests

### APIs

* **TMDb API** (recommended)
* **OMDB API** (optional)

---

## 📁 Required Files

* `netflix_titles.csv` (Kaggle dataset)
* `background.jpg` (optional UI background)

---

## ▶️ Running the Application

1. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn pillow requests
   ```

2. Update the dataset path inside the script:

   ```python
   DATASET_PATH = "path/to/netflix_titles.csv"
   ```

3. Add your TMDb API key:

   ```python
   TMDB_API_KEY = "your_api_key"
   ```

4. Run the script:

   ```bash
   python Movie_recommendation_project.py
   ```

5. Enjoy personalized movie recommendations!

---

## 📸 Screenshots (Optional)

You can add screenshots here if desired:

```
/screenshots/
    ui_home.png
    recommendations.png
```

---

## 🌟 Future Enhancements

* Add trailer previews
* Add rating/year extraction
* Improve fuzzy matching with NLP
* Include collaborative filtering
* Add user profiles

---

If you'd like, I can also generate a **README.md file** version you can download, or customize it to make it more professional, fun, or visually appealing!
