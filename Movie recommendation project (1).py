import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ================= Configuration =================
DATASET_PATH =  r"C:\Users\rokay\Downloads\netflix_titles.csv"  # Update this path if needed

# TMDb API Configuration (RECOMMENDED - Better than OMDB)
# Get FREE API key from: https://www.themoviedb.org/settings/api
# Steps:
# 1. Sign up at themoviedb.org
# 2. Go to Settings > API
# 3. Request API key (choose Developer)
# 4. Paste your key below and set USE_OMDB = False
USE_OMDB = False  # Now using TMDb!
TMDB_API_KEY = "6ad9a6d3566dde2adf1c241c5d839044"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# OMDB API (for quick testing - may have network issues)
OMDB_API_KEY = "6ad9a6d3566dde2adf1c241c5d839044"

# ================= Load and Prepare Data =================
print("[INFO] Loading Netflix dataset...")
try:
    df = pd.read_csv(DATASET_PATH)
    df = df.fillna("")
    
    # Combine features for better recommendations
    df["combined_features"] = (
        df["description"] + " " +
        df["listed_in"] + " " +
        df["cast"] + " " +
        df["director"] + " " +
        df["type"]
    )
    
    print(f"[SUCCESS] Loaded {len(df)} movies/shows")
    
    # Build TF-IDF matrix
    print("[INFO] Building recommendation model...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    print("[SUCCESS] Model ready!")
    
    MODEL_LOADED = True
except FileNotFoundError:
    print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
    print("Please download from: https://www.kaggle.com/datasets/shivamb/netflix-shows")
    MODEL_LOADED = False
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    MODEL_LOADED = False

# ================= Recommendation Function =================
def recommend_movies(movie_title, top_n=5):
    """Get movie recommendations based on input movie title"""
    if not MODEL_LOADED:
        return ["Model not loaded. Please check dataset path."]
    
    # Check if movie exists in dataset
    if movie_title not in df["title"].values:
        # Try fuzzy matching
        matches = df[df["title"].str.contains(movie_title, case=False, na=False)]
        if len(matches) == 0:
            return [f"'{movie_title}' not found in dataset"]
        else:
            # Use the first match
            movie_title = matches.iloc[0]["title"]
            print(f"[INFO] Using closest match: {movie_title}")
    
    # Get movie index
    idx = df.index[df["title"] == movie_title][0]
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )[0]
    
    # Get top N similar movies (excluding the input movie itself)
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    # Return list of recommended movie titles
    recommendations = df["title"].iloc[similar_indices].tolist()
    
    return recommendations

# ================= UI Setup =================
root = tk.Tk()
root.title("🎬 Netflix Movie Recommendation System")
root.geometry("1000x650")
root.resizable(False, False)
root.configure(bg="#6a040f")  # Netflix dark background

# ================= Background =================
try:
    bg_image = Image.open(r"C:\Users\rokay\Downloads\background.jpg")
    bg_image = bg_image.resize((1000, 650))
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except:
    # Netflix-style gradient background
    bg_label = tk.Label(root, bg="#6a040f")
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_photo = None

# ================= Title =================
# Netflix logo style banner
title_frame = tk.Frame(bg_label, bg="#6a040f", bd=0)
title_frame.place(relx=0.5, y=60, anchor="center")

tk.Label(
    title_frame,
    text="🎬 NETFLIX",
    font=("Arial Black", 28, "bold"),
    fg="white",
    bg="#6a040f",
    padx=30,
    pady=15
).pack()

# Subtitle with glow effect
subtitle_frame = tk.Frame(bg_label, bg="#000000", bd=0)
subtitle_frame.place(relx=0.5, y=120, anchor="center")

tk.Label(
    subtitle_frame,
    text="AI-Powered Movie Recommendation System",
    font=("Segoe UI", 14, "italic"),
    fg="#ffffff",
    bg="#000000",
    padx=25,
    pady=8
).pack()

# Status indicator with better styling
status_frame = tk.Frame(bg_label, bg="#1a1a1a", bd=0)
status_frame.place(relx=0.5, y=165, anchor="center")

status_icon = "✓" if MODEL_LOADED else "✗"
status_color = "#00e676" if MODEL_LOADED else "#6a040f"
status_text = f"{status_icon} {len(df)} Movies & Shows Ready" if MODEL_LOADED else f"{status_icon} Model Not Loaded"

tk.Label(
    status_frame,
    text=status_text,
    font=("Segoe UI", 10, "bold"),
    fg=status_color,
    bg="#1a1a1a",
    padx=15,
    pady=5
).pack()

# ================= Entry =================
# Search container
search_frame = tk.Frame(bg_label, bg="#2a2a2a", bd=0)
search_frame.place(relx=0.5, y=220, anchor="center")

# Search icon/label
tk.Label(
    search_frame,
    text="🔍",
    font=("Segoe UI", 16),
    fg="#999999",
    bg="#2a2a2a",
    padx=10
).pack(side="left")

movie_entry = tk.Entry(
    search_frame,
    font=("Segoe UI", 14),
    width=40,
    bg="#2a2a2a",
    fg="white",
    insertbackground="white",
    relief="flat",
    bd=0
)
movie_entry.pack(side="left", padx=(0, 10), pady=15)

# Placeholder text effect
def on_entry_click(event):
    if movie_entry.get() == 'Search for movies or TV shows...':
        movie_entry.delete(0, "end")
        movie_entry.config(fg='white')

def on_focusout(event):
    if movie_entry.get() == '':
        movie_entry.insert(0, 'Search for movies or TV shows...')
        movie_entry.config(fg='#999999')

movie_entry.insert(0, 'Search for movies or TV shows...')
movie_entry.config(fg='#999999')
movie_entry.bind('<FocusIn>', on_entry_click)
movie_entry.bind('<FocusOut>', on_focusout)

# ================= Button =================
btn = tk.Button(
    bg_label,
    text="🎬 GET RECOMMENDATIONS",
    font=("Segoe UI", 13, "bold"),
    bg="#6a040f",
    fg="white",
    activebackground="#6a040f",
    activeforeground="white",
    relief="flat",
    bd=0,
    width=28,
    height=2,
    cursor="hand2"
)
btn.place(relx=0.5, y=285, anchor="center")

# Hover effect
def on_enter(e):
    btn['background'] = '#6a040f'

def on_leave(e):
    btn['background'] = '#6a040f'

btn.bind("<Enter>", on_enter)
btn.bind("<Leave>", on_leave)

# ================= Output Frame =================
output_frame = tk.Frame(bg_label, bg="#141414")
output_frame.place(relx=0.5, y=340, anchor="n", width=960, height=290)

poster_refs = []  # Store poster references

# ================= Poster Functions =================
def fetch_poster(title):
    """Fetch movie/TV show poster from TMDb or OMDB API"""
    if not USE_OMDB and TMDB_API_KEY != "YOUR_TMDB_API_KEY_HERE":
        return fetch_poster_tmdb(title)
    return fetch_poster_omdb(title)

def fetch_poster_tmdb(title):
    """Fetch poster from TMDb API"""
    try:
        # Clean up the title
        clean_title = title.strip()
        
        search_url = f"{TMDB_BASE_URL}/search/multi"
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'language': 'en-US'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"[ERROR] TMDb API returned status code: {response.status_code}")
            return create_placeholder_poster(title)
        
        data = response.json()
        
        if data.get('results') and len(data['results']) > 0:
            result = data['results'][0]
            poster_path = result.get('poster_path')
            
            print(f"[INFO] TMDb found: {result.get('name') or result.get('title')}")
            
            if poster_path:
                poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
                print(f"[INFO] Fetching poster from: {poster_url}")
                
                img_response = requests.get(poster_url, timeout=10)
                img = Image.open(BytesIO(img_response.content))
                img = img.resize((140, 210), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
        
        print(f"[WARNING] No TMDb results for '{title}'")
        return create_placeholder_poster(title)
        
    except Exception as e:
        print(f"[ERROR] TMDb exception for '{title}': {str(e)}")
        return create_placeholder_poster(title)

def fetch_poster_omdb(title):
    """Fetch poster from OMDB API"""
    try:
        # Clean up the title for better API results
        clean_title = title.strip()
        url = f"http://www.omdbapi.com/?t={clean_title}&apikey={OMDB_API_KEY}"
        
        response = requests.get(url, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"[ERROR] API returned status code: {response.status_code}")
            return create_placeholder_poster(title)
        
        data = response.json()
        
        # Debug: print what we got
        print(f"[DEBUG] Response for '{title}': {data.get('Response', 'N/A')}")
        
        if data.get('Response') == 'True' and data.get('Poster') and data['Poster'] != "N/A":
            poster_url = data['Poster']
            print(f"[INFO] Fetching poster from: {poster_url}")
            
            img_response = requests.get(poster_url, timeout=10)
            img = Image.open(BytesIO(img_response.content))
            img = img.resize((140, 210), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        else:
            error_msg = data.get('Error', 'No poster available')
            print(f"[WARNING] No poster for '{title}': {error_msg}")
            return create_placeholder_poster(title)
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout fetching poster for '{title}'")
        return create_placeholder_poster(title)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Connection error for '{title}'")
        return create_placeholder_poster(title)
    except Exception as e:
        print(f"[ERROR] Exception for '{title}': {str(e)}")
        return create_placeholder_poster(title)

def create_placeholder_poster(title):
    """Create a placeholder poster with gradient and movie icon"""
    from PIL import ImageDraw
    
    img = Image.new('RGB', (140, 210), color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    # Add gradient effect
    for i in range(210):
        color_value = int(26 + (60 * i / 210))
        draw.rectangle([(0, i), (140, i+1)], fill=(color_value, color_value, color_value))
    
    # Add Netflix-style N logo
    # Draw stylized N
    draw.rectangle([(40, 80), (50, 130)], fill='#E50914')  # Left bar
    draw.rectangle([(90, 80), (100, 130)], fill='#E50914')  # Right bar
    # Diagonal
    points = [(50, 80), (90, 130), (90, 125), (55, 80)]
    draw.polygon(points, fill='#E50914')
    
    # Add text "No Poster"
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = None
    
    text = "No Poster"
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (140 - text_width) // 2
        draw.text((x, 150), text, fill='#999999', font=font)
    
    return ImageTk.PhotoImage(img)

# ================= Main Functions =================
def get_recommendations():
    """Get recommendations from AI model"""
    if not MODEL_LOADED:
        messagebox.showerror("Error", "Model not loaded. Please check dataset path.")
        return
    
    movie = movie_entry.get().strip()
    # Remove placeholder text if present
    if movie == "" or movie == "Search for movies or TV shows...":
        messagebox.showwarning("Warning", "Please enter a movie/show title")
        return
    
    # Get recommendations from AI model
    recommendations = recommend_movies(movie, top_n=5)
    
    # Check if there was an error
    if len(recommendations) > 0 and "not found" in recommendations[0].lower():
        messagebox.showwarning("Not Found", recommendations[0])
        return
    
    # Display the recommendations with posters
    display_recommendations(recommendations)

def display_recommendations(recommendations):
    """Display movie recommendations with their posters"""
    # Clear previous results
    for widget in output_frame.winfo_children():
        widget.destroy()
    poster_refs.clear()

    # Show loading message
    loading_label = tk.Label(
        output_frame,
        text="🎬 Finding perfect matches...",
        font=("Segoe UI", 14, "italic"),
        fg="#E50914",
        bg="#141414"
    )
    loading_label.pack(pady=100)
    root.update()

    # Remove loading message
    loading_label.destroy()

    # Create scrollable container
    canvas = tk.Canvas(output_frame, bg="#141414", highlightthickness=0)
    scrollbar = tk.Scrollbar(output_frame, orient="horizontal", command=canvas.xview)
    scrollable_frame = tk.Frame(canvas, bg="#141414")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(xscrollcommand=scrollbar.set)

    # Display each movie with its poster
    for i, rec in enumerate(recommendations):
        # Movie card with shadow effect
        frame = tk.Frame(
            scrollable_frame,
            bg="#1f1f1f",
            highlightbackground="#E50914",
            highlightthickness=1,
            relief="flat"
        )
        frame.pack(side="left", padx=15, pady=20)

        # Poster container with rounded corners effect
        poster_container = tk.Frame(
            frame,
            bg="#000000",
            width=140,
            height=210,
            highlightbackground="#E50914",
            highlightthickness=2
        )
        poster_container.pack(pady=(10, 5))
        poster_container.pack_propagate(False)
        
        poster = fetch_poster(rec)
        lbl_img = tk.Label(poster_container, bg="#000000")
        lbl_img.pack(expand=True, fill="both")
        
        lbl_img.config(image=poster)
        lbl_img.image = poster
        poster_refs.append(poster)

        # Movie title with better styling
        lbl_text = tk.Label(
            frame,
            text=rec,
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#1f1f1f",
            wraplength=150,
            justify="center"
        )
        lbl_text.pack(pady=(5, 10))
        
        # Add rating/year placeholder (optional)
        info_label = tk.Label(
            frame,
            text="⭐ Recommended",
            font=("Segoe UI", 9),
            fg="#E50914",
            bg="#1f1f1f"
        )
        info_label.pack(pady=(0, 10))

    canvas.pack(side="top", fill="both", expand=True)
    scrollbar.pack(side="bottom", fill="x")

btn.config(command=get_recommendations)

# ================= Run Application =================
print("[INFO] Starting GUI...")
if MODEL_LOADED:
    print("[SUCCESS] Ready to recommend movies!")
else:
    print("[WARNING] Model not loaded. Please check dataset path.")

root.mainloop()