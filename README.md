# Sample Project - <Analyzing & Predicting Video Game Ratings (Steam Positive Rating Ratio)>

Exploratory analysis of video-game data (2000–2020) and a simple regression model to predict Steam’s positive rating ratio. Sources: RAWG API, Steam (Kaggle), VGSales (Kaggle). Figures are produced by main.py and can be displayed in results folder.

# Data sources
1) RAWG API — game title, released date, rating, platforms, genres (sampled to 10,000 rows; saved as data/rawg_10000_unfiltered.csv).
   
2) Steam (Kaggle: nikdavis/steam-store-games) — positive/negative ratings, price, average/median playtime, owners, categories, platforms, publisher, release date (saved as data/steam.csv).
   
3) VGSales (Kaggle: gregorut/videogamesales) — name, platform, year, genre, global sales (saved as data/vgsales.csv).

Cleaning & standardization (shared):
1) Rename to a common schema; parse year; keep 2000–2020; drop missing year.
2) Remove NSFW/non-game software tags.
3) Build canonical genre_std from source genres (action, adventure, role-playing, shooter, strategy, simulation, racing, fighting, platform, puzzle, sports, misc), and split/explode for genre analyses.
4) Steam modeling fields: total_reviews = positive_ratings + negative_ratings, positive_rating_ratio = positive_ratings / total_reviews, owners_mid = midpoint of the owners range.

# Results 
1) Q1 (RAWG): Top 10 most common genres after exploding genre_std; action/adventure/shooter dominate; strategy/simulation sizable.

2) Q2 (RAWG): Average rating by platform bucket (PC / PlayStation / Xbox / Switch); consoles slightly higher on average in this sample.

3) Q3 (Steam): Average positive rating ratio by price bins (Free, $0–5, $5–15, $15–30, $30–60); mid-price ranges ($5–30) score highest.

4) Q4 (Steam): Review volume vs positive ratio (<100, 100–999, 1,000+); higher volume generally corresponds to higher positive ratios.

5) Q5 (Steam): Years with the most highly rated games (pos_ratio ≥ 0.75, ≥10 reviews); peak around 2016–2017.

6) Q6 (Steam × VGSales): Avg positive_rating_ratio by genre_std vs avg global sales by genre_std; weak/negative correlation in this sample.

7) Prediction model: Linear Regression to predict pos_ratio using price, average_playtime, owners_mid, primary genre_std, platforms, and year. Numeric features standardized; categoricals label-encoded. Expected low R² due to noisy user ratings; coefficients reported and visualized.

# Installation
Set environment variables/keys:
.env with RAWG_API_KEY=your_key_here
Place kaggle.json in the project root; the data-collection notebook copies it to ~/.kaggle/kaggle.json
Python packages:
pandas, numpy, matplotlib, seaborn, scikit-learn, python-dotenv, requests, kaggle

# Running analysis 

From `src/` directory run:

`python main.py `

Results will appear in `results/` folder. All obtained will be stored in `data/`


# Notes and Limitations
1) Genre standardization is a compact mapping for an intro course. Some nuanced sub-genres are bucketed into misc.

2) Ratings are user-driven and noisy; a simple linear model won’t capture taste shifts, marketing, or franchise effects.

3) Results are descriptive; do not imply causation.