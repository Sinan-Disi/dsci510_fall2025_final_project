# to import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")





# to load all cleaned datasets
rawg = pd.read_csv("../data/rawg_clean.csv")
steam = pd.read_csv("../data/steam_clean.csv")
sales = pd.read_csv("../data/vgsales_clean.csv")

# To use standardized genre
rawg = rawg[rawg['genre_std'].notna()]
steam = steam[steam['genre_std'].notna()]
sales = sales[sales['genre_std'].notna()]



# to show basic info
print("RAWG dataset:", rawg.shape)
print("Steam dataset:", steam.shape)
print("Sales dataset:", sales.shape)



#Q1 to analyze RAWG data (Questions 1 and 2)
# Top genres by frequency
g = rawg.assign(genre_std=rawg['genre_std'].str.split(';')).explode('genre_std')
g['genre_std'] = g['genre_std'].str.strip()
genre_counts = g['genre_std'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='Blues_r')
plt.title("Top 10 Most Common Genres (RAWG)")
plt.xlabel("Number of Games")
plt.ylabel("Genre")
plt.savefig("../results/q1_top_genres.png", dpi=150, bbox_inches="tight")
plt.show()




#Q1 Average rating by genre (Top 10 most common genres)
# Get only the rows where the genre is one of the top 10
top_genres = genre_counts.index

avg_rating_genre = (
    g[g['genre_std'].isin(top_genres)]
      .groupby('genre_std')['rating']
      .mean()
      .reindex(top_genres)     
      .round(2)
)
plt.figure(figsize=(10,5))
sns.barplot(
    x=[avg_rating_genre[g] for g in top_genres],
    y=top_genres,
    palette='Greens_r'
)

# Add labels
for i, g in enumerate(top_genres):
    plt.text(avg_rating_genre[g] + 0.02, i, f'{avg_rating_genre[g]:.2f}', va='center')

plt.title("Average Rating for Top 10 Most Common Genres (RAWG)")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.xlim(1, 5)
plt.savefig("../results/q1_average_rating_top_genres.png", dpi=150, bbox_inches="tight")
plt.show()



# Q2 Average rating by platform (PC vs PlayStation vs Xbox vs Switch)

def map_platform(p: str):
    p = str(p).lower()
    if 'pc' in p: return 'PC'
    if 'playstation' in p: return 'PlayStation'
    if 'xbox' in p: return 'Xbox'
    if 'switch' in p: return 'Switch'
    return None

rawg_plat = (
    rawg[['rating', 'platforms']].dropna()
    .assign(platform=lambda d: d['platforms'].str.split(r'[;,/|]'))  # key fix
    .explode('platform')
)

rawg_plat['platform'] = rawg_plat['platform'].str.strip().str.lower()
rawg_plat['platform_group'] = rawg_plat['platform'].apply(map_platform)

avg_rating_platform = (
    rawg_plat.dropna(subset=['platform_group'])
             .groupby('platform_group')['rating']
             .mean()
             .round(2)
             .reindex(['PC', 'PlayStation', 'Xbox', 'Switch'])
             .dropna()
)

plt.figure(figsize=(10,5))
sns.barplot(x=avg_rating_platform.values, y=avg_rating_platform.index, palette='Oranges_r')
for i, val in enumerate(avg_rating_platform.values):
    plt.text(val + 0.02, i, f'{val:.2f}', va='center')
plt.title("Average Rating by Platform (RAWG)")
plt.xlabel("Average Rating")
plt.ylabel("Platform")
plt.xlim(0, 5)
plt.savefig("../results/q2_avg_rating_by_platform.png", dpi=150, bbox_inches="tight")
plt.show()




# Q3 Do cheaper games tend to receive higher user reviews?


# Shared numeric view of Steam (reuse in Q3/Q4/Q5)
steam_num = steam.copy()
steam_num['positive_ratings'] = pd.to_numeric(steam_num['positive_ratings'], errors='coerce')
steam_num['negative_ratings'] = pd.to_numeric(steam_num['negative_ratings'], errors='coerce')
steam_num['price']            = pd.to_numeric(steam_num.get('price'), errors='coerce')
steam_num['total_ratings']    = steam_num['positive_ratings'] + steam_num['negative_ratings']
steam_num['pos_ratio']        = steam_num['positive_ratings'] / steam_num['total_ratings']


##Average Positive Review Ratio by Price Range (Steam)

tmp = steam_num[(steam_num['total_ratings'] > 10) & (steam_num['price'].notna()) & (steam_num['price'] >= 0)].copy()
tmp['average_positive_rating_ratio'] = tmp['pos_ratio']  # reuse precomputed ratio

#Price bins
bins   = [-0.01, 0, 5, 15, 30, 60]
labels = ['Free', '$0–5', '$5–15', '$15–30', '$30–60']
tmp['price_bin'] = pd.cut(tmp['price'], bins=bins, labels=labels, right=True, include_lowest=True)

# Averages per bin 
avg_by_bin = (
    tmp.dropna(subset=['price_bin'])
       .groupby('price_bin', as_index=False)['average_positive_rating_ratio']
       .mean()
)

#Plot
plt.figure(figsize=(7,4))
sns.barplot(data=avg_by_bin, x='price_bin', y='average_positive_rating_ratio', order=labels, palette='Greens')
for i, v in enumerate(avg_by_bin['average_positive_rating_ratio']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
plt.title("Average Positive Rating Ratio by Price Range (Steam)")
plt.xlabel("Price Range ($)")
plt.ylabel("Average Positive Rating Ratio")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.2)
plt.savefig("../results/q3_pos_ratio_by_price.png", dpi=150, bbox_inches="tight")
plt.show()

# Simple correlation 
corr = tmp[['price', 'average_positive_rating_ratio']].corr().loc['price', 'average_positive_rating_ratio']
print(f"Correlation between price and average positive rating ratio: {corr:.3f}  (n={len(tmp):,})")




# Q4 — Review volume vs. positive rating ratio (Steam)
# Buckets: small (<100), medium (100–999), large (1,000+)

tmp = steam_num[steam_num['total_ratings'] > 0].copy()   # already has pos_ratio & totals

# bucket by review volume
bins = [0, 100, 1000, float('inf')]
labels = ['<100', '100–999', '1,000+']
tmp['review_bucket'] = pd.cut(tmp['total_ratings'], bins=bins, labels=labels, right=False)

# average positive rating ratio per bucket (+ counts)
avg_by_bucket = (
    tmp.groupby('review_bucket', as_index=False)
       .agg(pos_ratio=('pos_ratio', 'mean'),
            n_games=('pos_ratio', 'size'))
)
print("Average positive rating ratio by review-volume bucket:")
print(avg_by_bucket.round({'pos_ratio': 3}))

plt.figure(figsize=(7,4))
sns.barplot(data=avg_by_bucket, x='review_bucket', y='pos_ratio', palette='Purples')
for i, v in enumerate(avg_by_bucket['pos_ratio']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
plt.title("Average Positive Rating Ratio by Review Volume (Steam)")
plt.xlabel("Total Ratings (bucket)")
plt.ylabel("Average Positive Rating Ratio")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.2)
plt.savefig("../results/q4_pos_ratio_by_review_volume.png", dpi=150, bbox_inches="tight")
plt.show()



#Q5 Which years had the highest number of highly rated games? (Steam)
df = steam_num.copy()
if 'year' not in df.columns:
    df['year'] = pd.to_datetime(steam.get('released'), errors='coerce').dt.year


#Define "highly rated" using positive_rating_ratio and require some volume
df = df[(df['total_ratings'] >= 25) & df['year'].notna()].copy()
df['positive_rating_ratio'] = df['pos_ratio']

HIGH = 0.75  # threshold for "highly rated" (75%+ positive)
year_counts = (df[df['positive_rating_ratio'] >= HIGH]
               .groupby('year', as_index=False)
               .size()
               .rename(columns={'size':'highly_rated_count'})
               .sort_values('year'))


plt.figure(figsize=(9,4))
sns.lineplot(data=year_counts, x='year', y='highly_rated_count', marker='o')
plt.title(f"Number of Highly Rated (≥{int(HIGH*100)}% positive) Games per Year — Steam")
plt.xlabel("Release Year")
plt.ylabel("Count of Highly Rated Games")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
plt.savefig("../results/q5_highly_rated_by_year.png", dpi=150, bbox_inches="tight")
plt.show()

#Print the top years
print("Top years by count of highly rated games:")
print(year_counts.sort_values('highly_rated_count', ascending=False).head(10).to_string(index=False))



# Q6 Cross-dataset: Do higher-rated genres also sell more?
# steam positive_rating_ratio (avg by genre) vs VGSales global_sales (avg by genre)

tmp = steam_num[steam_num['genre_std'].notna() & (steam_num['total_ratings'] > 0)].copy()
tmp['pos_ratio'] = steam_num.loc[tmp.index, 'pos_ratio']  # reuse computed ratio

sg = tmp.assign(genre_std=tmp['genre_std'].str.split(';')).explode('genre_std')
sg['genre_std'] = sg['genre_std'].str.strip()

steam_by_genre = (
    sg.groupby('genre_std', as_index=False)
      .agg(steam_pos_ratio=('pos_ratio', 'mean'),
           n_steam=('pos_ratio', 'size'))
)

#VGSales: average global sales by the same canonical genre 
v = sales.copy()
v = v[v['genre_std'].notna()].copy()

v['global_sales'] = pd.to_numeric(v['global_sales'], errors='coerce')

sales_by_genre = (
    v.groupby('genre_std', as_index=False)
     .agg(avg_global_sales=('global_sales', 'mean'),
          n_sales=('global_sales', 'size'))
)

merged = steam_by_genre.merge(sales_by_genre, on='genre_std', how='inner')

#to drop tiny-sample genres (if there is any)
MIN_STEAM = 10
MIN_SALES = 10
merged = merged[(merged['n_steam'] >= MIN_STEAM) & (merged['n_sales'] >= MIN_SALES)]
k = min(5, len(merged))
top = merged.sort_values('steam_pos_ratio', ascending=False).head(k)

#to plot
plt.figure(figsize=(8, 4))
sns.barplot(data=top, x='steam_pos_ratio', y='genre_std', palette='Blues_r')
for i, val in enumerate(top['steam_pos_ratio']):
    plt.text(val + 0.01, i, f'{val:.2f}', va='center')
plt.title("Steam: Avg Positive Rating Ratio by Genre (Top, intersect with Sales)")
plt.xlabel("Average Positive Rating Ratio")
plt.ylabel("Genre")
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.15)
plt.tight_layout()
plt.savefig("../results/q6_avg_pos_rating_ratio_by_genre.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(data=top, x='avg_global_sales', y='genre_std', palette='Greens_r')
for i, val in enumerate(top['avg_global_sales']):
    plt.text(val + 0.01, i, f'{val:.2f}', va='center')
plt.title("VGSales: Avg Global Sales by Genre (same Top)")
plt.xlabel("Average Global Sales (Millions)")
plt.ylabel("Genre")
plt.grid(axis='x', alpha=0.15)
plt.tight_layout()
plt.savefig("../results/q6_avg_global_sales_by_genre.png", dpi=150, bbox_inches="tight")
plt.show()

#Simple correlation (genre level)
corr = merged[['steam_pos_ratio','avg_global_sales']].dropna().corr().iloc[0,1]
print(f"Correlation (genre avg Steam positive ratio vs. avg global sales): {corr:.3f}   (n={len(merged)})")


#Q7 Simple linear regression to predict Steam positive_rating_ratio 

df = steam_num.copy()                      
df = df[df['total_ratings'] >= 10].copy()

# helper for owners
def owners_mid(x):
    s = str(x).replace(',', '').replace('–','-').strip()
    if '-' in s:
        a, b = s.split('-', 1)
        a = pd.to_numeric(a, errors='coerce'); b = pd.to_numeric(b, errors='coerce')
        return (a + b) / 2
    return pd.to_numeric(s, errors='coerce')

df['owners_mid'] = df['owners'].apply(owners_mid)
df['genre_primary'] = (
    df['genre_std'].astype(str).str.split(';').str[0].str.strip().replace({'nan':'unknown'})
)

# base numeric features
X = pd.DataFrame({
    'price'            : pd.to_numeric(df['price'], errors='coerce').clip(lower=0).fillna(0),
    'average_playtime' : pd.to_numeric(df['average_playtime'], errors='coerce').fillna(0),
    'owners_mid'       : pd.to_numeric(df['owners_mid'], errors='coerce').fillna(0),
    'year'             : pd.to_numeric(df['year'], errors='coerce')
})
y = df['pos_ratio'].astype(float)

# split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# fill year with train median (also avoid leakage)
year_med = X_train['year'].median()
X_train['year'] = X_train['year'].fillna(year_med)
X_test['year']  = X_test['year'].fillna(year_med)

genre_train = df.loc[X_train.index, 'genre_primary']
global_mean = y_train.mean()
genre_mean  = y_train.groupby(genre_train).mean()

X_train = X_train.assign(
    genre_te = genre_train.map(genre_mean).fillna(global_mean)
)
X_test = X_test.assign(
    genre_te = df.loc[X_test.index, 'genre_primary'].map(genre_mean).fillna(global_mean)
)

# scale numeric columns AFTER split (no leakage)
num_cols = ['price','average_playtime','owners_mid','year','genre_te']
scaler = StandardScaler().fit(X_train[num_cols])

X_train_scaled = X_train.copy();  X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

#regression
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# coeffs and PNG
coef = pd.Series(model.coef_, index=X_train_scaled.columns).abs().sort_values(ascending=False).head(8)
plt.figure(figsize=(7,4))
plt.barh(coef.index, coef.values)
plt.gca().invert_yaxis()
plt.title("Top feature influence (|coefficient|)")
plt.xlabel("|Coefficient|")
plt.tight_layout()
plt.savefig("../results/q7_regression_coeffs.png", dpi=150, bbox_inches="tight")
plt.show()






