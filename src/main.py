#!/usr/bin/env python
# coding: utf-8

# In[58]:


# to import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from config import RAWG_CLEAN, STEAM_CLEAN, VGSALES_CLEAN, RESULTS_DIR, MIN_STEAM_REVIEWS, POS_RATIO_HIGH_THRESH, REVIEW_BINS, REVIEW_LABELS, TEST_SIZE, RANDOM_STATE, FIG_DPI, PRICE_BINS_FREE, PRICE_LABELS_Q3, PRICE_LABELS_Q7, map_platform
import warnings
warnings.filterwarnings("ignore")


# In[47]:


# to load all cleaned datasets
rawg  = pd.read_csv(RAWG_CLEAN)
steam = pd.read_csv(STEAM_CLEAN)
sales = pd.read_csv(VGSALES_CLEAN)

# To use standardized genre
rawg  = rawg[rawg["genre_std"].notna()]
steam = steam[steam["genre_std"].notna()]
sales = sales[sales["genre_std"].notna()]


# In[48]:


# to show basic info
print("RAWG dataset:", rawg.shape)
print("Steam dataset:", steam.shape)
print("Sales dataset:", sales.shape)


# In[49]:


#Q1---Top genres by frequency---(RAWG DATA)
g = rawg.assign(genre_std=rawg["genre_std"].str.split(";")).explode("genre_std")
g["genre_std"] = g["genre_std"].str.strip()
genre_counts = g["genre_std"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="Blues_r")
plt.title("Top 10 Most Common Genres (RAWG)")
plt.xlabel("Number of Games")
plt.ylabel("Genre")
plt.savefig(RESULTS_DIR / "q1_top_genres.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# In[50]:


#Q1---Average rating by genre (RAWG data)--- (Top 10 most common genres)
# Get only the rows where the genre is one of the top 10
top_genres = genre_counts.index

avg_rating_genre = (
    g[g["genre_std"].isin(top_genres)]
      .groupby("genre_std")["rating"]
      .mean()
      .reindex(top_genres)
      .round(2)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=[avg_rating_genre[g] for g in top_genres],
    y=top_genres,
    palette="Greens_r",
)

for i, gg in enumerate(top_genres):
    plt.text(avg_rating_genre[gg] + 0.02, i, f"{avg_rating_genre[gg]:.2f}", va="center")

plt.title("Average Rating for Top 10 Most Common Genres (RAWG)")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.xlim(1, 5)
plt.savefig(RESULTS_DIR / "q1_average_rating_top_genres.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# In[51]:


# Q2---Average rating by platform (RAWG data) --- (PC vs PlayStation vs Xbox vs Switch)
rawg_plat = (
    rawg[["rating", "platforms"]].dropna()
    .assign(platform=lambda d: d["platforms"].str.split(r"[;,/|]"))
    .explode("platform")
)

rawg_plat["platform"] = rawg_plat["platform"].str.strip().str.lower()
rawg_plat["platform_group"] = rawg_plat["platform"].apply(map_platform)

avg_rating_platform = (
    rawg_plat.dropna(subset=["platform_group"])
             .groupby("platform_group")["rating"]
             .mean()
             .round(2)
             .reindex(["PC", "PlayStation", "Xbox", "Switch"])
             .dropna()
)

plt.figure(figsize=(10, 5))
sns.barplot(x=avg_rating_platform.values, y=avg_rating_platform.index, palette="Oranges_r")
for i, val in enumerate(avg_rating_platform.values):
    plt.text(val + 0.02, i, f"{val:.2f}", va="center")
plt.title("Average Rating by Platform (RAWG)")
plt.xlabel("Average Rating")
plt.ylabel("Platform")
plt.xlim(0, 5)
plt.savefig(RESULTS_DIR / "q2_avg_rating_by_platform.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# In[52]:


# Q3---Do cheaper games tend to receive higher user reviews?(Steam data)---

# Shared numeric view of Steam (reuse in Q3/Q4/Q5/Q7)
steam_num = steam.copy()
steam_num["positive_ratings"] = pd.to_numeric(steam_num["positive_ratings"], errors="coerce")
steam_num["negative_ratings"] = pd.to_numeric(steam_num["negative_ratings"], errors="coerce")
steam_num["price"]            = pd.to_numeric(steam_num.get("price"), errors="coerce")
steam_num["total_ratings"]    = steam_num["positive_ratings"] + steam_num["negative_ratings"]
steam_num["pos_ratio"]        = steam_num["positive_ratings"] / steam_num["total_ratings"]

# Average Positive Review Ratio by Price Range (Steam)
tmp = steam_num[
    (steam_num["total_ratings"] >= MIN_STEAM_REVIEWS)
    & (steam_num["price"].notna())
    & (steam_num["price"] >= 0)
].copy()
tmp["average_positive_rating_ratio"] = tmp["pos_ratio"]

# price bins from config (with explicit Free bucket)
tmp["price_bin"] = pd.cut(
    tmp["price"],
    bins=PRICE_BINS_FREE,
    labels=PRICE_LABELS_Q3,
    right=True,
    include_lowest=True,
)

avg_by_bin = (
    tmp.dropna(subset=["price_bin"])
       .groupby("price_bin", as_index=False)["average_positive_rating_ratio"]
       .mean()
)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=avg_by_bin,
    x="price_bin",
    y="average_positive_rating_ratio",
    order=PRICE_LABELS_Q3,
    palette="Greens",
)
for i, v in enumerate(avg_by_bin["average_positive_rating_ratio"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
plt.title("Average Positive Rating Ratio by Price Range (Steam)")
plt.xlabel("Price Range ($)")
plt.ylabel("Average Positive Rating Ratio")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.2)
plt.savefig(RESULTS_DIR / "q3_pos_ratio_by_price.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

corr = tmp[["price", "average_positive_rating_ratio"]].corr().loc["price", "average_positive_rating_ratio"]
print(f"Correlation between price and average positive rating ratio: {corr:.3f}  (n={len(tmp):,})")


# In[53]:


# Q4---Review volume vs. positive rating ratio (Steam data)
# Buckets: small (<100), medium (100–999), large (1,000+)
tmp = steam_num[steam_num["total_ratings"] > 0].copy()

tmp["review_bucket"] = pd.cut(
    tmp["total_ratings"],
    bins=REVIEW_BINS,
    labels=REVIEW_LABELS,
    right=False,
)

avg_by_bucket = (
    tmp.groupby("review_bucket", as_index=False)
       .agg(pos_ratio=("pos_ratio", "mean"),
            n_games=("pos_ratio", "size"))
)

print("Average positive rating ratio by review-volume bucket:")
print(avg_by_bucket.round({"pos_ratio": 3}))

plt.figure(figsize=(7, 4))
sns.barplot(data=avg_by_bucket, x="review_bucket", y="pos_ratio", palette="Purples")
for i, v in enumerate(avg_by_bucket["pos_ratio"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
plt.title("Average Positive Rating Ratio by Review Volume (Steam)")
plt.xlabel("Total Ratings (bucket)")
plt.ylabel("Average Positive Rating Ratio")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.2)
plt.savefig(RESULTS_DIR / "q4_pos_ratio_by_review_volume.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# In[54]:


#Q5---Which years had the highest number of highly rated games? (Steam data)---
df = steam_num.copy()
if "year" not in df.columns:
    df["year"] = pd.to_datetime(steam.get("released"), errors="coerce").dt.year

df = df[(df["total_ratings"] >= 25) & df["year"].notna()].copy()
df["positive_rating_ratio"] = df["pos_ratio"]

HIGH = POS_RATIO_HIGH_THRESH

year_counts = (
    df[df["positive_rating_ratio"] >= HIGH]
      .groupby("year", as_index=False)
      .size()
      .rename(columns={"size": "highly_rated_count"})
      .sort_values("year")
)

plt.figure(figsize=(9, 4))
sns.lineplot(data=year_counts, x="year", y="highly_rated_count", marker="o")
plt.title(f"Number of Highly Rated ({int(HIGH*100)}%+ positive) Games per Year - Steam")
plt.xlabel("Release Year")
plt.ylabel("Count of Highly Rated Games")
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
plt.savefig(RESULTS_DIR / "q5_highly_rated_by_year.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

print("Top years by count of highly rated games:")
print(year_counts.sort_values("highly_rated_count", ascending=False).head(10).to_string(index=False))


# In[55]:


# Q6---Cross-dataset: Do higher-rated genres also sell more? (Steam and VGSales Data)

# steam positive_rating_ratio (avg by genre) vs VGSales global_sales (avg by genre)
tmp = steam_num[steam_num["genre_std"].notna() & (steam_num["total_ratings"] > 0)].copy()
tmp["pos_ratio"] = steam_num.loc[tmp.index, "pos_ratio"]

sg = tmp.assign(genre_std=tmp["genre_std"].str.split(";")).explode("genre_std")
sg["genre_std"] = sg["genre_std"].str.strip()

steam_by_genre = (
    sg.groupby("genre_std", as_index=False)
      .agg(steam_pos_ratio=("pos_ratio", "mean"),
           n_steam=("pos_ratio", "size"))
)

v = sales.copy()
v = v[v["genre_std"].notna()].copy()
v["global_sales"] = pd.to_numeric(v["global_sales"], errors="coerce")

sales_by_genre = (
    v.groupby("genre_std", as_index=False)
     .agg(avg_global_sales=("global_sales", "mean"),
          n_sales=("global_sales", "size"))
)

merged = steam_by_genre.merge(sales_by_genre, on="genre_std", how="inner")

k = min(7, len(merged))
top = merged.sort_values("steam_pos_ratio", ascending=False).head(k)

plt.figure(figsize=(8, 4))
sns.barplot(data=top, x="steam_pos_ratio", y="genre_std", palette="Blues_r")
for i, val in enumerate(top["steam_pos_ratio"]):
    plt.text(val + 0.01, i, f"{val:.2f}", va="center")
plt.title("Steam: Avg Positive Rating Ratio by Genre (Top, intersect with Sales)")
plt.xlabel("Average Positive Rating Ratio")
plt.ylabel("Genre")
plt.xlim(0, 1)
plt.grid(axis="x", alpha=0.15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "q6_avg_pos_rating_ratio_by_genre.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(data=top, x="avg_global_sales", y="genre_std", palette="Greens_r")
for i, val in enumerate(top["avg_global_sales"]):
    plt.text(val + 0.01, i, f"{val:.2f}", va="center")
plt.title("VGSales: Avg Global Sales by Genre (same Top)")
plt.xlabel("Average Global Sales (Millions)")
plt.ylabel("Genre")
plt.grid(axis="x", alpha=0.15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "q6_avg_global_sales_by_genre.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()

corr = merged[["steam_pos_ratio", "avg_global_sales"]].dropna().corr().iloc[0, 1]
print(f"Correlation (genre avg Steam positive ratio vs. avg global sales): {corr:.3f}   (n={len(merged)})")


# In[56]:


# Q7 Simple linear regression to predict Steam positive_rating_ratio 
# Prepare base steam_num
df = steam_num.copy()

if "year" not in df.columns:
    df["year"] = pd.to_datetime(steam["released"], errors="coerce").dt.year

def owners_mid(x):
    s = str(x).replace(",", "").replace("–", "-").replace("—", "-").strip()
    if "-" in s and not s.startswith("-"):
        a, b = s.split("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

df["owners_mid"] = steam["owners"].apply(owners_mid)

df = df[(df["total_ratings"] >= MIN_STEAM_REVIEWS) & df["year"].notna()].copy()

df["price"]            = df["price"].clip(lower=0).fillna(0)
df["average_playtime"] = pd.to_numeric(df["average_playtime"], errors="coerce")
df["median_playtime"]  = pd.to_numeric(df["median_playtime"], errors="coerce")

df["is_free"]           = (df["price"] == 0).astype(int)
df["log_price"]         = np.log1p(df["price"])
df["log_owners_mid"]    = np.log1p(df["owners_mid"])
df["log_total_ratings"] = np.log1p(df["total_ratings"])

df["price_bin"] = pd.cut(
    df["price"],
    bins=PRICE_BINS_FREE,
    labels=PRICE_LABELS_Q7,
    right=True,
    include_lowest=True,
)

price_dummies = pd.get_dummies(df["price_bin"], prefix="price", drop_first=True)
df = pd.concat([df, price_dummies], axis=1)
price_dummy_cols = list(price_dummies.columns)

df["genre_list"] = df["genre_std"].astype(str).str.split(";")

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df["genre_list"])

k = 5
kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
df["genre_cluster"] = kmeans.fit_predict(genre_matrix)

num_cols = [
    "log_price",
    "is_free",
    "log_owners_mid",
    "log_total_ratings",
    "year",
    "median_playtime",
    "genre_cluster",
] + price_dummy_cols

df = df.dropna(subset=num_cols + ["pos_ratio"])

X_raw = df[num_cols].values
y     = df["pos_ratio"].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()

grid = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best alpha:", grid.best_params_["alpha"])
print(f"Best CV R2: {grid.best_score_:.3f}")

y_pred = best_model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R2:  {r2:.3f}")
print(f"Test RMSE:{rmse:.3f}")

coef_raw = pd.Series(best_model.coef_, index=num_cols)

pretty_names = {
    "log_total_ratings": "Total ratings (log)",
    "log_owners_mid":    "Owners (log)",
    "log_price":         "Price (log)",
    "price_0_5":         "Price ($0-5)",
    "price_5_15":        "Price ($5-15)",
    "price_15_30":       "Price ($15-30)",
    "price_30_60":       "Price ($30-60)",
    "genre_cluster":     "Genre cluster",
}

coef_pretty = coef_raw.rename(index=pretty_names).abs().sort_values(ascending=False)

print("\nFeature |coef| ranking:")
print(coef_pretty)

plt.figure(figsize=(7, 4))
coef_pretty.sort_values().plot(kind="barh")
plt.xlabel("|Coefficient|")
plt.title("Q7 Linear regression (Ridge, L2 regularization)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "q7_ridge_coefficients.png", dpi=FIG_DPI, bbox_inches="tight")
plt.show()


# In[57]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')


# In[ ]:




