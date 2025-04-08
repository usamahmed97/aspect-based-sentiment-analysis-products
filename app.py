# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import ast
from nltk.stem import PorterStemmer

app = FastAPI(title="Product Recommendation API")

origins = [
    "http://localhost:3000",
    "https://aspect-based-sentiment-analysis-web.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

product_df = pd.read_csv("data/product_data.csv", converters={
    "aggregated_aspect_scores": lambda x: ast.literal_eval(x) if pd.notnull(x) else {}
})

ps = PorterStemmer()

def normalize_aspect(aspect: str) -> str:
    tokens = aspect.lower().split()
    return " ".join(ps.stem(token) for token in tokens)

def get_product_score(agg_scores: Dict[str, float], normalized_desired: set) -> float:
    total = 0
    found = False
    for aspect, score in agg_scores.items():
        norm = normalize_aspect(aspect)
        if norm in normalized_desired:
            total += score
            found = True
    return total if found else None

def recommend_products(desired_aspects: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
    product_level = product_df.drop_duplicates(subset=["itemName"]).copy()
    normalized_desired = set(normalize_aspect(asp) for asp in desired_aspects)
    product_level["combined_score"] = product_level["aggregated_aspect_scores"].apply(
        lambda d: get_product_score(d, normalized_desired)
    )
    filtered = product_level[product_level["combined_score"].notnull()]
    ranked = filtered.sort_values("combined_score", ascending=False)
    return ranked.head(top_n).to_dict(orient="records")

# Define request and response models.
class RecommendationRequest(BaseModel):
    aspects: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "Welcome to the Product Recommendation API!"}

@app.get("/products", response_model=RecommendationResponse)
async def get_all_products():
    recs = product_df.to_dict(orient="records")
    return {"recommendations": recs}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_products_endpoint(request: RecommendationRequest):
    recs = recommend_products(request.aspects, top_n=5)
    return {"recommendations": recs}
