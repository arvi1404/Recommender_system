import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import Users, User_product_data, Product_data

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def calculate_avg_cg(relevance_scores):
    return float(np.sum(relevance_scores) / (5 * len(relevance_scores)))

# Function to calculate DCG
def calculate_dcg(relevance_scores):
    return float(np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)]))

# Function to calculate NDCG
def calculate_ndcg(relevance_scores):
    # Calculate DCG
    dcg_value = calculate_dcg(relevance_scores)
    # Calculate IDCG
    relevance_scores.sort(reverse=True)
    idcg_value = calculate_dcg(relevance_scores)
    return float(dcg_value / idcg_value) if idcg_value != 0 else 0

# Get product details form the database
def get_product_data():
    try:
        product_data = Product_data.query.all()
        data_list = []
        for data in product_data:
            data_list.append({
                'Prod Id': int(data.prod_id),
                'Product Url': data.product_url,
                'Product Name': data.product_name,
                'Description': data.description,
                'Filtered Description': data.filtered_description,
                'List Price': float(data.list_price),
                'Sale Price': float(data.sale_price),
                'Brand': data.brand,
                'Category': data.category,
                'Filtered Category': data.filtered_category,
                'Available': bool(data.available),
                'Rating': float(data.rating),
                'Rating Count': float(data.rating_count),
                'Tags': data.tags,
            })
        df = pd.DataFrame(data_list)
        return df
    
    except Exception as e:
        return {'error': str(e)}

def get_user_and_product_data():
    try:
        user_product_data = User_product_data.query.all()
        # Convert the query result into a list of dictionaries
        data_list = []
        for data in user_product_data:
            data_list.append({
                'User Id': int(data.user_id),
                'Prod Id': int(data.product_id),
                'Product Name': data.product_name,
                'List Price': float(data.price),
                'Rating': float(data.rating)
            })
        df = pd.DataFrame(data_list)
        return df
    
    except Exception as e:
        return {'error': str(e)}
    
# Function to fetch product details
def fetch_product_by_id(product_id, product_data):
    if product_id in list(product_data['Prod Id']):
        return product_data[product_data['Prod Id'] == product_id].iloc[0]
