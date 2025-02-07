# Recommendation System for E-Commerce

This project aims to enhance the online shopping experience by developing a personalized recommendation system for an e-commerce platform. By analyzing user preferences and behavior, the system generates tailored product suggestions, improving product discovery, increasing sales, and fostering customer engagement.

---

## Data Preprocessing
To build an effective recommendation system, the initial product dataset underwent preprocessing to extract relevant products and generate appropriate tags.

---

## Frontend Development
The user interface is built using Flask, providing an interactive platform for users to engage with the recommendation system.

### Key Features
- **User Authentication**: Allows users to sign up and log in, enabling access to personalized recommendations.
- **Product Recommendation Page**: Displays personalized product suggestions, including product images, names, and prices.
- **Product Details Page**: Shows product descriptions, categories, ratings, review counts, and related product recommendations using content-based and collaborative filtering methods.
---

## Backend Development
The backend is responsible for managing user interactions and executing the recommendation algorithms.

### User Authentication
- Manages user sign-up and login using a MySQL database.
- Stores essential user details such as username, email, phone number, and password.

### Content-Based Filtering
- Identifies similar products based on their content, using product tags derived from product names, descriptions, and categories.
- Employs TF-IDF vectorization to process product tags and calculates cosine similarity between products.
- Returns top recommendations along with their relevance scores.

### User-Based Collaborative Filtering
- Assesses user similarity based on past purchase history and product ratings.
- Constructs a user-item matrix to measure similarity using cosine distance.
- Suggests products that similar users have purchased but the target user has not.

### Item-Based Collaborative Filtering
- Identifies product relationships based on co-occurrence in user purchase histories.
- Builds an item similarity matrix using collaborative filtering techniques.
- Computes weighted similarity scores to recommend products that are frequently purchased together.

### "Users Who Bought This Also Bought" Feature
- Extends item-based collaborative filtering to enhance recommendations.
- Identifies common purchases among users who have bought a specific item.
- Ranks recommendations based on a combination of similarity scores and user ratings to improve relevance.

### Hybrid Recommendation Model
- Provides personalised recommendations for customers who have an account
- Combines content-based filtering with collaborative filtering for enhanced accuracy.
- Merges recommendations from different models, eliminating duplicates to provide a more diverse selection of suggested products.

### Trending Products
- Highlights popular products based on user engagement.
- Calculates popularity scores by multiplying product ratings by the number of reviews.
- Ranks and displays top products, ensuring high-quality recommendations.

---

## Model Evaluation
The system evaluates recommendations using multiple performance metrics.

### Relevance Scores
- Measures how closely a recommended product aligns with user preferences based on past interactions.
- Uses product ratings as a key indicator of relevance, on a scale from 0 to 5.

### Cosine Similarity
- Computes similarity between products or users using cosine similarity.
- Applies to both TF-IDF-based content filtering and collaborative filtering models.

### Average Cumulative Gain (ACG)
- Evaluates how well recommendations compare to an ideal recommendation scenario.
- A perfect recommendation is one with a relevance score of 5 and cosine similarity of 1.

### Discounted Cumulative Gain (DCG)
- Measures recommendation quality by prioritizing higher-ranked items.
- Discounts the relevance score of lower-ranked recommendations.

### Normalized Discounted Cumulative Gain (NDCG)
- Normalizes DCG to assess the effectiveness of ranking order.
- Rewards systems that present relevant recommendations earlier in the list.

---

## Database Management
### Schema Design
The MySQL database is designed to efficiently manage user and product data.

#### Users Table
- Stores user authentication details, including username, email, phone number, and password.
- Ensures secure storage and retrieval of credentials.

#### Product Purchases Table
- Tracks user purchases and product metadata, including product IDs and ratings.
- Optimized for efficient querying and recommendation generation.

---

## Execution
To run the engine first set up a virtual enviroment by running setup.sh

```bash
chmod +x setup.sh
./setup.sh
```
Also ensure mySQL is installed and running in the server. The database can be created by running,

```python
python create_db.py
python data_loader.py
```
Start the app by,
```python
python app.py
```




