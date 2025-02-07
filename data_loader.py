import pandas as pd
from database import db
from models import Users, User_product_data, Product_data


trending_products = pd.read_csv("models/trending_product.csv")
train_data = pd.read_csv("models/product_and_user_data.csv")
user_data = pd.read_csv('models/user_data.csv')
product_data = pd.read_csv('models/product_data.csv')

train_data = train_data.where(pd.notnull(train_data), None)
user_data = user_data.where(pd.notnull(user_data), None)
product_data = product_data.where(pd.notnull(product_data), None)

product_data = product_data.drop_duplicates(subset=['Prod Id'])

def insert_users_data():
    """Inserts user data into the Users table."""
    for _, row in user_data.iterrows():
        user = Users(
            user_id=row['User Id'],
            username=row['username'],
            email=row['email'],
            phone=row['phone'],
            password=row['password']
        )
        db.session.add(user)
    db.session.commit()

def insert_user_product_data():
    """Inserts user-product interactions into the User_product_data table."""
    for _, row in train_data.iterrows():
        user_product = User_product_data(
            user_id=row['User Id'],
            product_id=row['Prod Id'],
            product_name=row['Product Name'],
            price=row['List Price'],
            rating=row['Rating']
        )
        db.session.add(user_product)
    db.session.commit()

def insert_product_data():
    """Inserts product data into the Product_data table."""
    for _, row in product_data.iterrows():
        product = Product_data(
            prod_id=row['Prod Id'],
            product_url=row['Product Url'],
            product_name=row['Product Name'],
            description=row['Description'],
            filtered_description=row['Filtered Description'],
            list_price=row['List Price'],
            sale_price=row['Sale Price'],
            brand=row['Brand'],
            category=row['Category'],
            filtered_category=row['Filtered Category'],
            rating=row['Rating'],
            rating_count=row['Rating Count'],
            available=row['Available'],
            tags=row['Tags']
        )
        db.session.add(product)
    db.session.commit()

def insert_all_data():
    """Calls all insert functions to populate the database."""
    insert_users_data()
    insert_user_product_data()
    insert_product_data()

