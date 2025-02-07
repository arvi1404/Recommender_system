from database import db

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'users_product_data' table
class User_product_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    product_id = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    price = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.String(100), nullable=False)

# Define the model for 'product_data' table
class Product_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prod_id = db.Column(db.String(10), nullable=True)
    product_url = db.Column(db.String(200), nullable=True)
    product_name = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(5000), nullable=True)
    filtered_description = db.Column(db.String(5000), nullable=True)
    list_price = db.Column(db.String(10), nullable=True)
    sale_price = db.Column(db.String(10), nullable=True)
    brand = db.Column(db.String(100), nullable=True)
    category = db.Column(db.String(300), nullable=True)
    filtered_category = db.Column(db.String(300), nullable=True)
    rating = db.Column(db.String(10), nullable=True)
    rating_count = db.Column(db.String(100), nullable=True)
    available = db.Column(db.String(10), nullable=True)
    tags = db.Column(db.String(2000), nullable=True)
