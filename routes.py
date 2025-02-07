from flask import Flask, request, render_template, redirect, session, flash, url_for, jsonify,Blueprint
import random
from recommendation import get_trending_products, content_based_recommendations, specific_item_based_recommendation,hybrid_recommendations
from Utils import get_product_data, truncate, get_user_and_product_data, fetch_product_by_id
from models import *
import pandas as pd

routes = Blueprint('routes', __name__)

#getting random image urls
random_image_urls = [
    "static/images/img_1.jpg",
    "static/images/img_2.jpg",
    "static/images/img_3.jpg",
    "static/images/img_4.jpg",
    "static/images/img_5.jpg",
]
trending_products = pd.read_csv("models/trending_product.csv")
random_prices = trending_products['List Price'].head(5).tolist()

@routes.route("/")
def index():
    # Get the top 5 trending products
    product_data = get_product_data()
    print(product_data)

    trending_products = get_trending_products(product_data, top_n=5)

    # Create a DataFrame with the trending products and their prices
    trending_prices = product_data.loc[product_data['Product Name'].isin(trending_products['Product Name']), ['Product Name','List Price']]
    
    # Convert to a dictionary for easier access in the template
    price_dict = trending_prices.set_index('Product Name')['List Price'].to_dict()

    # Generate random product image URLs for each trending product
    #random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]

    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        trending_products=trending_products,
        truncate=truncate,
        random_product_image_urls=random_image_urls,
        price_dict=price_dict,  # Pass the dictionary of prices
        zip=zip  # Passing zip function
    )

#routes
@routes.route("/main")
def main():
    return render_template('main.html', hybrid_rec=None)

# Sign up model
@routes.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        user_data = get_user_and_product_data()

        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        user_ids = user_data['User Id'].tolist()
        random_user_id = random.choice(user_ids)

        # Create a new user instance
        new_user = Users(username=username, email=email, phone=phone, password=password, user_id=random_user_id)
        db.session.add(new_user)
        db.session.commit()

        #random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        #prices = trending_products['List Price'].head(5).tolist()
        
        return render_template(
        'index.html',
        signup_success="True",
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_image_urls,
        random_price=random_prices,
        zip=zip # Passing zip function
    )

    return render_template('index.html',signup_success="False")

# login model
# Login route
@routes.route("/login", methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Please enter both username and password'})

    # Query the database to check if the user exists
    user = Users.query.filter_by(username=username, password=password).first()

    if user:
        session['username'] = user.username
        session['user_id'] = user.user_id
        return jsonify({'success': True, 'message': 'Login successful!'})
    else:
        return jsonify({'success': False, 'message': 'No user found, please enter a valid username'})

# logout model
@routes.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)  
    session.pop('user_id', None)    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('routes.index'))  

# Recommendations
@routes.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        products = request.form.get('products')
        target_user_id = request.form.get('user_id')
        product_data = get_product_data()

        # Check if user is logged in (i.e., target_user_id exists)
        if target_user_id:
            user_data = get_user_and_product_data()
            # Use hybrid recommendation if user is logged in
            hybrid_rec = hybrid_recommendations(user_data, product_data, target_user_id, products)
        else:
            # Use content-based recommendation only if user is not logged in
            hybrid_rec = content_based_recommendations(product_data, products)

        if hybrid_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            #random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            #prices = train_data['List Price'].head(5).tolist()

            return render_template('main.html', 
                                   hybrid_rec=hybrid_rec,
                                   truncate=truncate, 
                                   random_product_image_urls=random_image_urls,
                                   random_price=random_prices)

       
# add to cart
@routes.route("/add_to_cart", methods=['POST'])
def add_to_cart():
    if 'user_id' in session:  # Check if the user is logged in
        user_id = session['user_id']
        product_id = request.form['product_id']
        product_name = request.form['product_name']
        rating = request.form['rating']
        price = request.form['price']

        # Create a new Cart entry
        cart_item = User_product_data(user_id=user_id, product_id=product_id, product_name=product_name, rating=rating, price=price)
        db.session.add(cart_item)
        db.session.commit()

        # Return a JSON response indicating success
        return jsonify({'success': True, 'message': 'Product added to cart!'})

    else:
        # Return a JSON response indicating failure
        return jsonify({'success': False, 'message': 'Please log in to add items to your cart.'})

# Update product rating and rating count
@routes.route("/update_product_rating", methods=['POST'])
def update_product_rating():
    product_id = int(request.form['product_id'])

    new_user_product_data = get_user_and_product_data()

    agg_ratings = new_user_product_data.groupby('Prod Id').agg(mean_rating = ('Rating', 'mean'), rating_count=('Rating', 'count')).reset_index()
    agg_rating_for_product = agg_ratings[agg_ratings['Prod Id'] == product_id]
    new_rating = list(agg_rating_for_product['mean_rating'])[0]
    new_rating_count = list(agg_rating_for_product['rating_count'])[0]

    # Updata the product_data entry
    product = Product_data.query.filter_by(prod_id=str(product_id)).first()
    
    if product:
        product.rating = str(round(new_rating, 2))
        product.rating_count = str(new_rating_count)
        db.session.commit()
        print(f'Updated rating and rating count for product {product_id}')
    else:
        print(f"Product {product_id} not found")
    # Return a JSON response indicating success
    return jsonify({'success': True, 'message': 'Updated product rating and rating count!'})


# cart
@routes.route("/cart")
def cart():
    if 'user_id' in session:
        user_id = session['user_id']
        cart_items = User_product_data.query.filter_by(user_id=user_id).all()
        return render_template('cart.html', cart_items=cart_items)
    else:
        flash('Please log in to view your cart.')
        return redirect(url_for('routes.login'))

# remove item from cart 
@routes.route("/remove_from_cart", methods=['POST'])
def remove_from_cart():
    if 'user_id' in session:  # Check if the user is logged in
        user_id = session['user_id']
        product_id = request.form['product_id']  # Get product_id from the form

        # Find the cart item by user_id and product_id
        cart_item = User_product_data.query.filter_by(user_id=user_id, product_id=product_id).first()

        if cart_item:
            db.session.delete(cart_item)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Product removed from cart.'})
        else:
            return jsonify({'success': False, 'message': 'Product not found in cart.'})
    else:
        return jsonify({'success': False, 'message': 'Please log in to manage your cart.'})

# Route to render the product details page
@routes.route('/product/<int:product_id>')
def product_detail(product_id):
    # Fetch product details from your data source using the product_id
    user_data = get_user_and_product_data()
    product_data = get_product_data()
    #print(product_data[['Prod Id', 'Product Name', 'Rating', 'Rating Count']])
    product = fetch_product_by_id(product_id, product_data)

    # Content based recommendations
    content_based_rec, cb_scores, cb_metrics = content_based_recommendations(product_data, product['Product Name'], top_n=20)
    # Item based collaborative recommendations
    specific_item_based_rec, sib_scores, sib_metrics = specific_item_based_recommendation(product_id, user_data, product_data, n_top=20)

    #print(f'METRICS: Content Based Recommendations for Product {product['Prod Id']}: {cb_metrics}')
    print('METRICS: Content Based Recommendations for Product {}: {}'.format(product['Prod Id'], cb_metrics))
    print(f'METRICS: Specific Item Based Recommendation Scores for Product {product_id}: {sib_metrics}')

    # **Remove duplicate products based on 'Prod Id'**
    #content_based_rec = content_based_rec.drop_duplicates(subset=['Prod Id'])
    #specific_item_based_rec = specific_item_based_rec.drop_duplicates(subset=['Prod Id'])

    return render_template('product.html', 
                           product=product, 
                           content_based_recommendations=content_based_rec, 
                           item_based_recommendations=specific_item_based_rec)


# routes
@routes.route("/home")
def home():
    #random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    #prices = train_data['List Price'].head(5).tolist()

    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        signup_success="True",
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_image_urls,
        random_price=random_prices,
        zip=zip  # Passing zip function
    )
