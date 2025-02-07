from flask import Flask
from config import Config
from database import db
from routes import routes

app = Flask(__name__)

# Database configuration
app.secret_key = "12345678"
app.config.from_object(Config)

db.init_app(app)

app.register_blueprint(routes, url_prefix='/')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001)
