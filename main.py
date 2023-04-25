from flask import Flask

from website import create_app
import os

app = Flask(__name__)

# Set the FLASK_APP environment variable
os.environ['FLASK_APP'] = 'main.py'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))