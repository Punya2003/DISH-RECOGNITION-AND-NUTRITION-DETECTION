import csv
import math
import os
import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = r'C:\Users\WIN 10\OneDrive\Desktop\My new prj\Food-classification-main\static\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define label meaning
label = [
    # 'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
    # 'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
    # 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    # 'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla',
    # 'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder',
    # 'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes',
    # 'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots',
    # 'falafel', 'filet mignon', 'fish and chips', 'foie gras', 'french fries',
    # 'french onion soup', 'french toast', 'fried calamari', 'fried rice', 'frozen yogurt',
    # 'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich', 'grilled salmon',
    # 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog', 'huevos rancheros',
    # 'hummus', 'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
    # 'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette',
    # 'onion rings', 'oysters', 'pad thai', 'paella', 'pancakes', 'panna cotta',
    # 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib', 'pulled pork sandwich',
    # 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa', 'sashimi', 'scallops',
    # 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese', 'spaghetti carbonara',
    # 'spring rolls', 'steak', 'strawberry shortcake', 'sushi', 'tacos', 'octopus balls',
    # 'tiramisu', 'tuna tartare', 'waffles'
    'Burger', 'Butter Naan', 'Chapatti', 'Dal Makhani', 'Dhokla', 'Fried Rice', 'Jalebi', 
     'Kadai Paneer', 'Kulfi', 'Masala Dosa', 'Momos', 'Pani Puri', 'Pav Bhaji', 'Pizza' ,'Samosa'
]

nu_link = 'https://www.nutritionix.com/food/'

# Load the best saved model to make predictions
model_best = None
tensorflow.keras.backend.clear_session()
model_path = 'model_trained_101class.h5'

if os.path.exists(model_path):
    try:
        model_best = load_model(model_path)
        print('Model successfully loaded!')
    except Exception as e:
        print('Error loading the model:', e)
else:
    print(f'Model file {model_path} does not exist.')

start = [0]
passed = [0]
pack = [[]]
num = [0]

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open(r'C:\Users\WIN 10\OneDrive\Desktop\My new prj\Food-classification-main\nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            continue
        name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]

@app.route('/')
def index():
    img = r'C:\Users\WIN 10\OneDrive\Desktop\My new prj\Food-classification-main\static\static\profile.jpg'
    return render_template('index.html', img=img)

@app.route('/recognize')
def magic():
    return render_template('recognize.html', img='static/profile.jpg')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)

    pack[0] = []
    return render_template('recognize.html', img='static/profile.jpg')

@app.route('/predict')
def predict():
    global model_best
    if model_best is None:
        return "Model not loaded. Please check the server logs for more details."

    result = []
    print('total image', num[0])
    for i in range(start[0], num[0]):
        pa = dict()
        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        print('image filepath', filename)
        pred_img = filename
        pred_img = image.load_img(pred_img, target_size=(228, 228))
        pred_img = image.img_to_array(pred_img)
        pred_img = np.expand_dims(pred_img, axis=0)
        pred_img = pred_img / 255.

        pred = model_best.predict(pred_img)
        print("Pred")
        print(pred)

        if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
                math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
            pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

        top = pred.argsort()[0][-3:]
        label.sort()
        _true = label[top[2]]
        pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        x = dict()
        x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
        x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
        x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
        pa['result'] = x
        pa['nutrition'] = nutrition_table[_true]
        pa['food'] = f'{nu_link}{_true}'
        pa['idx'] = i - start[0]
        pa['quantity'] = 100

        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]
    print('successfully packed')
    # Compute the average source of calories
    for p in pack[0]:
        nutrients[0]['value'] += p['nutrition'][0]['value']
        nutrients[1]['value'] += p['nutrition'][1]['value']
        nutrients[2]['value'] += p['nutrition'][2]['value']
        nutrients[3]['value'] += p['nutrition'][3]['value']
        nutrients[4]['value'] += p['nutrition'][4]['value']

        nutrients[0]['value'] /= num[0]
        nutrients[1]['value'] /= num[0]
        nutrients[2]['value'] /= num[0]
        nutrients[3]['value'] /= num[0]
        nutrients[4]['value'] /= num[0]

    return render_template('results.html', pack=pack[0], whole_nutrition=nutrients)

@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')

def index():
    img = url_for('static', filename=r'static\profile.jpg')  # Ensure this path is correct
    return render_template('index.html', img=img)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run() 