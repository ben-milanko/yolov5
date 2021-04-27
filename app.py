from flask import Flask, request
app = Flask(__name__)

# flask run --host=0.0.0.0

@app.route('/')
def index():
    return 'Index Page'

@app.route('/alta', methods=['GET', 'POST'])
def alta():
    if request.method == 'GET':
        args = request.form
        # print(args)

    return args
