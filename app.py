from flask import Flask, render_template

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)