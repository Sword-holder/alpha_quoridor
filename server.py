from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route('/', methods=['GET'])
def init_game():
    return render_template('game.html')

@app.route('/getAction', methods=['POST'])
def get_action():
    pass

if __name__ == '__main__':
    app.run(debug=True)
