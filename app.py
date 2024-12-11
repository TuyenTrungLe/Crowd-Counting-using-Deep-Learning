from flask import Flask, render_template, request, jsonify

from final_model import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def estimate():
    file = request.files['image']
    if file:
        prediction = predict(file)
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'No image uploaded'})

if __name__ == "__main__":
    app.run(debug=True)