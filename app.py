from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import io
import base64
import os

app = Flask(__name__, static_folder='static')

# Load the trained model
model = joblib.load('lr_model.pkl')

# Initialize a DataFrame to store prediction history
history_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global history_df
    data = request.form.to_dict()
    features = [float(data['Open']), float(data['High']), float(data['Low']), float(data['Close']), float(data['Volume'])]
    features = np.array(features).reshape(1, -1)
    
    # Predict using the loaded model
    prediction = model.predict(features)[0]
    
    # Save the prediction to the history DataFrame
    new_entry = {**data, 'Prediction': prediction}
    history_df = history_df.append(new_entry, ignore_index=True)
    history_df.to_excel('prediction_history.xlsx', index=False)

    # Prepare data for ROC curve (using last 10 entries from history_df for demonstration)
    if len(history_df) > 1:
        y_true = history_df['Close'].apply(lambda x: 1 if float(x) > 0 else 0).values
        y_scores = history_df['Prediction'].values
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        roc_curve_data = base64.b64encode(buf.getvalue()).decode('ascii')
        buf.close()
    else:
        roc_curve_data = None
    
    return render_template('index.html', prediction=prediction, roc_curve_data=roc_curve_data)

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
