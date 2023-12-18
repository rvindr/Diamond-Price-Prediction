from flask import Flask, render_template, jsonify, request, url_for
from src.pipline.prediction import CustomData, PredictPipline
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict_price():

    if request.method == 'POST':
        try:
            data = CustomData(
                # get the values from the form using request.form
                carat=float(request.form['carat']),
                depth = float(request.form['depth']),
                table = float(request.form['table']),
                x = float(request.form['x']),
                y = float(request.form['y']),
                z = float(request.form['z']),
                cut = str(request.form['cut']),
                color = str(request.form['color']),
                clarity = str(request.form['clarity'])
            )
            df =data.get_data_as_dataframe()
            model = PredictPipline()
            pred = model.predict(df)

            result = round(pred[0], 2)

            return render_template('result.html', result = result, feature =np.array(df))
            
        except ValueError:
            return render_template('index.html', error = True)
    else:
        return render_template('index.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0')
