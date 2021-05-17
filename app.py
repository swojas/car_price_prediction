from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
	Fuel_Type_Diesel=0
	if request.method == 'POST':
		Present_Price=float(request.form['curr_price'])
		Kms_Driven=int(request.form['kmph'])
		Kms_Driven2=np.log(Kms_Driven)
		Owner=int(request.form['owner'])
		Age = int(request.form['age'])
		Fuel_Type_Petrol=request.form['fuel_type']

		if(Fuel_Type_Petrol=='Petrol'):
			Fuel_Type_Petrol=1
			Fuel_Type_Diesel=0
		elif(Fuel_Type_Petrol=='Diesel'):
			Fuel_Type_Petrol=0
			Fuel_Type_Diesel=1
		else:
			Fuel_Type_Petrol=0
			Fuel_Type_Diesel=0

		Seller_Type_Individual=request.form['seller']	
		Transmission_Mannual=request.form['transmission']

		prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Age,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
		output=round(prediction[0],2)
		if output<0:
			return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
		else:
			return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
	else:
		return render_template('index.html')


	return render_template('index.html', prediction_text='Selling price of bike/scooter {} Lacs'.format(output))

if __name__ == '__main__':
	app.run(debug=True, port='0000')

