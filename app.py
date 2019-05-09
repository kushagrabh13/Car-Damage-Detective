import os
import _pickle as cPickle
from flask import Flask, render_template, request
from auto_insurance_base_pipeline import AutoInsurance
import requests

import json
import urllib
import h5py
import pickle as pk
import numpy as np

from os.path import join, dirname, realpath
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash, Response
from werkzeug.utils import secure_filename

import keras.backend as K

ai = AutoInsurance()

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) 

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

# check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return render_template('index.html', result=None)

@app.route('/<a>')
def available(a):
	flash('{} coming soon!'.format(a))
	return render_template('index.html', result=None, scroll='third')

@app.route('/assessment')
def assess():
	return render_template('index.html', result=None, scroll='third')

@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
	K.clear_session()
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('assess'))
		
		file = request.files['file']

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(url_for('assess'))

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename) # used to secure a filename before storing it directly on the filesystem
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# return redirect(url_for('uploaded_file',
			#                         filename=filename))
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			model_results = ai.predict(filepath)

			return render_template('results.html', prediction=model_results, scroll='third', filename=filename)
	
	flash('Invalid file format - please try your upload again.')
	return redirect(url_for('assess'))

@app.route('/uploads/<filename>')
def send_file(filename):
	return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8888, debug=False)