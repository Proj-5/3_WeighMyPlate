"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response,jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import random
# matplotlib.use('Agg')  # Set the backend to Agg (non-interactive) for Matplotlib

app = Flask(__name__)





@app.route("/")
def hello_world():
    return render_template('index.html')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)

# # function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = predict_img.imgpath    
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"

    
# @app.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath,'uploads',f.filename)
#             print("upload folder is ", filepath)
#             f.save(filepath)
            
#             predict_img.imgpath = f.filename
#             print("printing predict_img :::::: ", predict_img)

#             file_extension = f.filename.rsplit('.', 1)[1].lower()    
#             if file_extension == 'jpg':
#                 process = Popen(["python", "detect.py", '--source', filepath, "--weights","yolov7.pt"], shell=True)
#                 process.wait()
                
                
#             elif file_extension == 'mp4':
#                 process = Popen(["python", "detect.py", '--source', filepath, "--weights","yolov7.pt"], shell=True)
#                 process.communicate()
#                 process.wait()

            
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
#     return render_template('index.html', image_path=image_path)
#     #return "done"



def handle_undefined(obj):
    if isinstance(obj, dict):
        return {key: handle_undefined(value) for key, value in obj.items() if value is not Undefined}
    elif isinstance(obj, list):
        return [handle_undefined(item) for item in obj if item is not Undefined]
    else:
        return obj

def csv_to_json(csv_path):
    # Open the CSV file
    with open(csv_path, 'r') as csv_file:
        # Read the CSV data
        csv_reader = csv.DictReader(csv_file)
        
        # Convert CSV to a list of dictionaries
        data = list(csv_reader)
        
        # Convert the list of dictionaries to a JSON object
        # json_object = json.dumps(data, indent=4)
        
        # return json_object
        return data


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)

            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            if file_extension in ['jpg', 'jpeg', 'png']:
                process = Popen(["python", "detect.py", '--source', filepath, "--weights", "yolov7.pt"], shell=True)
                process.wait()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename

    print("This is the image path:",image_path)

    # Generate a unique string (timestamp or random) to append to the image path
    unique_string = str(int(time.time()))  # or use str(random.randint(1000, 9999))
    
    #make json object
    
    csv_path = 'results.csv'
    json_data = csv_to_json(csv_path)
    # print(json_data)
 
    # Save the plot as an image (e.g., PNG)
    # plt.savefig('g1.png')
    # plt.savefig(f'g1_{unique_string}.png')

    # Optionally, you can also save the plot in other formats like JPEG, PDF, etc.
    # plt.savefig('g1.jpg')

    # Display the plot (optional)
    # plt.show()

    # Save the plot with a unique filename
    # plot_filename = f'g1_{unique_string}.png'
    # plt.savefig(os.path.join('static', plot_filename))

    # Extract object names
    print("Json data looks like this:")
    print(json_data)
    objects = [item["object_name"] for item in json_data]

    # Extract data for each nutrient
    nutrients_calories_carbs = ['Calories', 'Carbs']
    nutrient_data_calories_carbs = [[float(item[nutrient]) for item in json_data] for nutrient in nutrients_calories_carbs]

    nutrients_fat_protein_fiber = ['Protein', 'Fat', 'Fiber']
    nutrient_data_fat_protein_fiber = [[float(item[nutrient]) for item in json_data] for nutrient in nutrients_fat_protein_fiber]

    # Create a 2xN grid for subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

    # Grouped bar chart for Calories and Carbs
    bar_width = 0.35
    index = np.arange(len(objects))

    for i, nutrient in enumerate(nutrients_calories_carbs):
        axs[0].bar(index + i * bar_width, nutrient_data_calories_carbs[i], bar_width, label=nutrient)

        # Add values to the bar chart
        for j, value in enumerate(nutrient_data_calories_carbs[i]):
            axs[0].text(index[j] + i * bar_width, value + 0.5, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    axs[0].set_xlabel('Objects')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Calories and Carbs Information')
    axs[0].set_xticks(index + 2 * bar_width / 2)
    axs[0].set_xticklabels(objects)
    axs[0].legend()

    # Grouped bar chart for Fat, Protein, and Fiber
    # Grouped bar chart for Fat, Protein, and Fiber
    for i, nutrient in enumerate(nutrients_fat_protein_fiber):
        axs[1].bar(index + i * bar_width, nutrient_data_fat_protein_fiber[i], bar_width, label=nutrient)

        # Add values to the bar chart with vertical alignment adjustment
        for j, value in enumerate(nutrient_data_fat_protein_fiber[i]):
            axs[1].text(index[j] + i * bar_width, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    axs[1].set_xlabel('Objects')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Fat, Protein, and Fiber Information')
    axs[1].set_xticks(index + 2 * bar_width / 2)
    axs[1].set_xticklabels(objects)
    axs[1].legend()


    plt.tight_layout()

    # Save the plot with a unique filename
    plot_filename = f'g1_{unique_string}.png'
    plt.savefig(os.path.join('static', plot_filename))

    # Optionally, close the plot
    plt.close()
    # plt.show()


    # return render_template('index.html', image_path=image_path, g1_plot='g1.png')
    # return render_template('index.html', image_path=image_path, g1_plot=f'g1_{unique_string}.png')
    return render_template('index.html', image_path=image_path, g1_plot=plot_filename)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','yolov7.pt', source='local')
    model.eval()
    app.run(host="127.0.0.1", port=args.port)  # debug=True causes Restarting with stat

