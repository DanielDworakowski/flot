# Flask libraries
import logging
from random import randint
from flask import Flask, render_template, session, request, Response
from flask_ask import Ask, statement, question
import jsonpickle

# Image libraries
import numpy as np
from PIL import Image
import cv2

# Other libraries
import datetime, time
import threading

import torch.multiprocessing as mp
import torch

import os
import atexit

# Global Variables
home_path = '/home/kelvin'
frame = None
image = None

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

app.secret_key = 'ravioli ravioli give me the formuoli'

@app.route('/updateImage', methods=['POST'])
def image_update():
    r = request
    nparr = np.fromstring(r.data, np.uint8)

    global frame
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(frame.shape[1], frame.shape[0])}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@ask.launch
def welcome():
    msg = render_template('welcome')
    reprompt = render_template('prompt')

    return question(msg).reprompt(reprompt)

@ask.intent("SelfieIntent")
def selfie():
    msg = None

    global frame, image

    if frame is not None:
        image = frame
        msg = render_template('selfie_ok')
    else:
        msg = render_template('selfie_fail')

    return question(msg)

@ask.intent("ShowIntent", mapping={'name': 'Name'})
def showImage(name):
    global image
    msg = None

    # Show previous image
    if name.lower in ['last', 'previous', 'that']:
        if image is not None:
            Image.fromarray(image).show()
            msg = render_template('show_image')
        else:
            msg = render_template('show_fail')

    # Find image in home folder
    else:
        filt_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        imgPath = home_path + '/' + filt_name + ".png"
        if os.path.isfile(imgPath):
            Image.open(imgPath).show()
            msg = render_template('show_image')
        else:
            msg = render_template('find_fail')

    return question(msg)

@ask.intent("NameIntent", mapping={'name': 'Name'})
def nameImage(name):
    global image
    msg = None

    if image is not None:
        filt_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip()

        if os.path.isfile(home_path + '/' + filt_name + ".png"):
            msg = render_template('name_fail', name=name)
        else:
            try:
                Image.fromarray(image).save(home_path + '/' + filt_name + ".png")

                msg = render_template('name_image', name=name)
            except:
                msg = render_template('name_fail', name=name)
    else:
        msg = render_template('name_none')

    return question(msg)

@ask.intent("TwitterIntent", mapping={'name': 'Name'})
def tweetImage(name):
    global image
    msg = None

    # Tweet last image
    if name.lower in ['last', 'previous', 'that']:
        if image is not None:
            try:
                # Open and tweet last image

                msg = render_template('tweet_ok')
            except:
                msg = render_template('tweet_fail')
        else:
            msg = render_template('find_fail')

    # Find image in home folder
    else:
        filt_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        imgPath = home_path + '/' + filt_name + ".png"
        if os.path.isfile(imgPath):
            try:
                # Open and tweet image from path

                msg = render_template('tweet_ok')
            except:
                msg = render_template('tweet_fail')
        else:
            msg = render_template('find_fail')

    return question(msg)

@ask.intent("YesIntent")
def yes():
    msg = render_template('yes_retake')
    return question(msg)

@ask.intent("NoIntent")
def no():
    msg = render_template('no_retake')
    return question(msg)

@ask.intent("StopIntent")
def stop():
    msg = render_template('stop')
    return statement(msg)

app.run(debug=False)
