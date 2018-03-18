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

# Twitter
import twitter

# Other libraries
import datetime, time
import threading

import torch.multiprocessing as mp
import torch

import string
import os
import atexit

# Global Variables
home_path = os.path.expanduser("~")
frame = None
image = None

# Twitter setup
consumer_key = 'iTl0HLBQxe8V4JksVXwu8Xwus'
consumer_secret = 'o7I8GEd8JesXN2m27bDpmNtT4ZewvNpJ9axGZCiNQPNHmTHFlG'
access_token_key = '974666982678294529-0Ho7jjlHkjVblXZeahFuBtueSZ2LO6n'
access_token_secret = 'IxvugPcrPmjoiPlA78h1zWToctLoR3dr0AXxsTCCU3Knd'

# Helper functions
def format_filename(s):
    valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_')
    return filename


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

@ask.intent("ShowIntent", mapping={'name': 'Name', 'previous': 'Previous'})
def showImage(name, previous):
    global image
    msg = None

    print("Name: {}".format(name))
    print("Previous: {}".format(previous))

    # Show previous image
    if isinstance(previous, unicode):
        if (previous.lower() in ['last', 'previous', 'that']) and (image is not None):
            Image.fromarray(image).show()
            msg = render_template('show_image')
        else:
            msg = render_template('show_fail')

    # Find image in home folder
    elif isinstance(name, unicode):
        name = str(name)
        filt_name = format_filename(name)
        imgPath = home_path + '/' + filt_name + ".png"
        if os.path.isfile(imgPath):
            Image.open(imgPath).show()
            msg = render_template('show_image')
        else:
            msg = render_template('find_fail')

    # Couldn't match anything
    else:
        msg = render_template('find_fail')

    return question(msg)

@ask.intent("NameIntent", mapping={'name': 'Name'})
def nameImage(name):
    global image
    msg = None

    # If fibi has already taken a selfie
    if image is not None:

        print(name)
        print(type(name))
        print('Received name: {}'.format(name))

        # If name is provided
        if isinstance(name, unicode):
            name = str(name).lower()
            filt_name = format_filename(name)
            print('Filtered name: {}'.format(filt_name))

            # If image with that filename already exists
            if os.path.isfile(home_path + '/' + filt_name + ".png"):
                msg = render_template('name_fail', name=name)

            # Else, try saving under that name
            else:
                try:
                    Image.fromarray(image).save(home_path + '/' + filt_name + ".png")

                    msg = render_template('name_image', name=name)
                except:
                    msg = render_template('name_fail', name=name)

        # Else, try another name
        else:
            msg = render_template('name_no')
    # Else, prompt user to take image
    else:
        msg = render_template('name_none')

    return question(msg)
#
# @ask.intent("TwitterIntent", mapping={'name': 'Name', 'previous': 'Previous'})

@ask.intent("TwitterIntent", mapping={'name': 'Name'})
def tweetImage(name):
    global image, consumer_key, consumer_secret, access_token_key, access_token_secret
    msg = None

    status = 'Posted by Fibi!'
    twitterApi = twitter.Api(consumer_key=consumer_key,
                             consumer_secret=consumer_secret,
                             access_token_key=access_token_key,
                             access_token_secret=access_token_secret)

    print('Received name: {}'.format(name))
    print(type(name))

    # print('Received previous: {}'.format(previous))
    # print(type(previous))
    #
    # # Tweet last image
    # if isinstance(previous, str) and (previous.lower() in ['last', 'previous', 'that']):
    #     if image is not None:
    #         try:
    #             # Save last image in a temporary file
    #             print('Attempting to tweet...')
    #             imgPath = home_path + '/latestImage.png'
    #             print('Tweet successful')
    #             Image.fromarray(image).save(imgPath)
    #
    #             # Open and tweet last image
    #             twitterApi.PostUpdate(status, media=imgPath)
    #             msg = render_template('tweet_ok')
    #         except:
    #             msg = render_template('tweet_fail')
    #     else:
    #         msg = render_template('find_fail')

    # Tweet specified image in home folder
    # elif isinstance(name, unicode):
    if isinstance(name, unicode):
        name = str(name).lower()
        filt_name = format_filename(name)
        print('Filtered name: {}'.format(filt_name))

        imgPath = home_path + '/' + filt_name + ".png"
        if os.path.isfile(imgPath):
            try:
                # Open and tweet image from path
                print('Attempting to tweet...')
                f = open(imgPath, 'rb')
                twitterApi.PostUpdate(status, media=f)
                print('Tweet successful')
                msg = render_template('tweet_ok')
            except Exception as e:
                print(e)
                msg = render_template('tweet_fail')
        else:
            msg = render_template('find_fail')

    # Failed to find image with that name
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

app.run(debug=False, host='0.0.0.0')
