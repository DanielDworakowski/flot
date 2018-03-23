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
import asyncio

import torch.multiprocessing as mp
import torch

import string
from random import *
import os
import atexit

# Global Variables
home_path = os.path.expanduser("~")
frame = None
image = None
username = None
greeting_nums = 24
bye_nums = 6

# Twitter setup
consumer_key = 'iTl0HLBQxe8V4JksVXwu8Xwus'
consumer_secret = 'o7I8GEd8JesXN2m27bDpmNtT4ZewvNpJ9axGZCiNQPNHmTHFlG'
access_token_key = '974666982678294529-0Ho7jjlHkjVblXZeahFuBtueSZ2LO6n'
access_token_secret = 'IxvugPcrPmjoiPlA78h1zWToctLoR3dr0AXxsTCCU3Knd'

# Flask setup
app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)
app.secret_key = 'ravioli ravioli give me the formuoli'

# Twitter Image Post Function
def imagePost(imgPath):
    status = 'Posted by Fibi!'

    twitterApi = twitter.Api(consumer_key=consumer_key,
                             consumer_secret=consumer_secret,
                             access_token_key=access_token_key,
                             access_token_secret=access_token_secret)
    try:
        f = open(imgPath, 'rb')
        print('Attempting to tweet...')
        twitterApi.PostUpdate(status, media=f)
        print('Tweet Success')
    except:
        print('Failed to tweet with image at ' + imgPath)

# Helper functions
def format_filename(s):
    valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_')
    return filename

def voice_mod(s):
    if isinstance(s, str):
        return "<speak><prosody pitch='+33.3%'>" + s + '</prosody></speak>'

# Flask callbacks

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

# Flask Ask callbacks

@ask.launch
def welcome():
    msg = voice_mod(render_template('welcome'))
    reprompt = voice_mod(render_template('prompt'))

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

    msg = voice_mod(msg)

    return question(msg)

@ask.intent("UsernameIntent", mapping={'name': 'Name'})
def username(name):
    global username

    if isinstance(name, str):
        username = name
        msg = render_template('username', name=name)
    else:
        msg = render_template('username_fail')

    msg = voice_mod(msg)
    return question(msg)

@ask.intent("GreetingIntent")
def greeting():
    global username, greeting_nums
    name = username

    if not isinstance(name, str):
        name = ''

    msg = render_template('greeting_'+ str(randint(1, greeting_nums)), name=name)
    msg = voice_mod(msg)

    return question(msg)

@ask.intent("ExitIntent")
def bye():
    global username, bye_nums
    name = username

    if name is None:
        name = ''

    msg = render_template('bye_'+ str(randint(1, bye_nums)), name=name)
    msg = voice_mod(msg)

    reprompt = render_template('bye_reprompt')
    reprompt = voice_mod(reprompt)
    username = None

    return question(msg).reprompt(reprompt)

@ask.intent("ShowIntent", mapping={'name': 'Name', 'previous': 'Previous'})
def showImage(name, previous):
    global image
    msg = None

    print("Name: {}".format(name))
    print("Previous: {}".format(previous))

    # Show previous image
    if isinstance(previous, str):
        if (previous.lower() in ['last', 'previous', 'that']) and (image is not None):
            Image.fromarray(image).show()
            msg = render_template('show_image')
        else:
            msg = render_template('show_fail')

    # Find image in home folder
    elif isinstance(name, str):
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

    msg = voice_mod(msg)

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
        if isinstance(name, str):
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

    msg = voice_mod(msg)

    return question(msg)
#
# @ask.intent("TwitterIntent", mapping={'name': 'Name', 'previous': 'Previous'})

@ask.intent("TwitterIntent", mapping={'name': 'Name', 'previous': 'Previous'})
def tweetImage(name, previous):
    global image, consumer_key, consumer_secret, access_token_key, access_token_secret
    msg = None

    status = 'Posted by Fibi!'

    print('Received name: {}'.format(name))
    print(type(name))

    print('Received previous: {}'.format(previous))
    print(type(previous))

    # Tweet last image
    if isinstance(previous, str) and (previous.lower() in ['last', 'previous', 'that']):
        if image is not None:
            try:
                print('Tweeting last picture!')

                # Save last image in a temporary file
                imgPath = home_path + '/latestImage.png'
                Image.fromarray(image).save(imgPath)

                time.sleep(2)
                threading.Thread(target=imagePost,args=(imgPath,)).start()

                msg = render_template('tweet_ok')
            except Exception as e:
                print(e)
                print('Failed to tweet at this time')
                msg = render_template('tweet_fail')
        else:
            print('Could not find last taken image')
            msg = render_template('find_fail')

    # Tweet specified image in home folder
    # elif isinstance(name, unicode):
    elif isinstance(name, str):
        name = str(name).lower()
        filt_name = format_filename(name)
        print('Filtered name: {}'.format(filt_name))

        imgPath = home_path + '/' + filt_name + ".png"
        if os.path.isfile(imgPath):
            try:
                print('Tweeting picture ' + filt_name)
                # Open and tweet image from path
                # print('Attempting to tweet...')
                # f = open(imgPath, 'rb')
                # twitterApi.PostUpdate(status, media=f)
                # print('Tweet successful')

                threading.Thread(target=imagePost,args=(imgPath,)).start()

                msg = render_template('tweet_ok')
            except Exception as e:
                print(e)
                msg = render_template('tweet_fail')

    # Failed to find image with that name
    else:
        msg = render_template('find_fail')

    msg = voice_mod(msg)
    return question(msg)

@ask.intent("AMAZON.YesIntent")
def yes():
    msg = render_template('yes')
    msg = voice_mod(msg)
    return question(msg)

@ask.intent("AMAZON.NoIntent")
def no():
    msg = render_template('no')
    msg = voice_mod(msg)
    return question(msg)

@ask.intent("AMAZON.StopIntent")
def stop():
    msg = render_template('stop')
    msg = voice_mod(msg)
    return statement(msg)

@ask.intent("AMAZON.CancelIntent")
def cancel():
    msg = render_template('stop')
    msg = voice_mod(msg)
    return statement(msg)

@ask.intent("AMAZON.HelpIntent")
def  help():
    msg = render_template('help')
    msg = voice_mod(msg)
    return question(msg)

@ask.intent("AboutIntent")
def  about():
    msg = render_template('about')
    reprompt = render_template('about_reprompt')
    msg = voice_mod(msg)
    reprompt = voice_mod(reprompt)
    return question(msg).reprompt(reprompt)

# Run server at the end when everything is ready to go
app.run(debug=False, host='0.0.0.0')
