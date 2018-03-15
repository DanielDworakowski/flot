# Flask libraries
import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

# Image libraries
import numpy as np
from PIL import Image
import cv2

# Other libraries
import datetime, time
import threading

import torch.multiprocessing as mp
import torch

client = None
image = None

class flotapp(mp.Process):

    app = Flask(__name__)
    ask = Ask(app, "/")
    logging.getLogger("flask_ask").setLevel(logging.DEBUG)

    def __init__(self, cl):
        super(flotapp, self).__init__()
        global client
        client = cl

    @ask.launch
    def welcome():
        msg = render_template('welcome')
        reprompt = render_template('prompt')

        return question(msg).reprompt(reprompt)

    @ask.intent("SelfieIntent")
    def selfie():
        msg = None

        # Grab frames from VideoStreamClient.sharedFrame
        global client
        # if client is not None:
        #     image = client.clone().numpy()
        global image

        if image is not None:
            Image.fromarray(image).show()

            ts = time.time()
            cur_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')
            Image.fromarray(image).save(cur_stamp + ".png")

            msg = render_template('selfie_ok')
        else:
            msg = render_template('selfie_fail')

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

    def updateImage(self, img):
        self.image = img
        global image
        image = img

    def run(self):
        self.app.run(debug=True)
