# Flask libraries
import logging
from random import randint
from flask import Flask, render_template, g
from flask_ask import Ask, statement, question, session

# Image libraries
import numpy as np
from PIL import Image
import cv2

# Other libraries
import datetime
import threading

import torch.multiprocessing as mp
import torch

class flotapp(mp.Process):

    app = Flask(__name__)
    ask = Ask(app, "/")
    logging.getLogger("flask_ask").setLevel(logging.DEBUG)

    def __init__(self):
        super(flotapp, self).__init__()

    @ask.launch
    def welcome():
        msg = render_template('welcome')
        reprompt = render_template('prompt')

        if g.client is not None:
            # with app.app_context():
            print('Client: {}'.format(g.client[0]))
        else:
            print('g.client is None')

        return question(msg).reprompt(reprompt)

    @ask.intent("SelfieIntent")
    def selfie():
        msg = None

        image = None

        if g.client is not None:
            # with app.app_context():
            print('Client: {}'.format(g.client[0]))
        else:
            print('g.client is None')

        if image is not None:
            # cur_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')
            # Image.fromarray(image).save(cur_stamp + ".jpeg")

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

    def run(self):
        self.app.run(debug=True)

    def updateClient(self, client):
        with self.app.app_context():
            g.client = client
