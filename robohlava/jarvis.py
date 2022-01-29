"""
function JARVIS
 - Answers questions in english using wolfram alpha
TO RUN THIS CODE YOU NEED TO INSTALL THE FOLLOWING:
    - $ sudo apt-get install portaudio19-dev python-pyaudio
    - $ pip install PyAudio
    - $ pip install SpeechRecognition
"""

import wolframalpha
import speech_recognition as sr
from espeakng import ESpeakNG
import time
import pyttsx3

# User client key - get your own when you register at https://www.wolframalpha.com/
# This key will be disabled after some time
client = wolframalpha.Client("6YJQ9J-96JQX2JT5E")

# Setting STT and TTS methods
r = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

while True:
    with sr.Microphone() as source:
        print('Ask something: ')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print('You said: {}?'.format(text))
        except:     # Except some error occurs
            text = None
            print('Sorry could not recognize your voice')

        if format(text) == 'exit':  # Exit from function
            break

        if text is not None:
            try:
                res = client.query(format(text))
                answer = next(res.results).text
            except: # Except some error occurs
                answer = "I don't know the answer to that"
            print(answer)
            engine.say(answer)
            engine.runAndWait()
        answer = None
