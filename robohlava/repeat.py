"""
TO RUN THIS CODE YOU NEED TO INSTALL THE FOLLOWING:
    - $ sudo apt-get install portaudio19-dev python-pyaudio
    - $ pip install PyAudio
    - $ pip install SpeechRecognition
For STT we use google recognizer
For TTS we use googleTTS as woman and eSpeak as man
"""

import speech_recognition as sr
from espeakng import ESpeakNG
from gtts import gTTS
import playsound
from subprocess import check_output

# Pick gender man/woman
gender = 'woman'

r = sr.Recognizer()
esng = ESpeakNG()
esng.voice = 'czech'
esng.pitch = 50
esng.speed = 150
text = ''

while True:
    with sr.Microphone() as source:
        print('Speak Anything: ')
        audio = r.listen(source)
        try:
            sound = r.recognize_google(audio, language="sk-SK")
            text = format(sound)
        except:
            print('Sorry could not recognize your voice')
        print(text)

        if text == 'stop':
            break

        if gender is 'woman':
            tts = gTTS(text, lang='sk')
            filename = 'voice.mp3'
            tts.save(filename)
            playsound.playsound(filename)
        else:
            speak = check_output(['espeak', text, "-vsk", "-p30"])
