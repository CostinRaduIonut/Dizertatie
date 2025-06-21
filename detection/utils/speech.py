from gtts import gTTS
import uuid

def generate_speech(text_input: str):
    fname = f"/{str(uuid.uuid4())}.mp3"
    tts = gTTS(text=text_input, lang='ro')
    tts.save(fname)
    return fname
