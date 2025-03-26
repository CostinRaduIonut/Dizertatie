import keras
import numpy as np
import tensorflow as tf
import string
import cv2 as cv
import uuid
import zipfile
import random
import os
from itertools import islice, product

from PIL import Image
import string
import itertools
import random

litere = string.digits + string.ascii_uppercase + "abdefghnqrt"
classes = [x for x in litere]
number_symbol = [
    [0, 1],
    [0, 1],
    [1, 1]
]
uppercase_symbol = [
    [0, 0],
    [0, 0],
    [0, 1]
]
numbers = [
    [
        [0, 1],
        [1, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
]

characters = {

    '.': [
        [0, 0],
        [0, 0],
        [0, 1]
    ],
    '#': [
        [0, 1],
        [0, 1],
        [1, 1]
    ],

    '0':   [
        [0, 1],
        [1, 1],
        [0, 0]
    ],

    '1':  [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    '2':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    '3':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    '4':   [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    '5':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    '6':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    '7': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    '8':  [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    '9':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],

    'a': [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    'b':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    'c':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    'd':  [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    'e':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    'f':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    'g': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    'h': [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    'i':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
    'j': [
        [0, 1],
        [1, 1],
        [0, 0]
    ],
    'k': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'l': [
        [1, 0],
        [1, 0],
        [1, 0]
    ],
    'm': [
        [1, 1],
        [0, 0],
        [1, 0]
    ],
    'n': [
        [1, 1],
        [0, 1],
        [1, 0]
    ],
    'o': [
        [1, 0],
        [0, 1],
        [1, 0]
    ],
    'p': [
        [1, 1],
        [1, 0],
        [1, 0]
    ],
    'q': [
        [1, 1],
        [1, 1],
        [1, 0]
    ],
    'r': [
        [1, 0],
        [1, 1],
        [1, 0]
    ],
    's': [
        [0, 1],
        [1, 0],
        [1, 0]
    ],
    't': [
        [0, 1],
        [1, 1],
        [1, 0]
    ],
    'u': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'v': [
        [1, 0],
        [1, 0],
        [1, 1]
    ],
    'w': [
        [0, 1],
        [1, 1],
        [0, 1]
    ],
    'x': [
        [1, 1],
        [0, 0],
        [1, 1]
    ],
    'y': [
        [1, 1],
        [0, 1],
        [1, 1]
    ],
    'z': [
        [1, 0],
        [0, 1],
        [1, 1]
    ],
     'A': [
        [1, 0],
        [0, 0],
        [0, 0]
    ],

    'B':  [
        [1, 0],
        [1, 0],
        [0, 0]
    ],

    'C':   [
        [1, 1],
        [0, 0],
        [0, 0]
    ],

    'D':  [
        [1, 1],
        [0, 1],
        [0, 0]
    ],

    'E':  [
        [1, 0],
        [0, 1],
        [0, 0]
    ],
    'F':  [
        [1, 1],
        [1, 0],
        [0, 0]
    ],

    'G': [
        [1, 1],
        [1, 1],
        [0, 0]
    ],

    'H': [
        [1, 0],
        [1, 1],
        [0, 0]
    ],

    'I':  [
        [0, 1],
        [1, 0],
        [0, 0]
    ],
    'J': [
        [0, 1],
        [1, 1],
        [0, 0]
    ],
    'K': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'L': [
        [1, 0],
        [1, 0],
        [1, 0]
    ],
    'M': [
        [1, 1],
        [0, 0],
        [1, 0]
    ],
    'N': [
        [1, 1],
        [0, 1],
        [1, 0]
    ],
    'O': [
        [1, 0],
        [0, 1],
        [1, 0]
    ],
    'P': [
        [1, 1],
        [1, 0],
        [1, 0]
    ],
    'Q': [
        [1, 1],
        [1, 1],
        [1, 0]
    ],
    'R': [
        [1, 0],
        [1, 1],
        [1, 0]
    ],
    'S': [
        [0, 1],
        [1, 0],
        [1, 0]
    ],
    'T': [
        [0, 1],
        [1, 1],
        [1, 0]
    ],
    'U': [
        [1, 0],
        [0, 0],
        [1, 0]
    ],
    'V': [
        [1, 0],
        [1, 0],
        [1, 1]
    ],
    'W': [
        [0, 1],
        [1, 1],
        [0, 1]
    ],
    'X': [
        [1, 1],
        [0, 0],
        [1, 1]
    ],
    'Y': [
        [1, 1],
        [0, 1],
        [1, 1]
    ],
    'Z': [
        [1, 0],
        [0, 1],
        [1, 1]
    ],
}

chars_braille = characters

canShowBraille = False

# model = keras.models.load_model("nn_letters_to_braille_updated")
model = tf.keras.models.load_model(r"nn_letters_to_braille_updated", compile=False)


def detect_subimage(image):
    img_tensor = image[:]
    # img_tensor = tf.keras.preprocessing.image.img_to_array(image)
    img_tensor = tf.expand_dims(img_tensor, -1)
    img_tensor = tf.expand_dims(img_tensor, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor /= 255.0
    # img_tensor = tf.image.resize(img_tensor, (28, 28))
    prediction = model.predict(img_tensor)
    prediction = prediction[0]
    return classes[np.argmax(prediction)], np.max(prediction)


def draw_braille(img, xpos, ypos, pattern):
    dist = 8
    radius = 2
    rows = len(pattern)
    cols = len(pattern[0])
   
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            top_left = (xpos -5, ypos -5)
            bottom_right = (xpos + cols * dist - 2 , ypos + rows * dist-2)
            cv.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)
            color = (255, 255, 255)
            if pattern[i][j] == 1:
                color = (0, 0, 0)
            cv.circle(img, (xpos + j * dist, ypos + i * dist),
                      radius, color, thickness=2)

def string_to_braille(text: str, check_spelling=False, is_audio = False):
    # tts_braille.generate_speech(text)
    # print("TEXT:", text)
    image_width = 512
    image_height = 512
    braille_width = 16
    braille_height = 32
    px = 16
    py = 16
    imgs_braille = []
    img_braille = 255 * np.ones((image_height, image_width, 1))
    y = py * 2
    right = px // 2
    print(f"is_audio = {is_audio}")
    for i, letter in enumerate(text):
        if right < image_width - px * 4:
            right += braille_width + px // 2
        else:
            if y < image_height - py * 4:
                y += braille_height + py // 2
            else:
                y = py * 2
                # ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
                # cv.imwrite(ffname, img_braille)
                imgs_braille.append(img_braille)
                img_braille = 255 * np.ones((image_height, image_width, 1))

            right = px * 2

        if letter in characters.keys():
            draw_braille(img_braille, right, y, characters[letter])
        else:
            right += braille_width - 4

    if len(imgs_braille) > 1:
        merged_img = np.vstack(imgs_braille)
    else:
        merged_img = img_braille
    ffname = "braille_detectat/" + str(uuid.uuid4()) + ".jpg"
    cv.imwrite(ffname, merged_img)

    return text, ffname



def generate_combinations_and_braille():
    pangrams = [
		"the quick brown fox jumps over the lazy dog",
		"pack my box with five dozen liquor jugs",
		"how vexingly quick daft zebras jump",
		"bright vixens jump dozy fowl quack",
		"jackdaws love my big sphinx of quartz",
		"the five boxing wizards jump quickly",
		"waltz nymph for quick jigs vex bud",
		"jumpy frogs wax bemused over quaint dvds",
		"exquisite farm wench gives body jolt to prize stags",
		"crazy frederick bought many very exquisite opal jewels",
		"sixty zippers were quickly picked from the woven jute bag",
		"zany gnomes vexed dwarf jp with builtin flux",
		"jumping zebras vex quincys wild dog pack",
		"foxy diva jennifer lopez wasnt baking my quiche",
		"the wizard quickly jinxed the gnomes before they vaporized",
		"by jove my quick study of lexicography amazed dr finch",
		"victor quickly explained the baffling maze of jigsaw puzzles",
		"amazingly few discotheques provide jukeboxes",
		"judge sylvia quickly picked five dozen mixed bouquets",
		"big dwarves heckle jazzy mimes from quaint victors porches",
		"we kept six dozen glazed jugs of fine quality bourbon",
		"funky ostrich waves to puzzled juror behind six boxes",
		"jumpy squirrels vex badger while zipping over thick pinewood",
		"kooky blue jays never vexed a puzzled giraffe with dancing",
		"glen daftly jinxed the bewitching vampire and won quickly",
		"the puzzled goose vexed the jumping zebra in my quirky yard",
		"mix up five dozen iced jalapeno burgers with extra toppings",
		"a quirky wizard just fixed my broken gnomes bejeweled violin",
		"the jazzy vulture quietly flips dozens of baked muffins",
		"jumping over boxes vexes wild zebras quite a bit",
		"zack quickly mixed five dozen bright blue jugs of honey",
		"victors zombie penguin just fazed the quirky badger",
		"mixing up a box of jalapenos vexed my quirky dog",
		"quincy vexed the frozen wizard by jumping over his box",
		"a jumbo sixpack of pizza vexed my quirky friend joel",
		"zebras quickly judge jumping foxes with frozen mockery",
		"fix the broken zipper on that jazzy velvet quilt now",
		"the jolly queen vexed my quirky brothers frozen dog",
		"jumpy koalas vexed the awkward but dazzling fisherman",
		"a box of five dozen glazed pancakes quickly vexed jim",
		"jay quickly mixed odd frozen gazpacho with beets",
		"the dazzling quiz show baffled my jittery fox",
		"we judged the vampires quirky mix of boxed jellies",
		"six wizards vexed the jumpy queens awkward dog",
		"bizarre frozen kiwi jam puzzled my quirky pet dog",
		"a jumbo pizza box vexed the quirky wizard",
		"jumping zombies quickly vexed my baffled fox",
		"my frozen box of jumping quails vexed the wizard",
		"jinxed foxes quickly vexed my odd zebras jumping habit",
		"quaint gnomes joyfully vexed my big jazzy wizard dog",
	]
    output_folder = "braille_detectat"
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    for sentence in pangrams:
        print(sentence)
        _, img_path = string_to_braille(sentence)
        image_paths.append(img_path)


    zip_filename = "/braille_images_combinations.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))
    
    print(f"Braille images saved in: {zip_filename}")
    return zip_filename

# generate_combinations_and_braille

string_to_braille("hello world")