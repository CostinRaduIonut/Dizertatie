# Dizertatie

## 1. Project structure

The project is divided into two tasks: detection and recognition of Braille characters, and converting them into the Latin alphabet.

```
/utils ---> ['image_processing.py', 'video_to_images.py']
 ^
 |
 |
ROOT ---> /detection ---> ['main.py', 'train.py', 'test.py']
 |             |
 |             v
 |            /dataset ---> /generated_images
 v
 /recognition ---> ['main.py', 'train.py', 'test.py']
        |
        |
        v
    /dataset ---> ['/a', '/b', ..., '/z']
```
