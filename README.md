# Similarface
Find similarities between any amount of images with any amount of faces.
Stores 128d vectors of faces in sqlite so that 2nd+n run is faster.

# Usage
- Create the directories as per get_imgs() and match_imgs()
- Add images with faces in those directories
- Create the sqlite3 database as per schema
- Install all the dependencies (dlib, keras etc.)
- Google the network resnet50_coco_best_v2.0.1.h5 to download (too big for github)
- run the program with python3

# Emotions and Objects
...are turned off by default by you can turn them on and detect emotions and objects in your images as well if desired. These are not stored yet in Sqlite db. Caution: its very CPU/GPU heavy to detect emotions and objects.

# Pickle tips to Sqlite3
It is possible to dump and load numpy arrays, OpenCV images etc. to Sqlite by using the pickle seriealization functions pickle.dumps() and pickle.loads().

Notice that these two functions are different from the filepointer requiring pickle.load() and pickle.dump() - which we take advantage of:

They load and dump the byte code objects into RAM and uses that as file.
