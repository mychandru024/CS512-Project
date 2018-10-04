import sift

print("CS512-CV Project | SIFT Implementation\nPress esc to see the exit a window.\nEx: data/bolt.jpg")
filename = input("Please enter file path: ")
sift_engine = sift.SIFT(filename)