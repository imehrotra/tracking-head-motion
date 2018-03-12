import socket,os,sys,thread,argparse,threading
#import parse
import time
#from pynput.mouse import Button, Controller
#from pynput.keyboard import Key, Controller
# from pynput import mouse, keyboard
import classify2
from sklearn import neighbors, datasets, preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import Metrics as met
import csv
import numpy as np
'''
Presses right arrow key
'''
def keyPressR():
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.right)
    keyB.release(keyboard.Key.right)

def keyPressL():
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.left)
    keyB.release(keyboard.Key.left)

'''
input x and y coordinates on screen to click
'''
def mouseClick(x,y):
    cursor = mouse.Controller()
    
    # Set pointer position
    cursor.position = (x,y)
    #print('Now we have moved it to {0}'.format(mouse.position))

    # click, set 2 for double click
    cursor.click(mouse.Button.left, 1)

'''
moves mouse
'''
def mouseMove(x,y):
    cursor = mouse.Controller()
    
    # Set pointer position
    cursor.position = (x,y)

def flush(rotz,roty,uaccelz,xaccl):
    rotz = []
    roty = []
    uaccelz = []
    xaccl = []

def threadAPI(conn, clientaddr, x, y):
    Y = []
    Z = []
    knn, scaler = classify2.classify_with_window(Y, Z) # Change this to classify_with_window for other data...
    # knn = classify2.load('data.knn')
    # scaler = classify2.load('data.scaler')
    #count = 0
    #test = 0
    #test2 = 0
    

 #   ld = False
#    rd = False

    roty = []
    rotz = []
    uaccelz = []
    xaccl = []
    n_xaccl = []
    n_uaccelz = []
    n_roty = []
    n_rotz = []
    while 1:
        # @Isha, can you check the size of data? 
        data = conn.recv(999999)
        if data == '\n':
            continue
        if data == "All done":
            n_roty = np.array(roty)
            n_rotz = np.array(rot)
            n_uaccelz = np.array(uaccelz)
            n_xaccl = np.array(cxaccl)
            tmp = []
            tmp.append(n_xaccl.std())
            tmp.append(n_uaccelz.std())
            tmp.append(n_xaccl.max())
            tmp.append(np.median(n_roty))
            tmp.append(n_roty.mean())
            tmp.append(n_rotz.mean())
            tmp.append(n_rotz.min())
            Features = []
            Features.append(tmp)

            print "features", Features
            scaler.transform(Features)
            label = knn.predict(Features)
            print("result:", label)
            if label == "right down" or label == "right up":
                print "RIGHT TILT"
            elif label == "left up" or label == "left down":
                print "LEFT TILT"
            else:
                print "NOISY"
            roty = []
            rotz = []
            uaccelz = []
            xaccl = []
            n_xaccl = []
            n_uaccelz = []
            n_roty = []
            n_rotz = []
        else:
            line = data.split(",")
            #print("Line 0: " + line[0])
            roty.append(float(line[0]))
            rotz.append(float(line[1]))
            uaccelz.append(float(line[2]))
            xaccl.append(float(line[3]))


    conn.close()


    
# Main function of Server
def main():

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',help = 'The port server listens on')
    parser.add_argument('-x','--x',help = 'The x coordinate of click')
    parser.add_argument('-y','--y',help = 'The y coordinate of click')
    args = parser.parse_args()

    if args.port is None:
        port = 8081
    else:
        port = int(args.port)

    if args.x:
        x = args.x
    else:
        x = 1000

    if args.y:
        y = args.y
    else:
        y = 800
        
    #hostname
    host = ''

    # create a server socket with host and port
    while 1:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = server.bind((host, port))
            server.listen(5) #queue will hold 5 pending connections
            break;
        # Tries another port if specified port is unavailable
        except:
            if port > 65535:
                port = 8081
            else:
                port += 1
            pass
    
    #port server actually runs on
    print "Server Running on localhost:",port
   
    # accept client connection
    while 1:
        conn, clientaddr = server.accept()

        # create a thread to handle request
        thread.start_new_thread(threadAPI, (conn, clientaddr,x,y))
                   
    server.close()
    return 0

if __name__ == '__main__':
    main()
