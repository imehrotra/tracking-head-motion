import socket,os,sys,thread,argparse,threading
#import parse
import time
from pynput import mouse, keyboard
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller
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
    
'''
Presses left arrow key
'''
def keyPressL():
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.left)
    keyB.release(keyboard.Key.left)

def keyPressAltTab():
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.alt)
    keyB.press(keyboard.Key.tab)
    keyB.release(keyboard.Key.tab)
    keyB.release(keyboard.Key.alt)


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

'''
Spawned thread handling classifying head motions
Triggers computer event based on classification result
'''
def threadAPI(conn, clientaddr, x, y):
    #initialize classifier
    Y = []
    Z = []
    knn, scaler = classify2.classify(Y, Z)

    #Classification variables
    roty = []
    rotz = []
    uaccelz = []
    xaccl = []
    n_xaccl = []
    n_uaccelz = []
    n_roty = []
    n_rotz = []

    #Loop for listening server
    while 1:
        data = conn.recv(999999)

        if data == '\n':
            continue

        line = data.split(",")

        #client is done sending all data for one classification test
        if line[0] == "All done":

            #extract relevant features to run test on
            n_roty = np.array(roty)
            n_rotz = np.array(rotz)
            n_uaccelz = np.array(uaccelz)
            n_xaccl = np.array(xaccl)
            tmp = []
            tmp.append(n_xaccl.std())
            tmp.append(n_uaccelz.std())
            tmp.append(n_xaccl.max())
            tmp.append(np.median(n_rotz))
            tmp.append(n_roty.mean())
            tmp.append(n_rotz.mean())
            tmp.append(n_rotz.min())
            Features = []
            Features.append(tmp)
            print "features", Features

            #Run classification algorithm on features
            scaler.transform(Features)
            label = knn.predict(Features)
            print("result:", label)

            #computer event based on classification decision
            if label == "right down" or label == "right up":
                print "RIGHT TILT"

                keyPressAltTab()
                time.sleep(2)
                keyPressR()

                # Alternative: move mouse
                # mouseMove(843,447)

            elif label == "left up" or label == "left down":
                print "LEFT TILT"

                keyPressAltTab()
                time.sleep(2)
                keyPressL()

                # Alternative: move mouse
                #mouseMove(443,447)

            else:
                print "NOISY"

            #clear classification variables for next test
            roty = []
            rotz = []
            uaccelz = []
            xaccl = []
            n_xaccl = []
            n_uaccelz = []
            n_roty = []
            n_rotz = []
        else:
            roty.append(float(line[0]))
            rotz.append(float(line[1]))
            uaccelz.append(float(line[2]))
            xaccl.append(float(line[3]))

    #close socket connection
    conn.close()


    
# Main function of Server
def main():

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',help = 'The port server listens on')
    args = parser.parse_args()

    if args.port is None:
        port = 8081
    else:
        port = int(args.port)
        
    #hostname, localhost
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
