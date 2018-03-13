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

def keyPressR():
    '''
    Presses right arrow key
    '''
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.right)
    keyB.release(keyboard.Key.right)

def keyPressL():
    '''
    Presses left arrow key
    '''
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.left)
    keyB.release(keyboard.Key.left)


def mouseClick(x,y):
    '''
    input x and y coordinates on screen to click
    '''
    cursor = mouse.Controller()
    
    # Set pointer position
    cursor.position = (x,y)
    #print('Now we have moved it to {0}'.format(mouse.position))

    # click, set 2 for double click
    cursor.click(mouse.Button.left, 1)


def mouseMove(x,y):
    '''
    moves mouse
    '''
    cursor = mouse.Controller()
    
    # Set pointer position
    cursor.position = (x,y)

def flush(rotz,roty,uaccelz,xaccl):
    '''
    Flush queue of data
    '''
    rotz = []
    roty = []
    uaccelz = []
    xaccl = []

def threadAPI(conn, clientaddr, x, y):
    Y = []
    Z = []
    knn, scaler = classify2.classify_with_window(Y, Z) 
    roty = []
    rotz = []
    uaccelz = []
    xaccl = []
    cur_roty = []
    cur_rotz = []
    cur_uaccelz = []
    cur_xaccl = []

    ld = False
    rd = False


    while 1:
        data = conn.recv(999999)
        if data == '\n':
            continue
        line = data.split(",")
        # print("Len: ", len(line))
        roty.append(float(line[0]))
        rotz.append(float(line[1]))
        uaccelz.append(float(line[2]))
        xaccl.append(float(line[3]))

        if (len(roty) > 75):
            # Get first 50
            cur_roty = roty[:50]
            cur_rotz = rotz[:50]
            cur_uaccelz = uaccelz[:50]
            cur_xaccl = xaccl[:50]

            # deleted first 10
            del roty[:10]            
            del rotz[:10]    
            del uaccelz[:10]    
            del xaccl[:10] 

            # make into arrays
            n_roty = np.array(cur_roty)
            n_rotz = np.array(cur_rotz)
            n_uaccelz = np.array(cur_uaccelz)
            n_xaccl = np.array(cur_xaccl)
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

            if label == "right down":
                rd = True
            elif label == "right up":
                print "RIGHT TILT"
                rd = False
                flush(rotz,roty,uaccelz,xaccl)

            if label == "left down":
                ld = True
            elif label == "left up":
                print "LEFT TILT"
                ld = False
                flush(rotz,roty,uaccelz,xaccl)

            else:
                print "NOISY"
    
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
