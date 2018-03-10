import socket,os,sys,thread,argparse,threading
#import parse
import time
#from pynput.mouse import Button, Controller
#from pynput.keyboard import Key, Controller
from pynput import mouse, keyboard
import classify2
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
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


def threadAPI(conn, clientaddr, x, y):
    knn = classify2.classify()
    count = 0
    test = 0
    test2 = 0
    while 1:
        with open("recv.txt","w") as file:
            while count <= 50:
    # request from the client
                data = conn.recv(999999)
                file.write(data)
                count+=1
            
        #open file
        count = 0
        with open("recv.txt","r") as file:
            roty = []
            rotz = []
            uaccelz = []
            xaccl = []
            for row in file:
                line = row.split(",")
                roty.append(line[0])
                rotz.append(line[1])
                uaccelz.append(line[2])
                xaccl.append(line[3])
            roty = np.array(roty)
            rotz = np.array(roz)
            uaccelz = np.array(uaccelz)
            xaccl = np.array(xaccl)
        tmp = []
        tmp.append(xaccl.getDev())
        tmp.append(uaccelz.getDev())
        tmp.append(xaccl.getMax())
        tmp.append(roty.getMed())
        tmp.append(roty.getMean())
        tmp.append(rotz.getMean())
        tmp.append(rotz.getMin())
        Features = []
        Features.append(tmp)
        if test == 0:
            test = knn.predict(Features)
        else:
            test2 = knn.predict(Features)              
        if test == 'rd' and test2 == 'ru':
            keyPressR()
            print "right"
            test = 0
        elif test == 'ld' and test2 == 'lu':
            keyPressL()
            print "left"
            test == 0
        else:
            continue
    
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

    if args.x:
        x = args.x
    else:
        x = 1000

    if args.y:
        y = args.y
    else:
        y = 800
        
    #hostname and port name
    port = int(args.port)
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
