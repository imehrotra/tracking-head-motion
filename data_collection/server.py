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

    knn, scaler = classify2.classify()
    #count = 0
    #test = 0
    #test2 = 0
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
        # @Isha, can you check the size of data? 
        data = conn.recv(999999)
        if data == '\n':
            continue
        line = data.split(",")
        roty.append(float(line[0]))
        rotz.append(float(line[1]))
        uaccelz.append(float(line[2]))
        xaccl.append(float(line[3]))
        print("Data Time: " + line[4])
        print("Curr Time: ", time.time())  
        if (len(roty) > 50):
            # Get first 50
            cur_roty = roty[:50]
            cur_rotz = rotz[:50]
            cur_uaccelz = uaccelz[:50]
            cur_xaccl = xaccl[:50]

            # deleted first 25
            del cur_roty[:25]            
            del cur_rotz[:25]    
            del cur_uaccelz[:25]    
            del cur_xaccl[:25] 

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



        # with open("recv.txt","w") as file:

    # request from the client
                # file.write(data)
            
        #open file
        # count = 0
        # with open("recv.txt","r") as file:
            # roty = []
            # rotz = []
            # uaccelz = []
            # xaccl = []
            #for row in file:
            #    if row == '\n':
            #        continue
            #    line = row.split(",")
            #    roty.append(float(line[0]))
            #    rotz.append(float(line[1]))
            #    uaccelz.append(float(line[2]))
            #    xaccl.append(float(line[3]))
            # if data == '\n':
            #     continue
            # line = data.split(",")
            # roty.append(float(line[0]))
            # rotz.append(float(line[1]))
            # uaccelz.append(float(line[2]))
            # xaccl.append(float(line[3]))
            # print("Data Time: " + line[4])
            # print("Curr Time: ", time.time())
        # roty = np.array(roty)
        # rotz = np.array(rotz)
        # uaccelz = np.array(uaccelz)
        # xaccl = np.array(xaccl)
        # tmp = []
        # tmp.append(xaccl.std())
        # tmp.append(uaccelz.std())
        # tmp.append(xaccl.max())
        # tmp.append(np.median(roty))
        # tmp.append(roty.mean())
        # tmp.append(rotz.mean())
        # tmp.append(rotz.min())
        # Features = []
        # Features.append(tmp)
        # print "features", Features
        # if test == 0:
        #     scaler.transform(Features)
        #     test = knn.predict(Features)
        # else:
        #     test2 = knn.predict(Features)
        # if test == 'right down' and test2 == 'right up':
        #     # keyPressR()
        #     print "right"
        #     test = 0
        # elif test == 'left down' and test2 == 'left up':
        #     # keyPressL()
        #     print "left"
        #     test = 0
        # elif test2 == 'right down':
        #     print "start right down"
        #     test = test2
        # elif test2 == 'left down':
        #     print "start left down"
        #     test = test2
        # else:
        #     print("test:", test)
        #     print("test2:", test2)
        #     print "noisy"
        #     continue


        #     else:
        #         test2 = knn.predict(Features)              
        #     if test == 'right down' and test2 == 'right up':
        #         # keyPressR()
        #         print "right"
        #         test = 0
        #     elif test == 'left down' and test2 == 'left up':
        #         # keyPressL()
        #         print "left"
        #         test = 0
        #     elif test2 == 'right down':
        #         print "start right down"
        #         test = test2
        #     elif test2 == 'left down':
        #         print "start left down"
        #         test = test2
        #     else:
        #         print("test:", test)
        #         print("test2:", test2)
        #         print "noisy"
        #         continue
    
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
