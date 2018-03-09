import socket,os,sys,thread,argparse,threading
#import parse
import time
#from pynput.mouse import Button, Controller
#from pynput.keyboard import Key, Controller
from pynput import mouse, keyboard
import classify
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import Metrics as met
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
    while 1:
    # request from the client
        data = conn.recv(999999)
        with open("attitude.txt", "w") as text_file:
            text_file.write(data)
        (dataX,dataY,dataZ,dataW) = classify2.attitude_txt("attitude.txt","")
        data = conn.recv(999999)
        with open("xaccel.txt", "w") as text2:
            text2.write(data)
        (xaccel) = xyz_accl("xaccel.txt")
        data = conn.recv(999999)
        with open("uaccel.txt", "w") as text2:
            text2.write(data)
        (uaccel) = xyz_accl("uaccel.txt")
        tmp = []
        tmp.append(xaccl.getDev())
        tmp.append(uaccel.getDev())
        tmp.append(xaccl.getMax())
        tmp.append(dataZ.getMed())
        tmp.append(dataY.getMean())
        tmp.append(dataZ.getMean())
        tmp.append(dataZ.getMin())
        Features = []
        Features.append(tmp)
        test = knn.predict(tmp)
        
        if test: == 'right':
  #          mouseMove(x,y)
            keyPressR()
        else if test = 'left':
            keyPressL()
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
