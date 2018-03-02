import socket,os,sys,thread,argparse,threading
#import parse
import time
#from pynput.mouse import Button, Controller
#from pynput.keyboard import Key, Controller
from pynput import mouse, keyboard

'''
Presses right arrow key
'''
def keyPress():
    keyB = keyboard.Controller()
    keyB.press(keyboard.Key.right)
    keyB.release(keyboard.Key.right)

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

    while 1:
    # request from the client
        data = conn.recv(999999)   
    # ADD: HOW to Parse request?
    # ADD: test = Function call to test data?
        if test:
            mouseMove(x,y)
            keyPress()
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
