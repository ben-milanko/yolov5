import atexit
import socket
import numpy
import math
import cv2
import pygame

HOST = '192.168.1.8'  # Standard loopback interface address (alta)
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light',
           10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter'}
"""
Alta: 192.168.1.2
Chamonix: 192.168.1.19
Hakuba: 192.168.1.4
"""
H = {
    '192.168.1.2': numpy.asarray([[-8.41029555e-02, -7.15153371e-01,  2.40736158e+02],
                                  [-2.54633255e-01, -2.38135796e+00,  1.46226117e+03],
                                  [-7.54610883e-05, -2.14064010e-03,  1.00000000e+00]], numpy.float32),
    '192.168.1.19': numpy.asarray([[-0.06279556328526636, -0.8439838464409983, 636.6910024019749],
                                   [0.06002566118702526, -1.1014093105498994, 735.4478378609678],
                                   [-2.306609260541101e-05, -0.0014515780045251873, 1.0]], numpy.float32),
    '192.168.1.4': numpy.asarray([[-0.07137254811611343, -0.7789136260243587, 309.4492984150035],
                                  [-0.32071794990736197,-0.8947839825237189,751.4388837007813],
                                  [-0.00010434468403611912,-0.001968194635173746,1.0]], numpy.float32),
}

window = pygame.display.set_mode((1000, 1000))

typeDimensions = {"car": (4.7, 1.9), "person": (
    0.25, 0.45), "truck": (5, 2.2), "bus": (10, 2.2)}

colours = {'person': (255, 0, 255), 'bicycle': (255, 255, 0), 'car': (255, 0, 0), 'motorcycle': (0, 0, 0), 'bus': (255, 0, 255), 'truck': (0, 0, 255), 'traffic light': (0, 255, 0),
           'fire hydrant': (255, 0, 255), 'stop sign': (255, 0, 255), 'parking meter': (255, 0, 255)}

server = socket.socket()
server.bind((HOST, PORT))
server.listen(4)
print('Listening...')
window.fill((255, 255, 255))
pygame.display.flip()


client_socket, client_address = server.accept()
print(client_address, "has connected")


def closeconnection():
    server.close()
    print("Connection closed")


atexit.register(closeconnection)

while True:
    recieved_data = client_socket.recv(1024)
    data = numpy.frombuffer(recieved_data)
    # print(data)
    cls = []
    pts = []
    for i in range(0, len(data), 5):
        u = data[i+1]*1920
        v = (data[i+2]-data[i+4])*1080
        pts.append([u, v])
        cls.append(classes[data[i]])
    # print(cls)
    pts1 = numpy.asarray(pts, numpy.float32)
    pts2 = pts1.reshape(-1, 1, 2).astype(numpy.float32)
    transformed = cv2.perspectiveTransform(pts2, H[client_address[0]])
    # print(transformed)
    window.fill((255, 255, 255))
    for i in range(len(cls)):

        # print(transformed[0])
        # print(transformed[0][0][i])
        x = transformed[i][0][0]
        y = transformed[i][0][1]
        # print(
        #     f"Class: {cls[i]}, x: {int(x)}, y: {int(y)}")

        # width = typeDimensions[cls[i]][0]/4
        # height = typeDimensions[cls[i]][1]/4

        locationX = x/4
        locationY = y/4
        # locationX = (x - typeDimensions[cls[i]][0]/2)/4
        # locationY = (y - typeDimensions[cls[i]][1]/2)/4
        length = 0  # item[2]
        angle = math.radians(0)  # item[3])

        pygame.draw.circle(window, colours[cls[i]], (locationX, locationY), 5)
        # pygame.draw.rect(window, (0, 0, 255), (locationX,
        #                  locationY, locationX+width, locationY+height))

    pygame.display.flip()
