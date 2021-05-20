import argparse
import atexit
import collections
from datetime import datetime
import itertools
import socket
import math
import sys
import time
from typing import Iterable
import cv2
import pprint as pp
import numpy as np
from dataclasses import dataclass
import socketserver
import threading
from scipy.optimize import curve_fit
from _thread import *


def func(t, a, b, c, d):
    return a*t**3 + b*t**2 + c*t+d

def func2(t, a, b, c):
    return a*t**2 + b*t + c

DISTANCE_LIMIT = 200
TRACK_LENGTH = 200
FILT_LENGTH = 8

SERVER_HOST = '192.168.1.8'  # Standard loopback interface address (whistler)
# Port to listen on (non-privileged ports are > 1023)
SERVER_PORT = 65432

UI_HOST = '192.168.1.8'
UI_PORT = 7777

assert FILT_LENGTH <= TRACK_LENGTH

frame = 0
scale = 6
xOff = 1000
yOff = 2000

classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light',
           10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter'}

"""
Alta: 192.168.1.2 https://www.youtube.com/watch?v=1EiC9bvVGnk
Chamonix: 192.168.1.19 https://www.youtube.com/watch?v=DoUOrTJbIu4
Hakuba: 192.168.1.4 https://www.youtube.com/watch?v=6aJXND_Lfk8
"""
H = {
    '192.168.1.2': np.asarray([[-8.41029555e-02, -7.15153371e-01,  2.40736158e+02],
                                [-2.54633255e-01, -2.38135796e+00,
                                1.46226117e+03],
                                [-7.54610883e-05, -2.14064010e-03,  1.00000000e+00]], np.float32),
    '192.168.1.19': np.asarray([[-0.06279556328526636, -0.8439838464409983, 636.6910024019749],
                                [0.06002566118702526, -
                                    1.1014093105498994, 735.4478378609678],
                                [-2.306609260541101e-05, -0.0014515780045251873, 1.0]], np.float32),
    '192.168.1.4': np.asarray([[-0.07137254811611343, -0.7789136260243587, 309.4492984150035],
                                [-0.32071794990736197, -
                                0.8947839825237189, 751.4388837007813],
                                [-0.00010434468403611912, -0.001968194635173746, 1.0]], np.float32),
}


@dataclass
class Position:
    x: float
    y: float
    t: datetime


class PersistentObject():
    def __init__(self, id: int, detection_type: str, initial_pos: Position, last_assigned_frame: int, timestamp: datetime):
        self.id: int = id
        self.detection_type: str = detection_type
        self.track = collections.deque(maxlen=TRACK_LENGTH)
        self.last_assigned_frame = last_assigned_frame
        self.track.append(initial_pos)

        self.velocity: float = 0
        self.rotation: float = 0
        self.no_detections: int = 0

        self.filtered_track = collections.deque(maxlen=TRACK_LENGTH)
        self.filtered: Position = initial_pos
        self.filtered_track.append(self.filtered)
        self.filtered_previous: Position = None
        self.timestamp: datetime = timestamp
        self.poptY = [0, 0, 0]
        self.poptX = [0, 0, 0]

        # self.poptY = [0, 0, 0, 0]
        # self.poptX = [0, 0, 0, 0]

    def add_position(self, pos: Position, frame: int,  time: datetime, repeat: bool = False):
        self.last_assigned_frame = frame
        self.timestamp = time

        if len(self.track) >= 1:
            self.set_filtered()
            # delta_t = (pos.t-self.track[-1].t).total_seconds()
            last: Position = self.filtered_track[0]
            delta_t = (self.filtered.t - last.t).total_seconds()
            if delta_t > 0:
                # delta_x = self.filtered_previous.x - self.filtered.x
                # delta_y = self.filtered_previous.y - self.filtered.y

                delta_x = last.x - self.filtered.x
                delta_y = last.y - self.filtered.y

                self.velocity = np.linalg.norm((delta_x, delta_y)) / delta_t
                if delta_x == 0:
                    self.rotation = 0
                else:
                    self.rotation = np.arctan(delta_y / delta_x)

        self.track.append(pos)
        if not repeat:
            self.no_detections = 0

    def distance_to(self, pos: Position) -> float:
        return np.linalg.norm((self.filtered.x - pos.x, self.filtered.y-pos.y))

    def get_position(self) -> Position:
        return self.filtered

    def set_filtered(self) -> None:

        _track = self.track if len(self.track) < FILT_LENGTH else list(
            itertools.islice(self.track, len(self.track)-FILT_LENGTH, len(self.track)))

        mean_x = np.mean([[pos.x for pos in _track]])
        mean_y = np.mean([[pos.y for pos in _track]])

        self.filtered_previous = self.filtered
        self.filtered = Position(mean_x, mean_y, self.track[-1].t)
        self.filtered_track.append(self.filtered)
        self.solve()

    def solve(self):
        if len(self.filtered_track) > 5:
            x, y, t = [], [], []
            latest = self.filtered_track[-1].t.timestamp()
            for pos in self.filtered_track:
                x.append(pos.x)
                y.append(pos.y)
                t.append(pos.t.timestamp() - latest)
            x = np.asarray(x)
            y = np.asarray(y)
            t = np.asarray(t)
            try:
                # self.poptX, _ = curve_fit(func, t, x)
                # self.poptY, _ = curve_fit(func, t, y)
                self.poptX, _ = curve_fit(func2, t, x)
                self.poptY, _ = curve_fit(func2, t, y)
                # self.poptX[0] /= 2
                # self.poptY[0] /= 2
            except:
                # self.poptX = [0, 0, 0, 0]
                # self.poptY = [0, 0, 0, 0]
            # print(self.poptX, self.poptY)
                self.poptY = [0, 0, 0]
                self.poptX = [0, 0, 0]
            # plt.plot(t, func(t, *self.poptX), label="Fitted Curve")

        # if len(self.filtered_track) > 5:
        #     x,y,t = [],[],[]
        #     for pos in self.filtered_track:
        #         x.append(pos.x)
        #         y.append(pos.y)
        #         t.append(pos.t.timestamp())
        #     x = np.asarray(x)
        #     y = np.asarray(y)
        #     t = np.asarray(t)

        #     A = np.column_stack([np.ones(len(x)), x, x**2, y, y**2])
        #     result, _, _, _ = np.linalg.lstsq(A, t)
        #     # a, c, e = result
        #     print(result)
    def future(self, time: float) -> tuple:
        x = func2(time, *self.poptX)
        y = func2(time, *self.poptY)
        return (x,y)

class Persistence():
    def __init__(self):
        self.next_object_id = 0
        self.objects: Iterable[PersistentObject] = []
        self.frame = 0

    def add_detection(self, x: float, y: float, detection_type: str, frame: int, time: datetime) -> None:
        self.frame = frame
        min = math.inf
        matched_obj: PersistentObject = None
        pos = Position(x, y, time)
        for obj in self.objects:
            delta_t = (time-pos.t).total_seconds()
            next_pos = obj.future(delta_t)
            dist = obj.distance_to(Position(next_pos[0], next_pos[1], time))
            dist = obj.distance_to(pos)
            if dist < min and dist < DISTANCE_LIMIT:
                min = dist
                matched_obj = obj

        if matched_obj is None:
            new_obj = PersistentObject(
                self.next_object_id, detection_type, pos, frame, time)
            self.objects.append(new_obj)
            self.next_object_id += 1
        else:
            matched_obj.add_position(pos, frame, time)
            if obj.detection_type is not detection_type:
                matched_obj.detection_type = detection_type

    def remove_oldest_samples(self, frame: int) -> None:
        self.frame = frame
        for obj in list(self.objects):
            if obj.last_assigned_frame < frame:
                obj.track.pop(0)
                if len(obj.track) == 0:
                    self.objects.remove(obj)

    def fill_in_samples(self, frame: int, time: datetime) -> None:
        self.frame = frame
        for obj in list(self.objects):
            if obj.last_assigned_frame < frame:
                pos = obj.get_position()
                pos.t = datetime.now()
                obj.add_position(pos, frame, time, repeat=True)
                obj.no_detections += 1
                if (obj.no_detections >= 20):
                    self.objects.remove(obj)

    def send(self) -> bytes:
        data = []
        for obj in self.objects:
            pos = obj.filtered
            cls = next(key for key, value in classes.items()
                       if value == obj.detection_type)
            data.extend([obj.id, pos.x, pos.y, obj.rotation,
                        obj.velocity, cls, obj.timestamp.timestamp()])
            # data.extend([obj.id, pos.x, pos.y, obj.rotation, obj.velocity, cls, obj.timestamp.timestamp(), *obj.poptX, *obj.poptY])
        print(f"{len(data)/7} Objects detected")
        print(data)
        return np.asarray(data).tobytes()


persistence = Persistence()


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    # def __init__(self, callback, *args, **keys):
    #     self.frame: int = 0
    #     # SocketServer.BaseRequestHandler.__init__(self, *args, **keys)

    def handle(self):
        print("Client connected to server")
        while True:
            # if persistence.frame != self.frame:
            # self.frame = persistence.frame
            data = persistence.send()
            self.request.sendall(data)
            time.sleep(0.1)
            # else:
            #     time.sleep(0.01)

def threaded_client(connection, client_address):
    connection.send(str.encode('Welcome to the Servern'))
    while True:
        recieved_data = connection.recv(2048)
        data = np.frombuffer(recieved_data)
        cls = []
        pts = []

        # For some reason socket will send packets back of incorrect length,
        mod = len(data) % 5
        if mod:
            new_len = len(data) - mod
            if not new_len:
                continue
            data = data[0:new_len]

        for i in range(0, len(data), 5):
            u = data[i+1]*1920
            v = (data[i+2]-data[i+4])*1080
            pts.append([u, v])
            cls.append(classes[int(data[i])])

        pts1 = np.asarray(pts, np.float32)
        pts2 = pts1.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts2, H[client_address[0]])

        time = datetime.now()

        for i in range(len(cls)):
            x = transformed[i][0][0]
            y = transformed[i][0][1]
            persistence.add_detection(x, y, cls[i], frame, time)

        persistence.fill_in_samples(frame, time)


def main(args):

    ui_server = socketserver.ThreadingTCPServer(
        (UI_HOST, UI_PORT), ThreadedTCPRequestHandler)
    server_thread = threading.Thread(target=ui_server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)

    server = socket.socket()
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(4)
    print('Listening on 192.168.1.8...')



    # server.setblocking(0)
    clients = {}
    client_count = 2
    # client_socket, client_address = server.accept()
    # print(client_address, "has connected")

    # client_socket.setblocking(False)
    for i in range(client_count):
        client_socket, client_address = server.accept()
        print(client_address, "has connected")
        clients[client_address] = client_socket
        print(f'Waiting on {client_count - (i+1)} more connection(s)')
        start_new_thread(threaded_client, (client_socket, client_address))

    def closeconnection():
        ui_server.shutdown()
        server.close()
        print("Connection closed")

    atexit.register(closeconnection)

    if not args.no_pygame:
        import pygame
        window = pygame.display.set_mode((1000, 1000))

        colours = {'person': (255, 0, 255), 'bicycle': (255, 255, 0), 'car': (255, 0, 0), 'motorcycle': (0, 0, 0), 'bus': (255, 0, 255), 'truck': (0, 0, 255), 'traffic light': (0, 255, 0),
                   'fire hydrant': (255, 0, 255), 'stop sign': (255, 0, 255), 'parking meter': (255, 0, 255)}
        window.fill((255, 255, 255))
        pygame.display.flip()

    while True:
        
        # client, address = server.accept()
        # print('Connected to: ' + address[0] + ':' + str(address[1]))
        # start_new_thread(threaded_client, (client, address))

        # pygame.draw.circle(
        #     window, (255,255,255), (0, 0), 20)
        # recieved_data = []
        # # for key in clients:
        # #     print(key)
        # # clients[client_address].recv(2048)
        # recieved_data = client_socket.recv(2048)

        # data = np.frombuffer(recieved_data)
        # cls = []
        # pts = []

        # # Pcakets 
        # mod = len(data) % 5
        # if mod:
        #     new_len = len(data) - mod
        #     if not new_len:
        #         continue
        #     data = data[0:new_len]

        # for i in range(0, len(data), 5):
        #     u = data[i+1]*1920
        #     v = (data[i+2]-data[i+4])*1080
        #     pts.append([u, v])
        #     cls.append(classes[int(data[i])])

        # pts1 = np.asarray(pts, np.float32)
        # pts2 = pts1.reshape(-1, 1, 2).astype(np.float32)
        # transformed = cv2.perspectiveTransform(pts2, H[client_address[0]])

        # time = datetime.now()

        # for i in range(len(cls)):
        #     x = transformed[i][0][0]
        #     y = transformed[i][0][1]
        #     persistence.add_detection(x, y, cls[i], frame, time)

        # persistence.fill_in_samples(frame, time)

        if not args.no_pygame:
            pygame.time.Clock().tick(60)
            window.fill((255, 255, 255))
            for obj in persistence.objects:
                for i in range(len(obj.filtered_track),0, -1):

                    pos = obj.filtered_track[i]
                    locationX = int((pos.x+xOff)/scale)
                    locationY = int((pos.y+yOff)/scale)

                    pygame.draw.circle(
                        window, colours[obj.detection_type], (locationX, locationY), 5 if i == len(obj.filtered_track)-1 else 3)

                    pos = obj.track[i]
                    locationX = int((pos.x+xOff)/scale)
                    locationY = int((pos.y+yOff)/scale)

                    pygame.draw.circle(
                        window, (0,0,0), (locationX, locationY), 3)

                for i in range(10):

                    # x = func(i/20, *obj.poptX)
                    # y = func(i/20, *obj.poptY)
                    x = func2(i/20, *obj.poptX)
                    y = func2(i/20, *obj.poptY)
                    pygame.draw.circle(
                        window, (0, 255, 255), ((x+xOff)/scale, (y+yOff)/scale), 3)

            pygame.display.flip()

        # frame += 1
        # if frame == sys.maxsize-1:
        #     frame = 0
        # persistence.send()
        # ui_server.handle_request()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-pygame', action='store_true')
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Bye")
        sys.exit()
