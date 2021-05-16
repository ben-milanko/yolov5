import argparse
import atexit
import collections
from datetime import datetime
import itertools
import socket
import math
from typing import Iterable
import cv2
import pprint as pp
import numpy as np
from pykalman import KalmanFilter
from dataclasses import dataclass
from statsmodels.tsa.arima.model import ARIMA

DISTANCE_LIMIT = 200
TRACK_LENGTH = 20
FILT_LENGTH = 4

assert FILT_LENGTH <= TRACK_LENGTH

@dataclass
class Position:
    x: float
    y: float
    t: datetime

class PersistentObject():
    def __init__(self, id: int, detection_type: str, initial_pos: Position, last_assigned_frame: int):
        self.id: int = id
        self.detection_type: str = detection_type
        self.track = collections.deque(maxlen=TRACK_LENGTH)
        self.last_assigned_frame = last_assigned_frame
        self.track.append(initial_pos)
        
        self.velocity = 0
        self.rotation = 0
        self.no_detections = 0

        self.filtered: Position = initial_pos
        self.filtered_previous: Position = None

    def add_position(self, pos: Position, frame: int, repeat: bool=False):
        self.last_assigned_frame = frame
        
        if len(self.track) >= 1:
            self.set_filtered()
            delta_t = (pos.t-self.track[-1].t).total_seconds()
            if delta_t > 0:
                delta_x = self.filtered_previous.x - self.filtered.x
                delta_y = self.filtered_previous.y - self.filtered.y
                
                self.velocity = np.linalg.norm((delta_x, delta_y)) / delta_t
                self.rotation = np.arctan(delta_y / delta_x)

        self.track.append(pos)
        if not repeat:
            self.no_detections = 0

    def distance_to(self, pos: Position) -> float:
        return np.linalg.norm((self.filtered.x - pos.x, self.filtered.y-pos.y))

    def get_position(self) -> Position:
        return self.filtered

    def set_filtered(self) -> None:
        
        _track =self.track if len(self.track) < FILT_LENGTH else list(itertools.islice(self.track, len(self.track)-FILT_LENGTH, len(self.track)))
        
        mean_x = np.mean([[pos.x for pos in _track]])
        mean_y = np.mean([[pos.y for pos in _track]])

        self.filtered_previous = self.filtered
        self.filtered = Position(mean_x, mean_y, self.track[-1].t)

class Persistence():
    def __init__(self):
        self.next_object_id = 0
        self.objects: Iterable[PersistentObject] = []

    def add_detection(self, x: float, y: float, detection_type: str, frame: int, time: datetime) -> None:
        min = math.inf
        matched_obj: PersistentObject = None
        pos = Position(x, y, time)
        for obj in self.objects:
            if obj.detection_type is detection_type:
                dist = obj.distance_to(pos)
                if dist < min and dist < DISTANCE_LIMIT:
                    min = dist
                    matched_obj = obj

        if matched_obj is None:
            new_obj = PersistentObject(
                self.next_object_id, detection_type, pos, frame)
            self.objects.append(new_obj)
            self.next_object_id += 1
        else:
            matched_obj.add_position(pos, frame)

    def remove_oldest_samples(self, frame: int) -> None:
        for obj in list(self.objects):
            if obj.last_assigned_frame < frame:
                obj.track.pop(0)
                if len(obj.track) == 0:
                    self.objects.remove(obj)

    def fill_in_samples(self, frame: int) -> None:
        for obj in list(self.objects):
            if obj.last_assigned_frame < frame:
                pos = obj.get_position()
                pos.t = datetime.now()
                obj.add_position(pos, frame, repeat=True)
                obj.no_detections += 1
                if (obj.no_detections >= 20):
                    self.objects.remove(obj)

    def send(self) -> None:
        data = {}
        for obj in self.objects:
            pos = obj.filtered
            data[obj.id] = (pos.x, pos.y, obj.rotation, obj.velocity, obj.detection_type)
        
        pp.pprint(data)

def main(args):
    HOST = '192.168.1.8'  # Standard loopback interface address (whistler)
    PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

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

    persistence = Persistence()

    if not args.no_pygame:
        import pygame
        window = pygame.display.set_mode((1000, 1000))

        typeDimensions = {"car": (4.7, 1.9), "person": (
            0.25, 0.45), "truck": (5, 2.2), "bus": (10, 2.2)}

        colours = {'person': (255, 0, 255), 'bicycle': (255, 255, 0), 'car': (255, 0, 0), 'motorcycle': (0, 0, 0), 'bus': (255, 0, 255), 'truck': (0, 0, 255), 'traffic light': (0, 255, 0),
                   'fire hydrant': (255, 0, 255), 'stop sign': (255, 0, 255), 'parking meter': (255, 0, 255)}

    server = socket.socket()
    server.bind((HOST, PORT))
    server.listen(4)
    print('Listening on 192.168.1.8...')

    if not args.no_pygame:
        window.fill((255, 255, 255))
        pygame.display.flip()

    clients = {}
    client_count = 2
    client_socket, client_address = server.accept()
    print(client_address, "has connected")

    # for i in range(client_count):
    #     client_socket, client_address = server.accept()
    #     print(client_address, "has connected")
    #     clients[client_address] = client_socket
    #     print(f'Waiting on {client_count - (i+1)} more connection(s)')

    def closeconnection():
        server.close()
        print("Connection closed")

    atexit.register(closeconnection)

    frame = 0

    while True:
        recieved_data = []
        # for key in clients:
        #     print(key)
        # clients[client_address].recv(2048)
        client_address.recv(2048)

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

        persistence.fill_in_samples(frame)

        if not args.no_pygame:
            window.fill((255, 255, 255))
            for i in range(len(cls)):

                x = transformed[i][0][0]
                y = transformed[i][0][1]

                locationX = x/4
                locationY = y/4

                pygame.draw.circle(
                    window, colours[cls[i]], (locationX, locationY), 5)

            pygame.display.flip()

        frame += 1
        persistence.send()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pygame', action='store_true')
    args = parser.parse_args()

    main(args)
