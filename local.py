# # from remote import MyService
# # import rpyc

# # hosts = {"14.137.209.102":2940, "14.137.209.102":2156}

# # if __name__ == "__main__":
# #     from rpyc.utils.server import ThreadedServer

# #     for host in hosts:
# #         t = ThreadedServer(MyService, hostname=host, port=hosts[host])
# #         t.start()

# #     # asleep = rpyc.async_(c.modules.time.sleep)

# import requests

# payload = {'key1': 'value1', 'key2': ['value2', 'value3']}
# r = requests.get('192.168.1.2', params=payload)
import socket

HOST = '192.168.1.8'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv()
            if not data:
                break
            conn.sendall(data)