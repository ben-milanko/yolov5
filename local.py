# from remote import MyService
# import rpyc

# hosts = {"14.137.209.102":2940, "14.137.209.102":2156}

# if __name__ == "__main__":
#     from rpyc.utils.server import ThreadedServer

#     for host in hosts:
#         t = ThreadedServer(MyService, hostname=host, port=hosts[host])
#         t.start()

#     # asleep = rpyc.async_(c.modules.time.sleep)

import requests

payload = {'key1': 'value1', 'key2': ['value2', 'value3']}
r = requests.get('192.168.1.2', params=payload)