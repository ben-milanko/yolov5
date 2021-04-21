from remote import MyService
import rpyc

hosts = {"14.137.209.102":2940, "14.137.209.102":2156}

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer

    for host in hosts:
        t = ThreadedServer(MyService, hostname=host, port=hosts[host])
        t.start()

    asleep = rpyc.async_(c.modules.time.sleep)