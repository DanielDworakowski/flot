import Pyro4

@Pyro4.expose
class MyPyroThing(object):
    # ... methods that can be called go here...
    val = 'CAN YOU SEE ME?'

    def greet(self):
        print('Hello world!')

    def get(self):
        return self.val

daemon = Pyro4.Daemon()

# Ensure that a Pyro4 nameserver is running by calling: pyro4-ns
# pyro4-ns uses the default port 9090 on localhost
ns = Pyro4.locateNS()

uri = daemon.register(MyPyroThing)

# Use PyroServer.testing as name server
ns.register('PyroServer.testing', uri)

print('Ready')
daemon.requestLoop()
