import Pyro4

@Pyro4.expose
class MyPyroThing(object):
    # ... methods that can be called go here...
    val = 'CAN YOU SEE ME?'

    def greet(self):
        print('Hello world!')

    def get(self):
        return self.val

    def set(self, string):
        self.val = string

daemon = Pyro4.Daemon()

# Ensure that a Pyro4 nameserver is running by calling: pyro4-ns
# pyro4-ns uses the default port 9090 on localhost
ns = Pyro4.locateNS()


test = MyPyroThing()

uri = daemon.register(test)

# Use PyroServer.testing as name server
ns.register('PyroServer.testing', uri)

print('Ready')

test.set('changed!')

daemon.requestLoop()
