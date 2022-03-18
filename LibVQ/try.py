

def b(x=2, y=3):
    print(x, y)

def a(c=3, *args, **kwargs):

    b(**kwargs)

a(c=3, x=1, y=2, dd=4)