import os
trainpath = 'test'


# Get image list
def getImageList():
    imagelist = [os.path.join(parent, filename)
                 if filename.lower().endswith((
                     '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm',
                     '.pgm', '.ppm', '.tif', '.tiff'))
                 else None
                 for parent, dirnames, filenames in os.walk(trainpath)
                 for filename in filenames]
    return [filename for filename in imagelist if filename is not None]


def argwrapper(args):
    return args[0](*args[1:])
