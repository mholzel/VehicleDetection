import cv2, itertools, math, numpy, os, pickle, time, utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

'''
The purpose of this file is to whittle down the amount of cells we have to scan 
when looking for cars. Specifically, starting from a grid of all of the cells 
that one would typically scan, you can remove cells that a car would not realistically 
appear in at the specified scale
'''


class Rect():
    '''
    A class that holds a matplotlib rectangle and some other dimensions (for convenience)
    '''

    def __init__(self, xpos, ypos, xbox_left, ytop_draw, ystart, win_draw, color):
        self.x = xbox_left
        self.y = ytop_draw + ystart
        self.w = win_draw
        self.xpos = xpos
        self.ypos = ypos
        self.rect = Rectangle((self.x, self.y), self.w, self.w, fill=False, edgecolor=color)


def create_rectangles(ax, img, ystart, ystop, scale, pix_per_cell, cell_per_block, symmetric=True):
    '''
    Create a list of rectangles that WOULD be searched if we were going to
    use this function naively.
    '''

    # Create the colors that we will iterate though when drawing rectangles
    colors = itertools.cycle(
        ["#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ffff00", "#ffffff"])

    # Define blocks and steps as above
    nxblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    count = 0
    rects = []
    win_draw = np.int(window * scale)
    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Scale
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)

            if ((ytop_draw + ystart + win_draw) > ystop) or ((xbox_left + win_draw) > img.shape[1]):
                continue
            elif not symmetric and (xbox_left + win_draw / 2 <= img.shape[1] / 2):
                # We are only going to keep rectangles in the right hand side of the image
                # Since we assume that there is symmetry, you can duplicate these cells on the
                # left hand side.
                continue

            count += 1
            rect = Rect(xpos, ypos, xbox_left, ytop_draw, ystart, win_draw, next(colors))
            ax.add_patch(rect.rect)
            rects.append(rect)

    # Add a rectangle in the upper left to give us an idea of scale
    # Don't forget to remove this point.
    rect = Rect(0, 0, img.shape[1] // 2, img.shape[0] // 3, 0, win_draw, next(colors))
    ax.add_patch(rect.rect)
    rects.append(rect)
    return rects


class Rects():
    '''
    A container class for a list of rectangles.
    When the mouse point in the image is clicked,
    all of the rectangles containing that
    point are removed from the list.

    When you double click on the top of the image, in the whitespace,
    the
    '''

    def __init__(self, rects, fig, filename):
        self.rects = rects
        self.fig = fig
        self.filename = filename
        self.exported = False

    def update(self, event):
        tmp = []
        for rect in self.rects:
            contains, b = rect.rect.contains(event)
            if contains:
                try:
                    rect.rect.remove()
                except:
                    pass
            else:
                tmp.append(rect)
        print('Done updating')
        self.rects = tmp
        self.fig.canvas.draw()

    def export(self, save):
        # Convert the rectangle to a list containing the upper left points and size (width and height are equal)
        rects = []
        for rect in self.rects:
            rects.append([rect.x, rect.y, rect.w, rect.xpos, rect.ypos])
        if save:
            pickle.dump(rects, open(self.filename, "wb"))
        else:
            return rects
        self.exported = True

    def isExported(self):
        return self.exported


def generate_rectangles(img, scales, pix_per_cell, cell_per_block, ystop=None, save=False):
    # Load the image that we will use to choose our rectangles
    img = utils.load(img)

    # Now, let's iterate through the scales that we want to include
    height = img.shape[0]
    for scale in scales:

        # Show the image (we will add rectangles later)
        fig, ax = plt.subplots(1, 1, figsize=(6,8))
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0, left=0, top=.95, right=1)
        ax.imshow(img)
        ax.axis('off')

        # Choose an appropriate starting and stopping height for this scale
        ystart = height // 2
        if ystop is None:
            if scale <= 1:
                ystop = math.floor(height * .66)
            elif scale >= 3:
                ystop = height
            else:
                ystop = math.floor(height * (.66 + (scale - 1) / 2 * (1 - .66)))

        # Create and draw all of the rectangles
        rects = create_rectangles(ax, img, ystart, ystop, scale, cell_per_block=cell_per_block,
                                  pix_per_cell=pix_per_cell, symmetric=utils.ipython())

        # Create the rectangle holder object, which will allow us to click to remove
        # unrealistic rectangles
        mainDir = "search_rectangles/"
        subDir = str(pix_per_cell) + "_" + str(cell_per_block) + "/"
        name = "rectangles" + str(int(scale * 100)) + ".p"
        filename = mainDir + subDir + name
        R = Rects(rects, fig, filename=filename)

        # Define the click function that will export the rectangles when
        # double clicking in the whitespace on the top of the image,
        # and update the rectangles otherwise.
        def onclick(event, rects, save):
            if event.dblclick and event.ydata is None:
                rects.export(save)
            else:
                rects.update(event)

        if not utils.ipython():
            fig.canvas.mpl_connect('button_press_event', lambda x: onclick(x, R, save))
            plt.show(block=False)
            while not R.isExported():
                plt.pause(.1)
            print("exported")


def show_rectangles(dir, img):
    # Create the colors that we will iterate though when drawing rectangles
    colors = itertools.cycle(
        ["#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ffff00", "#ffffff"])

    files = os.listdir(dir)
    cols = 3
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols)
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0, left=0, top=1, right=1)
    axes = numpy.array(axes).ravel()
    for ax in axes:
        ax.axis('off')
        ax.imshow(img)
    for ax, file in zip(axes, files):
        rects = pickle.load(open(os.path.join(dir, file), "rb"))
        mirror(rects, img.shape[1])
        for rect in rects:
            r = Rectangle((rect[0], rect[1]), rect[2], rect[2], fill=False,
                          edgecolor=next(colors))
            ax.add_patch(r)
    plt.show()


def mirror(rects, width):
    aug = []
    for rect in rects:
        aug.append(list(rect))
        aug[-1][0] = width - aug[-1][0] - aug[-1][2]
    rects.extend(aug)


if __name__ == "__main__":
    # Load the image that we will use to choose our rectangles
    img = 'project_video/frame741.jpg'
    img = utils.load(img)
    scales = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5]

    if True:
        generate_rectangles(img, scales, 8, 2, ystop=img.shape[0], save=False)
    else:
        show_rectangles('search_rectangles/16_2/', img)
