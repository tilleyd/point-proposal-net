# ppn.data
# author: Duncan Tilley

def generate_patches(img, patch_size, overlap=0):
    """
    Creates overlapping patches from the source image.

    Returns a list of images of size patch_size.

    img
        The source image.
    patch_size
        The patch size as (h, w).
    overlap
        The number of pixels to overlap neighbouring patches.
    """
    import numpy as np

    ph, pw = patch_size
    ih, iw = img.shape
    num_cols = int(np.ceil((iw - overlap*2) / (pw - overlap*2)))
    num_rows = int(np.ceil((ih - overlap*2) / (ph - overlap*2)))
    patches = []
    offsets = []
    for r in range(0, num_rows):
        for c in range(0, num_cols):
            py = (ph - overlap) * r
            px = (pw - overlap) * c
            if px+pw > iw:
                px -= ((px+pw) - iw)
            if py+ph > ih:
                py -= ((py+ph) - ih)
            patch = img[py:py+ph, px:px+pw].copy()
            patches.append(patch)
            offsets.append((px, py))
    return patches, offsets

def create_labeled_set(dataset, config):
    """
    Converts a raw data dictionary to a labeled data dictionary that can be
    used by the PPN model.

    dataset
        The raw data set.
    config
        The configuration dictionary. See ppn.config.ppn_config.
    """
    import ppn._data_labeling as lbl

    labeled = {}

    anchors = lbl.get_anchors(config)
    conf, reg = [], []
    for i in range(0, len(dataset['coords'])):
        patch_conf, patch_reg = lbl.get_anchor_labels(anchors,
                                                      dataset['coords'][i],
                                                      config)
        conf.append(patch_conf)
        reg.append(patch_reg)

    labeled['x'] = dataset['maps']
    labeled['y_conf'] = conf
    labeled['y_reg'] = reg

    return labeled

class Data(object):

    def __init__(self,
                 image_directory='data/image/',
                 label_directory='data/label/',
                 patch_images=False):
        """
        Reads the images and labels into memory.

        image_directory
            The directory where the images are located.
        label_directory
            The directory where the labels are located. Each label file must be
            a list of x,y,j floats, where x,y is the pixel coordinate of a
            source and j is the source intensity.
        """

        import os
        import numpy as np

        images = [n for n in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, n))]
        labels = [n for n in os.listdir(label_directory) if os.path.isfile(os.path.join(label_directory, n))]
        images.sort()
        labels.sort()

        assert(len(images) == len(labels))

        print('Reading %d images:' %(len(images)))
        print('  %s .. %s' %(images[0], images[-1]))
        print('  %s .. %s' %(labels[0], labels[-1]))

        self.maps = []
        self.coords = []
        self.brights = []
        source_count = 0
        for i in range(0, len(images)):
            img_file = os.path.join(image_directory, images[i])
            lbl_file = os.path.join(label_directory, labels[i])
            img_ext = img_file.split('.')[-1]
            if img_ext == 'fits':
                # fits file, read using astropy
                from astropy.io import fits
                img = np.squeeze(fits.getdata(img_file))
            elif img_ext == 'npy':
                # saved numpy file, load array
                img = np.squeeze(np.load(img_file))
            else:
                # attempt to read using opencv
                import cv2 as cv
                img = np.squeeze(cv.imread(img_file))

            assert(len(img.shape) == 2)
            imgh, imgw = img.shape

            # linearly scale the image
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img - img_min) / (img_max - img_min)

            # read the labels
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                img_coords = []
                img_brights = []
                for l in lines:
                    l = l.split(',')
                    # read the pixel coordinates and intensity
                    x, y = float(l[0]), float(l[1])
                    if x >= 0 and x < imgw and y >= 0 and y < imgh:
                        source_count += 1
                        img_coords.append( (int(x), int(y)) )
                        img_brights.append( float(l[2]) )

            if not patch_images:
                self.maps.append(np.array([img]))
                self.coords.append(img_coords)
                self.brights.append(img_brights)
            else:
                def get_sources_in_patch(all_coords, all_brights,
                                         x, y, w, h):
                    """
                    Returns the source coordinates within a patch of the image.
                    """
                    sources = []
                    brights = []
                    for i in range(0, len(all_coords)):
                        sx,sy = all_coords[i]
                        if sx >= x and sy >= y and sx < x + w and sy < y + h:
                            sources.append((sx - x, sy - y))
                            brights.append(all_brights[i])
                    return sources, brights

                patches, patch_offsets = generate_patches(img, (224, 224), 4)
                for j in range(0, len(patches)):
                    x, y = patch_offsets[j]
                    patch_coords, patch_brights = get_sources_in_patch(img_coords,
                                                                       img_brights,
                                                                       x, y,
                                                                       224, 224)
                    self.maps.append(np.array([patches[j]]))
                    self.coords.append(patch_coords)
                    self.brights.append(patch_brights)


        print('Done, read %d sources' %(source_count))

    def display_image(self, index, show_sources=False, predictions=None):
        """
        Opens a window displaying the image at the given index.

        index
            The index of the image to display.
        show_sources
            If true, a red dot will be drawn in the location of each labeled
            source.
        predictions
            An array of [x, y] prediction locations to show.
        """
        import numpy as np
        import cv2 as cv

        image = self.maps[index][0]

        # convert to RGB
        image = np.transpose(np.tile([image], (3, 1, 1)), [1, 2, 0]).copy()

        if show_sources:
            for x,y in self.coords[index]:
                image[y][x] = [0.0, 1.0, 1.0]
                cv.circle(image, (int(x), int(y)), 3, (0, 165, 255))

        if predictions is not None:
            for x,y in predictions:
                if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
                    image[int(y)][int(x)] = [0.0, 0.0, 1.0]
                    cv.circle(image, (int(x), int(y)), 3, (0, 0, 255))

        cv.namedWindow('ppn-image', cv.WINDOW_AUTOSIZE)
        cv.imshow('ppn-image', image)
        cv.waitKey()

    def save_image(self, filename, index, show_sources=False, predictions=None, size=None):
        """
        Saves the image at the given index to disk.

        filename
            The name of the saved image.
        index
            The index of the image to save.
        show_sources
            If true, a red dot will be drawn in the location of each labeled
            source.
        predictions
            An array of [x, y] prediction locations to show.
        size
            If not None, a sub-image at the (0, 0) corner will be taken of size (size, size).
        """
        import cv2 as cv
        import numpy as np

        image = self.maps[index][0]

        # convert to RGB
        image = np.transpose(np.tile([image], (3, 1, 1)), [1, 2, 0]).copy()

        if show_sources:
            for x,y in self.coords[index]:
                image[y][x] = [0.0, 1.0, 1.0]
                cv.circle(image, (int(x), int(y)), 4, (0, 255, 255), thickness=2)

        if predictions is not None:
            for x,y in predictions:
                if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
                    image[int(y)][int(x)] = [0.0, 0.0, 1.0]
                    cv.circle(image, (int(x), int(y)), 3, (0, 0, 255))

        cv.imwrite(filename, (image*255)[:size,:size])

    def split_data(self, fractions, augment=None, shuffle=False):
        """
        Splits the data into sets.

        Returns a list of dictionaries with the keys 'maps', 'coords' and
        'brights'.

        fraction
            A list of fractions of the set to allocate to each corresponding
            dictionary.
        augment
            A list of booleans. If true, the set at that index will be augmented
            by flipping images (effectively quadrupaling the size).
        shuffle
            Whether the samples should be shuffled, otherwise the first
            set of samples will be used in order for the first set.
        """
        import numpy as np

        def rotate_image(img, coords):
            size = img[0].shape[0]
            img = np.array([np.rot90(img[0])])
            rot_coords = []
            for x, y in coords:
                ny = size - x
                nx = y
                rot_coords.append((nx, ny))
            return img, rot_coords

        first = {}
        second = {}

        m, c, b = self.maps, self.coords, self.brights
        if shuffle:
            permute = np.random.permutation(len(m))
            m = [m[i] for i in permute]
            c = [c[i] for i in permute]
            b = [b[i] for i in permute]
        else:
            m, c, b = m.copy(), c.copy(), b.copy()

        sets = []
        first = 0
        for i in range(0, len(fractions)):
            p = fractions[i]
            a = augment[i] if augment is not None else False
            num = round(len(m) * p)
            last = first + num
            print('%.2f from' %(p), first, 'to', last - 1, ('(aug)' if a else ''))
            maps = m[first:last]
            coords = c[first:last]
            brights = b[first:last]

            if a:
                anum = num * 4
                amaps = []
                acoords = []
                abrights = []

                for j in range(0, num):
                    img, coord, bright = maps[j], coords[j], brights[j]
                    for k in range(0, 4):
                        amaps.append(img)
                        acoords.append(coord)
                        abrights.append(bright)
                        img, coord = rotate_image(img, coord)

                maps = amaps
                coords = acoords
                brights = abrights

            sets.append({
                'maps': maps,
                'coords': coords,
                'brights': brights
            })
            first = last

        return sets

    def patch_image(self, index):
        """
        Returns an array of image patches and patch offsets for the specified
        image.

        index
            The index of the image to patch.
        """
        import numpy as np

        reshaped_patches = []
        patches, patch_offsets = generate_patches(self.maps[index][0], (224, 224), 4)
        for j in range(0, len(patches)):
            reshaped_patches.append([patches[j]])
        return np.array(reshaped_patches), np.array(patch_offsets)

    def scale_and_patch_image(self, index, scale):
        """
        First scales an image by tiling it along both axes, then returns an
        array of image patches and patch offsets for the specified image.

        index
            The index of the image to patch.
        scale
            The factor for scaling, e.g. 2 will have twice the width and twice
            the height.
        """
        import numpy as np
        import time
        scale = int(scale)
        img = self.maps[index][0]

        before = time.time()
        reshaped_patches = []
        patches, patch_offsets = generate_patches(np.tile(img, [scale, scale]), (224, 224), 4)
        for j in range(0, len(patches)):
            reshaped_patches.append([patches[j]])
        after = time.time()
        return np.array(reshaped_patches), np.array(patch_offsets), after - before

    def num_images(self):
        """
        Returns the number of images.
        """
        return len(self.maps)

    def get_image(self, index):
        """
        Returns the image, source coordinates and source fluxs of the specified
        image.

        index
            The index of the image to get.
        """
        return self.maps[index], self.coords[index], self.brights[index]
