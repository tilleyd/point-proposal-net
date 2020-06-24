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
                        img_coords.append( (int(np.round(x)), int(np.round(y))) )
                        img_brights.append( float(l[2]) )

            if not patch_images:
                self.maps.append([img])
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
                    self.maps.append([patches[j]])
                    self.coords.append(patch_coords)
                    self.brights.append(patch_brights)


        print('Done, read %d sources' %(source_count))

    def display_image(self, index, show_sources=False):
        """
        Opens a window displaying the image at the given index.

        index
            The index of the image to display.
        show_sources
            If true, a red dot will be drawn in the location of each labeled
            source.
        """
        import numpy as np
        import cv2 as cv

        image = self.maps[index][0]

        if show_sources:
            image = np.transpose(np.tile([image], (3, 1, 1)), [1, 2, 0])
            for x,y in self.coords[index]:
                image[y][x] = [0.0, 0.0, 1.0]

        cv.namedWindow('ppn-image', cv.WINDOW_AUTOSIZE)
        cv.imshow('ppn-image', image)
        cv.waitKey()

    def save_image(self, filename, index, show_sources=False):
        """
        Saves the image at the given index to disk.

        filename
            The name of the saved image.
        index
            The index of the image to save.
        show_sources
            If true, a red dot will be drawn in the location of each labeled
            source.
        """
        import cv2 as cv

        image = self.maps[index]

        if show_sources:
            image = np.transpose(np.tile([image], (3, 1, 1)), [1, 2, 0])
            for x,y in self.coords[index]:
                image[y][x] = [0.0, 0.0, 1.0]

        cv.imwrite(filename, image*255)

    def split_data(self, fractions, shuffle=False):
        """
        Splits the data into two sets.

        Returns a list of dictionaries with the keys 'maps', 'coords' and
        'brights'.

        fraction
            A list of fractions of the set to allocate to each corresponding
            dictionary.
        shuffle
            Whether the samples should be shuffled, otherwise the first
            set of samples will be used in order for the first set.
        """
        first = {}
        second = {}

        m, c, b = self.maps, self.coords, self.brights
        if shuffle:
            import numpy as np
            permute = np.random.permutation(len(m))
            m = [m[i] for i in permute]
            c = [c[i] for i in permute]
            b = [b[i] for i in permute]
        else:
            m, c, b = m.copy(), c.copy(), b.copy()

        sets = []
        first = 0
        for p in fractions:
            num = round(len(m) * p)
            last = first + num
            print('%.2f from' %(p), first, 'to', last)
            sets.append({
                'maps': m[first:last],
                'coords': c[first:last],
                'brights': b[first:last]
            })
            first = last

        return sets
