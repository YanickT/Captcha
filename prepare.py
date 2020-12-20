import numpy as np
from skimage import measure

THRESHOLD = 20


def totalize(img):
    """
    Calculate threshold and map all colors below to withe (#FFFFFF) and all above to black (#000000).
    :param img: PIL.Image = Image to totalize
    :return: np.array[height, width, 3] = totalized pixels
    """
    pixels = np.asarray(img)
    line_grey = min([pixels[0, x, 0] for x in range(pixels.shape[1] // 2)])
    # Since all pixels are a shade of grey the values in rgb match (r==g==b)
    pixels = np.where(pixels > line_grey - THRESHOLD, 255, 0)
    return pixels


def reduce_boxes(boxes, n_o_b):
    """
    Reduces the number of boxes by combining them. Combine the closesd boxes
    :param boxes: List[List[int]] = List of bboxes of the chars
    :param n_o_b: int = wanted number of boxes
    :return: List[List[int]] = modified boxes
    """

    while len(boxes) > n_o_b:
        sizes = list(map(lambda box: box[2] - box[0], boxes))
        mini = min(sizes)
        index = sizes.index(mini)
        # select the smallest box

        if index == 0:
            # only one neighbour (second)
            neighbour = 1
        elif index == len(boxes) - 1:
            # only one neighbour (next to last)
            neighbour = index - 1
        else:
            # choose neighbour with smallest distance between borders
            neighbour = 2 * int(
                boxes[index][0] - boxes[index - 1][2] > boxes[index + 1][0] - boxes[index][2]) - 1 + index

        if neighbour > index:
            boxes[neighbour] = (boxes[index][0],
                                min(boxes[index][1], boxes[neighbour][1]),
                                boxes[neighbour][2],
                                max(boxes[index][3], boxes[neighbour][3]))
        else:
            boxes[neighbour] = (boxes[neighbour][0],
                                min(boxes[index][1], boxes[neighbour][1]),
                                boxes[index][2],
                                max(boxes[index][3], boxes[neighbour][3]))

        del boxes[index]
    return boxes


def extract_chars(img):
    """
    Extract the pixels for each of the five chars.
    :param img: PIL.Image = Image which contains the chars
    :return: List[np.array[size, width, 3]] = List of the extracted chars
    """
    chars = [np.zeros((20, 20), np.int32) for i in range(5)]  # check if size is really necessary!

    pixels = totalize(img)[:, :, 0]
    pixels = np.rot90(pixels, 3)
    # rotate pixels, this will cause the first segement (from left to right) to be labeled 1, the second one 2
    # since measure.label labels them from top to bottom

    labeled = measure.label(pixels, connectivity=2, background=255)
    boxes = [region.bbox for region in measure.regionprops(labeled)]

    if len(boxes) > 5:
        # to many boxes => reduce number of boxes
        boxes = reduce_boxes(boxes, 5)

    # to few boxes (expected 5)
    if 2 < len(boxes) < 5:
        labeled2 = measure.label(pixels, connectivity=1, background=255)
        boxes2 = [region.bbox for region in measure.regionprops(labeled2)]

        to_modify = [box in boxes for box in boxes2]
        # due to connectivity mode 1 does boxes2 is larger than boxes
        # check which boxes exist in both

        to_correct = 5 - to_modify.count(True)
        if to_correct > 3:
            # to complicated and would take to long => dump image
            return []

        # check which elements coincide
        segments_state = [to_modify[0]]
        for state in to_modify[1:]:
            if state:
                segments_state.append(True)
            elif segments_state[-1]:
                segments_state.append(False)

        if segments_state.count(False) > 1:
            # to complicated and would take to long => dump image
            return []

        # determine the segment of false (not coincide) boxes
        start_index = to_modify.index(False)
        if start_index == len(segments_state) - 1:
            segment = boxes2[start_index:]
            pre = boxes2[:start_index]
            suf = []
        else:
            end_index = to_modify.index(True, start_index)
            segment = boxes2[start_index: end_index]
            pre = boxes2[:start_index]
            suf = boxes2[end_index:]

        # split segment into to_correct boxes
        modified_segment = reduce_boxes(segment, to_correct)
        # recombine to (hopefully) correct boxes
        boxes = pre + modified_segment + suf

    for i, region in enumerate(boxes):
        char_array = np.rot90(pixels[region[0]:min(region[2] + 1, region[0] + 20),
                              region[1]:min(region[3] + 1, region[1] + 20)])
        chars[i][:char_array.shape[0], :char_array.shape[1]] = char_array
    return chars
