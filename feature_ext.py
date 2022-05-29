from skimage.measure import *
import pandas as pd  
import numpy as np
import glob
import cv2 as cv
from skimage.io import imsave
from PIL import Image


cols = ['h_asym', 'v_asym', 'Nb_colors', 'Asym_Black', 'Asym_Blue Gray', 'Asym_Dark Brown', 'Asym_Light Brown',
                     'Asym_White', 'max_axe', 'min_axe']
Height, Width = 288, 384

def extract_feature(image, mask, contour):
    """
    This method extracts Asymmetry, Border, and Diamter features along with
    lesion area, centroid, and perimeter. Performs affine transformation
    :param image: 3-d numpy array of an RGB image
    :param mask: binary image of the lesion mask image
    :param contour: list of contour points of the lesion
    :return: a list of all the features along with area, centroid,
    perimeter of the lesion, and transformed image
    """
    moments = cv.moments(contour)
    #            contour_area = moments['m00']
    # (mH, mW) = mask.shape[:2]
    contour_area = cv.countNonZero(mask)
    # try:
    contour_centroid = [int(moments['m10'] / (moments['m00']+1e-8)),
                        int(moments['m01'] / (moments['m00']+1e-8))]
    contour_perimeter = cv.arcLength(contour, True)

    # Get max and min diameter
    rect = cv.fitEllipse(contour)
    (x, y) = rect[0]
    (w, h) = rect[1]
    angle = rect[2]

    if w < h:
        if angle < 90:
            angle -= 90
        else:
            angle += 90
    rows, cols = mask.shape
    rot = cv.getRotationMatrix2D((x, y), angle, 1)
    cos = np.abs(rot[0, 0])
    sin = np.abs(rot[0, 1])
    nW = int((rows * sin) + (cols * cos))
    nH = int((rows * cos) + (cols * sin))

    rot[0, 2] += (nW / 2) - cols / 2
    rot[1, 2] += (nH / 2) - rows / 2

    warp_mask = cv.warpAffine(mask, rot, (nH, nW))
    warp_img = cv.warpAffine(image, rot, (nH, nW))
    warp_img_segmented = cv.bitwise_and(warp_img, warp_img,
                                            mask=warp_mask)

    cnts, hierarchy = cv.findContours(warp_mask, cv.RETR_TREE,
                                                cv.CHAIN_APPROX_NONE)
    areas = [cv.contourArea(c) for c in cnts]
    if len(areas)!=0:
        contour = cnts[np.argmax(areas)]
        xx, yy, nW, nH = cv.boundingRect(contour)
        #        cv2.rectangle(warp_mask,(xx,yy),(xx+w,yy+h),(255,255,255),2)
        warp_mask = warp_mask[yy:yy + nH, xx:xx + nW]

        # get horizontal asymmetry
        flipContourHorizontal = cv.flip(warp_mask, 1)
        flipContourVertical = cv.flip(warp_mask, 0)

        diff_horizontal = cv.compare(warp_mask, flipContourHorizontal,
                                        cv.CV_8UC1)
        diff_vertical = cv.compare(warp_mask, flipContourVertical,
                                    cv.CV_8UC1)

        diff_horizontal = cv.bitwise_not(diff_horizontal)
        diff_vertical = cv.bitwise_not(diff_vertical)

        h_asym = cv.countNonZero(diff_horizontal)
        v_asym = cv.countNonZero(diff_vertical)
        features = [{'area': int(contour_area), 'centroid': contour_centroid,
                    'perimeter': int(contour_perimeter),
                    'max_axe': max([nW, nH]), 'min_axe': min([nW, nH]),  # Normalize params
                    'h_asym': round(float(h_asym) / contour_area, 2),
                    'v_asym': round(float(v_asym) / contour_area, 2)},
                cv.bitwise_not(diff_horizontal),
                cv.bitwise_not(diff_vertical),
                warp_img_segmented]
        return features
def color_contours(frame, hsv, colors, max_area):
    """
    Extract the color contour of a region determined by upper and lower
    threshold values of colors.
    :param frame:  3-d numpy array of an RGB image
    :param hsv: 3-d numpy array of an HSV image
    :param colors: A list of tuples containing upper and lower color thresholds
    :param max_area: Total object area
    :return: numpy array of contour points if color is detected else None
    """
    hsv_low = colors[0]
    hsv_high = colors[1]
    mask = cv.inRange(hsv, hsv_low, hsv_high)
    res = cv.bitwise_and(frame, frame, mask=mask)
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(res, cv.RETR_TREE,
                                                cv.CHAIN_APPROX_NONE)
    cnt = len(contours)
    if cnt > 0:
        new_contours = []
        for i in np.arange(cnt):
            a = cv.contourArea(contours[i])
            # To determine the color is not spurious noise
            if a > max_area * 0.02:
                new_contours.append(contours[i])

        return np.array(new_contours)
    else:
        return []
def extract_largest_contour(gray_image):
    mask_contours, hierarchy = \
        cv.findContours(gray_image, cv.RETR_EXTERNAL,
                         cv.CHAIN_APPROX_NONE)
    cnt = len(mask_contours)
    if cnt > 0:
        area = np.zeros(cnt)
        for i in np.arange(cnt):
            area[i] = cv.contourArea(mask_contours[i])
        max_area_pos = np.argpartition(area, -1)[-1:][0]
        return [mask_contours, max_area_pos]
    else:
        return []
def colorometric_features(features, image, mask, org_image):
    tolerance = 30
    image = cv.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    value_threshold = np.uint8(cv.mean(hsv_image)[2]) \
                            - tolerance
    hsv_colors = { 
        'Blue Gray': [np.array([15, 0, 0]),
                        np.array([179, 255, value_threshold]),
                        (0, 153, 0), 'Blue Gray'],  # Green
        'White': [np.array([0, 0, 145]),
                    np.array([15, 80, value_threshold]),
                    (255, 255, 0), 'White'],  # Cyan
        'Light Brown': [np.array([0, 80, value_threshold + 3]),
                        np.array([15, 255, 255]), (0, 255, 255), 'Light Brown'],
        # Yellow
        'Dark Brown': [np.array([0, 80, 0]),
                        np.array([15, 255, value_threshold - 3]),
                        (0, 0, 204), 'Dark Brown'],  # Red
        'Black': [np.array([0, 0, 0]), np.array([15, 140, 90]),
                    (0, 0, 0), 'Black'],  # Black
    }
    iter_colors = [
        [50, (0, 0, 255)],
        [100, (0, 153, 0)],
        [200, (255, 255, 0)],
        [400, (255, 0, 0)]
    ]
    borders = 2
    ret_val = extract_largest_contour(mask)
    mask_contours = ret_val[0]
    max_area_pos = ret_val[1]
    contour = mask_contours[max_area_pos]
    cnt = len(mask_contours)
    contour_image = org_image.copy()
    color=(255, 0, 0)
    if cnt > 0:
        cv.drawContours(contour_image, mask_contours,
                            max_area_pos,
                            color,
                            2)
        contour_binary = np.zeros(image.shape[:2],
                                        dtype=np.uint8)
        cv.drawContours(contour_binary, mask_contours,
                            max_area_pos,
                            255,
                            2)
        contour_area = cv.contourArea(contour)
        segmented_img = cv.bitwise_and(
            org_image, org_image,
            mask=mask)
        segmented_img[segmented_img == 0] = 255
    no_of_colors = []
    feature_set = features[0]
    # dist = []
    hsv = cv.cvtColor(segmented_img, cv.COLOR_BGR2HSV)
    color_contour = np.copy(org_image)
    for color in hsv_colors:
        #            print color
        cnt = color_contours(segmented_img, hsv,
                                    hsv_colors[color],
                                    contour_area)
        centroid = []
        color_attr = {}
        if len(cnt) > 0:
            for contour in cnt:
                moments = cv.moments(contour)
                if moments['m00'] == 0:
                    continue
                color_ctrd = [int(moments['m10'] / moments['m00']),
                                int(moments['m01'] / moments['m00'])]

                centroid.append(color_ctrd)
        if len(centroid) != 0:
            cv.drawContours(color_contour, cnt, -1,
                                hsv_colors[color][2],
                                2)
            asym_color = np.mean(np.array(centroid), axis=0)
            dist = ((asym_color[0] -
                        feature_set['centroid'][
                            0]) ** 2 + (asym_color[1] -
                                        feature_set['centroid'][
                                            1]) ** 2) ** 0.5
            color_attr['color'] = color
            color_attr['centroids'] = centroid
            feature_set['Asym_' + hsv_colors[color][3]] = \
                round(dist / feature_set['max_axe'], 4)
            no_of_colors.append(color_attr)
        else:
            feature_set['Asym_' + hsv_colors[color][3]] = 0
    feature_set['colors_attr'] = no_of_colors
    feature_set['Nb_colors'] = len(no_of_colors)
    feature_vector = np.array([feature_set[col] for col in cols],
                                  dtype=np.float32)
    return feature_vector

def Geometric_features(mask, image, name):
    regions = regionprops(mask, image)
    for item, prop in enumerate(regions):
        props = dir(prop)
    proprieties = [i for i in props if (not i.startswith('_') and i[-1]!='_')]
    to_remove = ['slice', 'image_intensity', 'image_filled', 'image_convex', 'image', 'coords', 'centroid_weighted', 'bbox']
    [proprieties.remove(i) for i in  to_remove]
    props = regionprops_table((mask>0).astype(int), image, properties=proprieties)
    formfactor = 4.0 * np.pi * props["area"][0] / props["perimeter"][0] ** 2
    denom = max(4.0 * np.pi * props["area"][0], 1)
    compactness = props["perimeter"][0] ** 2 / denom
    data = pd.DataFrame(props)
    data['compactness'] = compactness
    data['formfactor'] = formfactor
    data['ID'] = name
    data.insert(0, 'ID', data.pop('ID'))
    return data
def merge(name, data, feature_vector, cols):
    dict={'ID':name}
    for i in range(len(cols)):
        dict[cols[i]] =  feature_vector[i]
    df = pd.DataFrame(dict, index=[0])
    merged = pd.merge(data[data['ID']==name], df, on="ID")
    return merged

def main(mask_files, image_files):
    data = pd.DataFrame()
    count = 0
    for i in range(len(mask_files)):
        org_image = cv.resize(cv.imread(image_files[i]), (Height, Width))
        name = image_files[i].split('/')[-1].split('.')[0]
        print('i, name', i, name)
        image = org_image.copy()
        image_ = org_image.copy()
        mask = cv.resize(cv.imread(mask_files[i]), (Height, Width))
        if mask.ndim!=2:
            mask = mask[:,:,0]
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(image_,contours,0,255,-1)
        # imsave('example_'+str(i)+'.jpg',image_)
        # Extract Geometrical features
        df = Geometric_features(np.array(mask), np.array(image), name)
        if len(contours[0]) > 5:
            features = extract_feature(image, mask, contours[0])
            # Extract Coloremetric features
            if features is not None:
                if len(features)!=0:
                    feature_vector = colorometric_features(features, image, mask,org_image)
                    merged =  merge(name, df, feature_vector, cols)
                    data = pd.concat([data, merged])
                    data.to_csv('tmp.csv')
                else:
                    count+=1
            else:
                count+=1    
        else:
            count+=1
    print(count, 'images out of',len(mask_files),' , had less than 5 elements in the contours')
    return data
# if __name__ == "__main__":
#     data = main()
