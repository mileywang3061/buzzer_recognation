import cv2
import time


def get_information(framerate):
    """
    :param framerate:
    :return:
    """
    framelength = 0.025  # 帧长20~30ms
    framesize = framelength * framerate  # 每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
    NFFT = framesize  # NFFT必须与时域的点数framsize相等，即不补零的FFT
    overlapSize = 2.0 / 5 * framesize
    overlapSize = int(round(overlapSize))  # 取整

    return NFFT, framesize, overlapSize


def buzzer_voice_detect(gray_image):
    """
     This function is using in the algorithm of the first edition
     not be used in the second edition(the rules)
     """
    for k in range(len(gray_image)):
        if (gray_image[k] == 0 and gray_image[k - 1] == 0 and gray_image[k - 2] == 1 and gray_image[k - 3] == 0 and
            gray_image[k - 4] == 0) or (gray_image[k] == 0 and gray_image[k - 1] == 0 and gray_image[k - 2] == 0 and gray_image[k - 3] == 1 and
                                        gray_image[k - 4] == 1 and gray_image[k - 5] == 0 and gray_image[k - 6] == 0):
            # print('yes')
            # print('we find fengming')
            # break
            return True
    return False


def getting_list(image):
    """
     This function is using in the algorithm of the first edition
     not be used in the second edition(the way to get the balck-and-white_image)
     """

    result = []
    for i in range(int(image.shape[0])):
        count = 0
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                count = count + 1
            else:
                count = count
        if count <= int(image.shape[1] / 2) & count >= int(image.shape[1] / 3):
            result.append(1)
        else:
            result.append(0)
    return result


def take_first(res):
    return res[0]


def time_changeing(time_given, time_predict):
    time_multi = time_given / time_predict
    return time_multi


def rules(img, threshold, time_multi):
    """
     This function is the main rule of the second edition algorithm

     """
    # get the gray image , balck-and-white_image and get the connected domain satisfying conndtions
    # img = cv2.imread(im_path)
    # img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # get the center of the rectangles of all the connected domain satisfying conndtions
    cen_x = []
    for i in range(len(contours)):
        st_x, st_y, width, height = cv2.boundingRect(contours[i])
        if width >= int(15 * time_multi) and height >=8  and width > height and int(width / height) >= int(2*time_multi):
            cv2.rectangle(img_bin, (st_x, st_y), (st_x + width, st_y + height), (0, 255, 0), 2)
            x = st_x + int(width / 2)
            y = st_y + int(height / 2)
            cen_x.append([x, y])
    # sort the list on the lateral axis
    cen_x.sort(key=take_first)
    # compare the vertical axis of all the selected center point and find out the point nearly on the same level
    count = 0
    for i in range(0, len(cen_x)):
        cen_y = []
        for j in range(i, len(cen_x)):
            if cen_x[i][1] - cen_x[j][1] >= -7  and cen_x[i][1] - cen_x[j][1] <= 7 :
                cen_y.append(cen_x[j])
        # after delet the point unsatisfied on the same level, compare two point to find out whether they cound find out period
        for k in range(0, len(cen_y)):
            if len(cen_y) >= 3:
                if cen_y[k][0] - cen_y[k - 1][0] >= cen_y[k - 1][0] - cen_y[k - 2][0] - int(35 * time_multi) and cen_y[k][0] - cen_y[k - 1][0] <= cen_y[k - 1][
                    0] - cen_y[k - 2][
                    0] + int(35 * time_multi):
                    count = count + 1
    if count >= 1:
        return True
    else:
        return False
