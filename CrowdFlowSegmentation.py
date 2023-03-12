import numpy as np
import cv2 as cv


class CrowdFlowSegmentation:
    """docstring for ClassName."""

    def __init__(self, frame, no_of_frames, stride):
        self.stride = stride
        self.no_of_frames = no_of_frames
        self.frame_no = 1
        self.previous = frame
        self.previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.vid_height = np.shape(self.previous_gray)[0]
        self.vid_width = np.shape(self.previous_gray)[1]
        self.avg_opt_flow = np.zeros(
            (self.vid_height, self.vid_width, 2), dtype=np.float64
        )
        self.mask = np.zeros_like(self.previous)
        self.pixel_pos_streaklines = self.init_pixel_pos_streaklines()
        self.streaklines_color = np.full(
            (self.vid_width // self.stride, self.vid_height // self.stride, 3),
            [0, 255, 255],
        )
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        # Parameters for lucas kanade optical flow
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.p0 = cv.goodFeaturesToTrack(
            self.previous_gray, mask=None, **self.feature_params
        )

    def to_sparse(self, next):
        # Create a mask image for drawing purposes

        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

        self.p0 = cv.goodFeaturesToTrack(
            self.previous_gray, mask=None, **self.feature_params
        )

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(
            self.previous_gray, next_gray, self.p0, None, **self.lk_params
        )

        # Select good points
        if p1 is not None:
            good_next = p1[st == 1].astype(int)
            good_previous = self.p0[st == 1].astype(int)
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_next, good_previous)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv.line(
                self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2
            )
            next_temp = cv.circle(next, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        img = cv.add(next_temp, self.mask)

        # Now update the previous frame and previous points
        self.previous_gray = next_gray.copy()
        self.p0 = good_next.reshape(-1, 1, 2)

        return img

    def to_dense(self, next):
        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(
            self.previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(self.previous)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        self.previous_gray = next_gray.copy()

        return bgr

    def to_streaklines(self, next):
        # if (self.frame_no == 1):
        #    self.pixel_pos_streaklines = self.init_pixel_pos_streaklines(stride)

        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(
            self.previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # we have a 4D array, consider it as 3D for understanding purposes
        # flow contains dx, dy of point at x,y
        # each point in 3D array needs to be looked in flow array, then add dx,dy of that point and store it in next
        # frame position in 3D array

        fi = self.frame_no
        for f in self.pixel_pos_streaklines[self.frame_no : 0 : -1, :, :, :]:
            xi = 0
            for x in f:
                yi = 0
                for y in x:
                    try:
                        x_pos = self.pixel_pos_streaklines[fi - 1, xi, yi, 0]
                        y_pos = self.pixel_pos_streaklines[fi - 1, xi, yi, 1]
                        x_val = round(
                            self.pixel_pos_streaklines[fi - 1, xi, yi, 0]
                            - flow[y_pos, x_pos, 0]
                        )
                        y_val = round(
                            self.pixel_pos_streaklines[fi - 1, xi, yi, 1]
                            - flow[y_pos, x_pos, 1]
                        )
                        if x_val < self.vid_width and y_val < self.vid_height:
                            self.pixel_pos_streaklines[fi, xi, yi, 0] = x_val
                            self.pixel_pos_streaklines[fi, xi, yi, 1] = y_val
                        else:
                            pass
                    except Exception as e:
                        print(e)
                    yi += 1
                xi += 1
            fi -= 1

        # Drawing streaklines

        fi = self.frame_no
        for f in self.pixel_pos_streaklines[self.frame_no : 0 : -1, :, :, :]:
            xi_ind = 0
            for x in f:
                yi_ind = 0
                for y in x:
                    if fi == 0:
                        try:
                            img = cv.line(
                                next, (y[0], y[1]), (y[0], y[1]), (0, 255, 255), 2
                            )
                        except Exception as e:
                            print(e)
                    else:
                        xi = self.pixel_pos_streaklines[fi - 1, xi_ind, yi_ind, 0]
                        yi = self.pixel_pos_streaklines[fi - 1, xi_ind, yi_ind, 1]
                        try:
                            if y[0] == 0 and y[1] == 0:
                                pass
                            else:
                                self.update_streaklines_color(
                                    xi_ind,
                                    yi_ind,
                                    self.pixel_pos_streaklines[0, xi_ind, yi_ind, 0],
                                    self.pixel_pos_streaklines[0, xi_ind, yi_ind, 1],
                                    self.pixel_pos_streaklines[
                                        self.frame_no, xi_ind, yi_ind, 0
                                    ],
                                    self.pixel_pos_streaklines[
                                        self.frame_no, xi_ind, yi_ind, 1
                                    ],
                                )
                                img = cv.line(
                                    next,
                                    (y[0], y[1]),
                                    (xi, yi),
                                    self.streaklines_color[xi_ind, yi_ind].tolist(),
                                    2,
                                )
                        except Exception as e:
                            print(e)
                        pass

                    yi_ind += 1
                xi_ind += 1
            fi -= 1

        if self.frame_no <= 2:
            img = next

        self.previous_gray = next_gray.copy()
        self.frame_no += 1

        return img

    def init_pixel_pos_streaklines(self):
        pixel_pos_streaklines = np.full(
            [
                self.no_of_frames,
                self.vid_width // self.stride,
                self.vid_height // self.stride,
                2,
            ],
            0,
        )
        for index, x in np.ndenumerate(pixel_pos_streaklines[0, :, :, :]):
            x, y, z = index
            if z == 0:
                pixel_pos_streaklines[0, x, y, z] = x * self.stride
            else:
                pixel_pos_streaklines[0, x, y, z] = y * self.stride

        return pixel_pos_streaklines

    def update_streaklines_color(self, xi_ind, yi_ind, x1, y1, x2, y2):
        if x1 < x2 and y1 < y2:
            self.streaklines_color[xi_ind, yi_ind] = self.color[1].tolist()
        elif x1 < x2 and y1 > y2:
            self.streaklines_color[xi_ind, yi_ind] = self.color[2].tolist()
        elif x1 > x2 and y1 < y2:
            self.streaklines_color[xi_ind, yi_ind] = self.color[3].tolist()
        elif x1 > x2 and y1 > y2:
            self.streaklines_color[xi_ind, yi_ind] = self.color[4].tolist()
        else:
            pass

    # expects image path and returns watershed image

    def to_similarity(self, next):
        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(
            self.previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.avg_opt_flow += flow
        avg_opt_flow = self.avg_opt_flow / (self.frame_no + 2)
        # self.avg_opt_flow+=flow
        mag, ang = cv.cartToPolar(avg_opt_flow[..., 0], avg_opt_flow[..., 1])

        hsv = np.zeros_like(self.previous)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        self.previous_gray = next_gray.copy()

        return bgr

    def to_watershed(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]

        return sure_fg + unknown
