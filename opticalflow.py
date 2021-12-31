import numpy as np
import cv2 as cv

class CrowdFlowSegmentation:
    """docstring for ClassName."""
    def __init__(self, frame, no_of_frames):
        self.no_of_frames = no_of_frames
        self.frame_no = 0
        self.previous = frame
        self.previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.vid_height = np.shape(self.previous_gray)[0]
        self.vid_width = np.shape(self.previous_gray)[1]
        self.mask = np.zeros_like(self.previous)
        self.pixel_pos_streaklines = self.init_pixel_pos_streaklines()
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        # Parameters for lucas kanade optical flow 
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                               maxLevel = 2,
                               criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = cv.goodFeaturesToTrack(self.previous_gray, mask = None, **self.feature_params) 
 

    def to_sparse(self, next):
        # Create a mask image for drawing purposes

        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)

        self.p0 = cv.goodFeaturesToTrack(self.previous_gray, mask = None, **self.feature_params)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.previous_gray, next_gray, self.p0, None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_next = p1[st==1].astype(int)
            good_previous = self.p0[st==1].astype(int)
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_next, good_previous)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            next_temp = cv.circle(next, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        img = cv.add(next_temp, self.mask)
    
        # Now update the previous frame and previous points
        self.previous_gray = next_gray.copy()
        self.p0 = good_next.reshape(-1, 1, 2)

        return img

    def to_dense(self, next):
        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(self.previous)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
        self.previous_gray = next_gray.copy()

        return bgr

    def to_streaklines(self, next):
        next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # we have a 4D array, consider it as 3D for understanding purposes
        # flow contains dx, dy of point at x,y
        # each point in 3D array needs to be looked in flow array, then add dx,dy of that point and store it in next
        # frame position in 3D array
        for index, x in np.ndenumerate(self.pixel_pos_streaklines[self.no_of_frames,:,:,:]):
            x,y,z = index
            if z == 0:
                flow = cv2.remap(self.previous_gray, )
                self.pixel_pos_streaklines[0,x,y,z] = self.pixel_pos_streaklines[0,x-1,y,z] - flow[..., 0]
            else:
                self.pixel_pos_streaklines[0,x,y,z] = self.pixel_pos_streaklines[0,x,y-1,z] - flow[..., 1]
                new_frame = cv2.remap(prev_frame, pixel_map)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(self.previous)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
        self.previous_gray = next_gray.copy()
        self.no_of_frames+=1

        return bgr

    def init_pixel_pos_streaklines(self):
        pixel_pos_streaklines = np.full([self.no_of_frames, self.vid_width, self.vid_height, 2], None) 
        for index, x in np.ndenumerate(pixel_pos_streaklines[0,:,:,:]):
            x,y,z = index
            if z == 0:
                pixel_pos_streaklines[0,x,y,z] = x
            else:
                pixel_pos_streaklines[0,x,y,z] = y

        return pixel_pos_streaklines

# expects image path and returns watershed image
    def to_watershed(self, img):

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        
        markers = cv.watershed(img,markers)
        img[markers == -1] = [255,0,0]
    
        return img
