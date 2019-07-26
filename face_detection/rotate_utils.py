import numpy as np
import cv2
import detect_utils

def draw_5_point(img, shape, color=(0,255,0), thickness=2):
    image_cv2 = cv2.line(img, (shape.part(0).x, shape.part(0).y), 
                              (shape.part(1).x, shape.part(1).y), 
                              color, thickness)
    image_cv2 = cv2.line(image_cv2, (shape.part(2).x, shape.part(2).y), 
                              (shape.part(3).x, shape.part(3).y), 
                              color, thickness)
    image_cv2 = cv2.line(image_cv2, (shape.part(3).x, shape.part(3).y), 
                              (shape.part(4).x, shape.part(4).y), 
                              color, thickness)
    image_cv2 = cv2.line(image_cv2, (shape.part(1).x, shape.part(1).y), 
                              (shape.part(4).x, shape.part(4).y), 
                              color, thickness)
    return image_cv2

def get_shape_detector(model_path="models/shape_predictor_5_face_landmarks.dat"):
    landmark_predictor = dlib.shape_predictor(model_path)
    return landmark_predictor

def extract_left_eye_center(shape):
    x = (shape.part(0).x + shape.part(1).x) // 2
    y = (shape.part(0).y + shape.part(1).y) // 2
    return (x, y)

def extract_right_eye_center(shape):
    x = (shape.part(2).x + shape.part(3).x) // 2
    y = (shape.part(2).y + shape.part(3).y) // 2
    return (x, y)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        if y2 > y1:
            return 90
        else: 
            return -90
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M, angle, (xc, yc)

class FaceAlignment:
    def __init__(self, desired_size=(299,299), desired_left_eye=(0.3, 0.3), landmark_predictor=None):
        '''
        desired_size: target size of aligned photo
        desired_left_eye: target left eye position after aligned (relative)
        '''
        
        self.desired_size = desired_size
        self.desired_left_eye = desired_left_eye
        self.landmark_predictor = landmark_predictor
        
    def align_from_path(self, img_path, bbox):
        img = self.load_rgb_image(img_path)
        rect = self.cvt_bbox2rect(bbox)
        landmark = self.predict_landmark(img, rect)
        img_aligned = self.align(img, landmark)
        return img_aligned
        
    def load_rgb_image(self, path):
        return dlib.load_rgb_image(path)
    
    def cvt_bbox2rect(self, bbox):
        return dlib.rectangle(*bbox)
    
    def predict_landmark(self, img, rect):
        return self.landmark_predictor(img, rect)
        
    def align(self, img, landmark):
        # get eyes, rotation
        left_eye = self._extract_left_eye_center(landmark)
        right_eye = self._extract_right_eye_center(landmark)
        _, angle, center = self._get_rotation_matrix(left_eye, right_eye)

        # calculate translation matrix
        desired_face_width = self.desired_size[0]
        desired_face_height = self.desired_size[1]

        self.desired_right_eye = 1.0 - self.desired_left_eye[0]

        desired_distance = (self.desired_right_eye - self.desired_left_eye[0])
        desired_distance *= desired_face_width
        distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)

        scale = desired_distance/distance

        translation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                        (left_eye[1] + right_eye[1]) // 2)

        tX = desired_face_width * 0.5
        tY = desired_face_height * self.desired_left_eye[1]

        translation_matrix[0, 2] += (tX - eyes_center[0])
        translation_matrix[1, 2] += (tY - eyes_center[1])

        img_aligned = cv2.warpAffine(img, translation_matrix, 
                                    (desired_face_width, desired_face_height), 
                                    flags=cv2.INTER_CUBIC)
        return img_aligned
        
        
    def _extract_left_eye_center(self, shape):
        x = (shape.part(0).x + shape.part(1).x) // 2
        y = (shape.part(0).y + shape.part(1).y) // 2
        return (x, y)

    def _extract_right_eye_center(self, shape):
        x = (shape.part(2).x + shape.part(3).x) // 2
        y = (shape.part(2).y + shape.part(3).y) // 2
        return (x, y)

    def _angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x2 - x1 == 0:
            if y2 > y1:
                return 90
            else: 
                return -90
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def _get_rotation_matrix(self, p1, p2):
        angle = self._angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M, angle, (xc, yc)
            
        
        
    def _extract_left_eye_center(self, shape):
        x = (shape.part(0).x + shape.part(1).x) // 2
        y = (shape.part(0).y + shape.part(1).y) // 2
        return (x, y)

    def _extract_right_eye_center(self, shape):
        x = (shape.part(2).x + shape.part(3).x) // 2
        y = (shape.part(2).y + shape.part(3).y) // 2
        return (x, y)

    def _angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x2 - x1 == 0:
            if y2 > y1:
                return 90
            else: 
                return -90
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def _get_rotation_matrix(self, p1, p2):
        angle = self._angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M, angle, (xc, yc)
            

def face_align_dlib_from_path(img_path, landmark_predictor, face_coor_array=None,
                    desired_size=(299,299), desired_left_eye=(0.3, 0.3)):
    # load data
    im = dlib.load_rgb_image(img_path)
    if face_coor_array is None:
        face_coor_array = [0, 0, im.shape[0], im.shape[1]]
    rect = dlib.rectangle(*face_coor_array)
    
    # get shape
    face_landmark = landmark_predictor(im, rect)

    # get eyes, rotation
    left_eye = extract_left_eye_center(face_landmark)
    right_eye = extract_right_eye_center(face_landmark)
    _, angle, center = get_rotation_matrix(left_eye, right_eye)
    
    # calculate translation matrix
    desired_face_width = desired_size[0]
    desired_face_height = desired_size[1]

    desired_left_eye = desired_left_eye
    desired_right_eye = 1.0 - desired_left_eye[0]

    desired_distance = (desired_right_eye - desired_left_eye[0])
    desired_distance *= desired_face_width
    distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)

    scale = desired_distance/distance
    
    translation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                    (left_eye[1] + right_eye[1]) // 2)
    
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    
    translation_matrix[0, 2] += (tX - eyes_center[0])
    translation_matrix[1, 2] += (tY - eyes_center[1])

    im_aligned = cv2.warpAffine(im, translation_matrix, 
                                (desired_face_width, desired_face_height), 
                                flags=cv2.INTER_CUBIC)
    return im_aligned

def face_align_dlib(im, face_landmark, rect,
                    desired_size=(299,299), desired_left_eye=(0.3, 0.3)):
    # get eyes, rotation
    left_eye = extract_left_eye_center(face_landmark)
    right_eye = extract_right_eye_center(face_landmark)
    _, angle, center = get_rotation_matrix(left_eye, right_eye)
    
    # calculate translation matrix
    desired_face_width = desired_size[0]
    desired_face_height = desired_size[1]

    desired_left_eye = desired_left_eye
    desired_right_eye = 1.0 - desired_left_eye[0]

    desired_distance = (desired_right_eye - desired_left_eye[0])
    desired_distance *= desired_face_width
    distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)

    scale = desired_distance/distance
    
    translation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                    (left_eye[1] + right_eye[1]) // 2)
    
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    
    translation_matrix[0, 2] += (tX - eyes_center[0])
    translation_matrix[1, 2] += (tY - eyes_center[1])

    im_aligned = cv2.warpAffine(im, translation_matrix, 
                                (desired_face_width, desired_face_height), 
                                flags=cv2.INTER_CUBIC)
    return im_aligned