import cv2

THRESHOLD = 0.8

def detect_face(image_path, size=(287,392)):
    """Detects the largest face in an image and resizes it to size parameter.

    Parameters:
      image_path: Path to the input image.
      size: A size (width, height) to resize the face image to.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    (x, y, w, h) = faces[0]
    face_img = image[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, size)
    return face_img

def compare_faces(face1, face2):
    """Compares two faces by calculating correlation between their histograms.

    Parameters:
      face1: First (actual / ground truth) face image.
      face2: Second (test) face image.
    """
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation

if __name__ == "__main__":
    face_img1 = detect_face('actual_face.jpeg')
    face_img2 = detect_face('unswirled_face.jpeg')

    if face_img1 is not None and face_img2 is not None:
        similarity_score = compare_faces(face_img1, face_img2)
        print("Similarity Score:", similarity_score)
        if similarity_score > THRESHOLD:
            print("Faces might be of the same person.")
        else:
            print("Faces likely of different people.")
    else:
        print("Face(s) not detected.")
