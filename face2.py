import cv2
import numpy as np
from PIL import Image
import os

# Load the DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Dictionary to store user names mapped to their IDs
user_mapping = {}

# Function to generate dataset by capturing images
def generate_dataset():
    # Ask for user details
    user_name = input("Enter your name: ").strip()
    user_id = input("Enter your ID (numeric): ").strip()

    # Store the user ID and name in the mapping
    user_mapping[int(user_id)] = user_name

    # Create a directory to store images if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    def face_cropped(img):
        try:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    cropped_face = img[y:y1, x:x1]
                    return cropped_face
            return None
        except Exception as e:
            print(f"Error in face_cropped: {e}")
            return None

    cap = cv2.VideoCapture(0)  # Default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        face = face_cropped(frame)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{user_id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            print(f"Saved image: {file_name_path}")
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)

            if cv2.waitKey(1) == 13 or img_id == 200:  # Stop after 500 images or Enter key
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collecting samples is completed for {user_name} (ID: {user_id})...")

# Function to train the classifier on the collected dataset
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    faces = []
    ids = []

    if len(path) == 0:
        print("Error: No images found in the dataset directory.")
        return

    for image_path in path:
        img = Image.open(image_path).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Training completed and model saved as 'classifier.xml'.")

# Function to draw boundary and recognize faces
def draw_boundary(img, classifier, scaleFactor, minNeighbour, color, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    classifier.setInput(blob)
    detections = classifier.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face = gray_image[y:y1, x:x1]
            try:
                id, pred = clf.predict(face)
                confidence_score = int(100 * (1 - pred / 300))

                label = "UNKNOWN"
                label_color = (255, 0, 0)  # Red for unknown
                if confidence_score > 73:
                    # Fetch the user's name from the mapping
                    label = user_mapping.get(id, "Raghav")
                    label_color = (0, 255, 0)  # Green for known

                cv2.rectangle(img, (x, y), (x1, y1), label_color, 2)
                cv2.putText(img, f"{label} {confidence_score}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 1, cv2.LINE_AA)
            except:
                continue

# Function to recognize faces in the video stream
def recognize(img, clf):
    draw_boundary(img, face_net, 1.1, 10, (255, 255, 255), clf)
    return img

# Main function to run face recognition
def run_face_recognition():
    if not os.path.exists("classifier.xml"):
        print("Error: Trained model 'classifier.xml' not found. Train the model first.")
        return

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, img = video_capture.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        img = recognize(img, clf)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Enter key to break
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Uncomment below lines to first generate dataset and then train the classifier
generate_dataset()
train_classifier("data")

# Uncomment to run face recognition
run_face_recognition()