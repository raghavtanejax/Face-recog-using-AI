import cv2
import numpy as np
from PIL import Image
import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, Label, Button
import threading
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.data_dir = "data"
        self.users_file = "users.json"
        self.classifier_file = "classifier.xml"
        self.face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.users = self.load_users()
        
        # UI Colors
        self.theme_bg = "#050505"      # Almost black
        self.theme_fg = "#00f0ff"      # Cyber Cyan
        self.theme_accent = "#00ff41"  # Hacker Green
        self.theme_font = ("Courier New", 12, "bold")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading users: {e}")
                return {}
        return {}

    def save_users(self):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save users: {e}")

    def face_cropped(self, img):
        try:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    # Clamp coordinates
                    x, y = max(0, x), max(0, y)
                    x1, y1 = min(w, x1), min(h, y1)
                    
                    cropped_face = img[y:y1, x:x1]
                    return cropped_face
            return None
        except Exception as e:
            print(f"Error in face_cropped: {e}")
            return None

    def draw_tech_brackets(self, img, x, y, x1, y1, color, thickness=2, length=20):
        # Top-Left
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        # Top-Right
        cv2.line(img, (x1, y), (x1 - length, y), color, thickness)
        cv2.line(img, (x1, y), (x1, y + length), color, thickness)
        # Bottom-Left
        cv2.line(img, (x, y1), (x + length, y1), color, thickness)
        cv2.line(img, (x, y1), (x, y1 - length), color, thickness)
        # Bottom-Right
        cv2.line(img, (x1, y1), (x1 - length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), color, thickness)

    def generate_dataset(self):
        user_name = simpledialog.askstring("INPUT REQ", "ENTER USER ALIAS:")
        if not user_name: return
        user_id = simpledialog.askinteger("INPUT REQ", "ENTER NUMERIC ID:")
        if user_id is None: return

        if str(user_id) in self.users:
             if not messagebox.askyesno("WARNING", f"ID {user_id} USED BY {self.users[str(user_id)]}. OVERWRITE?"):
                 return

        cap = cv2.VideoCapture(0)
        img_id = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            face = self.face_cropped(frame)
            
            # HUD Overlay for Capture
            cv2.putText(frame, "STATUS: DATALOGGING", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            cv2.putText(frame, f"SAMPLES: {img_id}/200", (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            if face is not None:
                img_id += 1
                face_resized = cv2.resize(face, (200, 200)) # Keep original face for saving
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                file_name_path = f"{self.data_dir}/user.{user_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face_gray)
                
                # Draw tech effect on main frame
                # Reread coordinates for drawing (simplified for speed, just green box flash)
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5) 

            cv2.imshow("SYSTEM TERMINAL", frame)
            if cv2.waitKey(1) == 13 or img_id == 200: break

        cap.release()
        cv2.destroyAllWindows()
        self.users[str(user_id)] = user_name.upper()
        self.save_users()
        messagebox.showinfo("SYSTEM", f"USER {user_name.upper()} REGISTERED IN DATABASE.")

    def train_classifier(self):
        path = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        faces, ids = [], []

        if not path:
            messagebox.showwarning("ERROR", "DATABASE EMPTY. INITIATE DATA COLLECTION.")
            return

        try:
            for image_path in path:
                img = Image.open(image_path).convert('L')
                imageNp = np.array(img, 'uint8')
                id = int(os.path.split(image_path)[1].split(".")[1])
                faces.append(imageNp)
                ids.append(id)

            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces, np.array(ids))
            clf.write(self.classifier_file)
            messagebox.showinfo("SYSTEM", "NEURAL NETWORK UPDATED SUCCESSFULLY.")
        except Exception as e:
            messagebox.showerror("CRITICAL ERROR", str(e))

    def recognize_face(self):
        if not os.path.exists(self.classifier_file):
            messagebox.showerror("ERROR", "MODEL MISSING. EXECUTE TRAINING PROTOCOL.")
            return

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read(self.classifier_file)
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, img = video_capture.read()
            if not ret: break

            img = self.draw_hud(img, clf)
            cv2.imshow("OBSERVATION DECK", img)
            if cv2.waitKey(1) == 13: break

        video_capture.release()
        cv2.destroyAllWindows()

    def draw_hud(self, img, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Cyber Overlay Global
        cv2.putText(img, f"SYS::ONLINE  {time.strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        cv2.line(img, (20, 40), (200, 40), (0, 255, 255), 2)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w, x1), min(h, y1)
                
                face = gray_image[y:y1, x:x1]
                if face.size == 0: continue

                try:
                    id, pred = clf.predict(face)
                    conf_score = int(100 * (1 - pred / 300))

                    # Logic
                    if conf_score > 73:
                        name = self.users.get(str(id), f"ID:{id}")
                        color = (0, 255, 0) # Green
                        status = "ACCESS GRANTED"
                    else:
                        name = "UNKNOWN ENTITY"
                        color = (0, 0, 255) # Red
                        status = "ACCESS DENIED"

                    # Visualization
                    self.draw_tech_brackets(img, x, y, x1, y1, color)
                    
                    # Target Info
                    text_x = x + 5
                    cv2.putText(img, name, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(img, f"CONF: {conf_score}%", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                    cv2.putText(img, status, (x, y1 + 20), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

                    # Scanline effect (simple line moving down)
                    scan_y = int(y + (time.time() * 200) % (y1 - y))
                    cv2.line(img, (x, scan_y), (x1, scan_y), (255, 255, 255), 1)

                except Exception: continue
        return img

    def run_gui(self):
        root = tk.Tk()
        root.title("BIOMETRIC ACCESS CONTROL")
        root.geometry("450x400")
        root.configure(bg=self.theme_bg)

        # Header
        lbl_title = Label(root, text="// FACE RECOGNITION SYSTEM //", 
                         font=("Courier New", 16, "bold"), 
                         bg=self.theme_bg, fg=self.theme_fg)
        lbl_title.pack(pady=30)

        # Buttons
        btn_config = {
            "font": self.theme_font, 
            "bg": "#1a1a1a", 
            "fg": self.theme_accent,
            "activebackground": self.theme_accent,
            "activeforeground": "#000",
            "relief": "flat",
            "width": 30,
            "height": 2,
            "cursor": "hand2"
        }

        # Hover effects
        def on_enter(e): e.widget['bg'] = '#333'
        def on_leave(e): e.widget['bg'] = '#1a1a1a'

        b1 = Button(root, text="[1] INITIALIZE NEW SUBJECT", command=self.generate_dataset, **btn_config)
        b1.pack(pady=10)
        
        b2 = Button(root, text="[2] UPDATE NEURAL NET", command=self.train_classifier, **btn_config)
        b2.pack(pady=10)

        b3 = Button(root, text="[3] ENGAGE SURVEILLANCE", command=self.recognize_face, **btn_config)
        b3.pack(pady=10)

        for b in [b1, b2, b3]:
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)

        # Exit
        btn_exit = Button(root, text="TERMINATE SESSION", command=root.destroy, 
                         font=("Courier New", 10), bg="#330000", fg="#ff0000", relief="flat")
        btn_exit.pack(pady=30)

        # Status footer
        lbl_status = Label(root, text="SYSTEM STATUS: STANDBY...", 
                          font=("Courier New", 10), bg=self.theme_bg, fg="#666")
        lbl_status.pack(side="bottom", pady=10)

        root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionSystem()
    app.run_gui()