from ultralytics import YOLO
import cv2
import pytesseract
import os
import sqlite3
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to create the table if it doesn't exist
def create_table():
    # Connect to the SQLite database
    conn = sqlite3.connect("vehicle_data.db")
    # Create a cursor object to execute SQL statements
    c = conn.cursor()
    # Create the "vehicles" table with two columns: "vehicle_number" and "bike_image_path"
    # The "IF NOT EXISTS" clause ensures that the table is only created if it doesn't already exist
    c.execute("CREATE TABLE IF NOT EXISTS vehicles (vehicle_number TEXT, bike_image_path TEXT)")
    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()


# Function to insert a record into the "vehicles" table
def insert_record(vehicle_number, bike_image_path):
    # Connect to the SQLite database
    conn = sqlite3.connect("vehicle_data.db")
    # Create a cursor object to execute SQL statements
    c = conn.cursor()
    # Insert the record into the "vehicles" table using parameterized SQL statement
    c.execute("INSERT INTO vehicles VALUES (?, ?)", (vehicle_number, bike_image_path))
    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()


# Initialize YOLO models and I put the result of all  trained models
person_bike_model = YOLO(
    r"C:\helmet detection and number plate extraction\main_model\person_on_bike_result\best.pt")
helmet_model = YOLO(
    r"C:\helmet detection and number plate extraction\main_model\helmet_or_helmetless_result\best.pt")
number_plate_model = YOLO(
    r"C:\helmet detection and number plate extraction\main_model\number_plate_recognition_result\best.pt")

# Setting up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"  # Update with the path
# to your Tesseract OCR executable
frame_count = 0

output_dir = r"C:\helmet detection and number plate extraction\main_model\result\out"  # Directory to save the output images

# if out not present in result folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
# Set up video capture
video_capture = cv2.VideoCapture('Inputvideo2.mp4')  # Use 0 for the default camera or provide the desired camera index

# Get video properties from the input video
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Initialize video writer and used for output.avi video
# also save it to the result folder
VIDEO = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = os.path.join(output_dir, 'output_video.avi')
video_writer = cv2.VideoWriter(output_video_path, VIDEO, fps, (width, height))


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray


while video_capture.isOpened():
    frame_count += 1
    logging.info(f"Frame {frame_count}")

    ret, frame = video_capture.read()          #frame------by------frame read
    if not ret:
        break

    # Process frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detecting person on a bike
    person_bike_results = person_bike_model.predict(img, conf=0.5)

    # Process each detection result
    for r in person_bike_results:
        boxes = r.boxes
        # Filter detections for person on a bike
        for box in boxes:
            cls = box.cls         # Get the class of the detected object
            # step1--->if person on bike
            if person_bike_model.names[int(cls)] == "Person_Bike":
                # Crop person on a bike image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_bike_image = frame[y1:y2, x1:x2]
                # Detect helmet on the person
                helmet_results = helmet_model.predict(person_bike_image, conf=0.5)
                # Process each helmet detection result
                for hr in helmet_results:
                    h_boxes = hr.boxes        # Get the detected bounding boxes for helmets
                    # Filter detections for no helmet
                    for h_bo in h_boxes:
                        h_cls = h_bo.cls
                        # step--->2 if person is without helmet
                        if helmet_model.names[int(h_cls)] != "With Helmet":
                            # Extract number plate from the person bike image
                            number_plate_results = number_plate_model.predict(person_bike_image, conf=0.5)
                            # Process each number plate detection result
                            for npr in number_plate_results:
                                np_boxes = npr.boxes
                                # Filter detections for number plate
                                for np_box in np_boxes:
                                    np_cls = np_box.cls

                                    # step3----->if licence plate detected
                                    if number_plate_model.names[int(np_cls)] == "License_Plate":
                                        # Crop number plate image
                                        np_x1, np_y1, np_x2, np_y2 = map(int, np_box.xyxy[0])
                                        number_plate_image = person_bike_image[np_y1:np_y2, np_x1:np_x2]
                                        # Perform OCR on the number plate image
                                        preprocessed_image = preprocess_image(number_plate_image)
                                        text = pytesseract.image_to_string(preprocessed_image).strip()

                                        if text and len(text) > 6 and text[0] == 'U':
                                            # Save the cropped number plate image
                                            output_file = f"person_violation_{frame_count}_{x1}.jpg"
                                            output_path = os.path.join(output_dir, output_file)
                                            cv2.imwrite(output_path, person_bike_image)
                                            # Create the "vehicles" table if it doesn't exist
                                            create_table()
                                            insert_record(text, output_path)
                                            # Print the extracted text
                                            logging.info(f"Number Plate Text: {text}")
                                            # Draw bounding boxes and labels on the frame
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                        (36, 255, 12), 2)

    video_writer.write(frame)
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
