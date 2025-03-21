import os
import cv2
import time

# Directory to store data
DATA_DIR = './gestures'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of classes
classes = ['Hi','Hello','Back','Beautiful','Bath','Call']  # Add more as needed
number_of_classes = len(classes)  # Number of classes to collect in one run

def collect_images(class_name, variation, dataset_size):
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        filename = f'{class_name}_v{variation + 1}_{counter}.jpg' if variation is not None else f'{class_name}_{counter}.jpg'
        cv2.imwrite(os.path.join(class_dir, filename), frame)
        counter += 1

# Open webcam
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, classes[j])
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {classes[j]}')

    # Option to skip the class
    skip_class = input(f"Do you want to skip the class '{classes[j]}'? (yes/no): ").strip().lower() == 'yes'
    if skip_class:
        print(f"Skipping class: {classes[j]}\n")
        continue

    # Ask whether to take two variations
    take_two_variations = input(f"Does the sign for '{classes[j]}' require two variations? (yes/no): ").strip().lower() == 'yes'
    total_images = 50

    # Loop for each variation
    for variation in range(2 if take_two_variations else 1):
        if take_two_variations:
            print(f"Collecting variation {variation + 1} for class: {classes[j]}")

        # Prompt user to start data collection
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" to start countdown!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) == ord('q'):
                break

        print('Starting in 5 seconds...')
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f'Starting in {i} seconds', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame", frame)
            cv2.waitKey(1000)  # Wait for 1 second

        # Collect images
        collect_images(classes[j], variation if take_two_variations else None, total_images)

cap.release()
cv2.destroyAllWindows()