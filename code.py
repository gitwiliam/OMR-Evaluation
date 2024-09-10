import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to capture image from webcam
def capture_omr_image():
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default camera)

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        cv2.imshow('OMR Sheet - Press Space to Capture', frame)  # Show the frame

        if cv2.waitKey(1) & 0xFF == ord(' '):  # If space is pressed, capture the image
            cv2.imwrite('omr_sheet.jpg', frame)  # Save the captured image
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all windows
    return frame


# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)  # Threshold the image
    return thresh


# Function to find and evaluate OMR bubbles
def evaluate_omr(image, correct_answers):
    # Detect contours (bubbles)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume each bubble is one of the contours (this can be customized)
    bubbles = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    # Initialize evaluation results
    marked_answers = []
    index = 0

    # Loop through detected bubbles
    for bubble in bubbles:
        x, y, w, h = cv2.boundingRect(bubble)
        aspect_ratio = w / float(h)
        
        # Filter out noise based on aspect ratio or size (only keep actual bubbles)
        if 0.9 <= aspect_ratio <= 1.1 and w > 20 and h > 20:
            # Crop the detected bubble
            bubble_roi = image[y:y + h, x:x + w]
            
            # Check if the bubble is filled (non-zero pixels mean it is marked)
            filled = cv2.countNonZero(bubble_roi)
            
            # Marked if filled area exceeds threshold
            if filled > 500:
                marked_answers.append(index)

            index += 1

    # Compare marked answers with correct answers
    score = sum(1 for a, b in zip(marked_answers, correct_answers) if a == b)
    total_questions = len(correct_answers)

    return score, total_questions


# Function to display results
def display_results(score, total):
    df = pd.DataFrame({
        'Metric': ['Total Questions', 'Correct Answers'],
        'Value': [total, score]
    })

    # Show the DataFrame
    print(df)

    # Plot the results using Matplotlib
    plt.bar(['Total Questions', 'Correct Answers'], [total, score], color=['blue', 'green'])
    plt.title('OMR Evaluation Results')
    plt.show()


# Main function
if __name__ == "__main__":
    # Step 1: Capture the OMR sheet image from webcam
    captured_image = capture_omr_image()

    # Step 2: Preprocess the image (grayscale, blur, threshold)
    processed_image = preprocess_image(captured_image)

    # Step 3: Define the correct answers for the OMR sheet
    # Example: Question 1 -> Option 2 (index starts from 0)
    correct_answers = [0, 1, 2, 1, 0]  # Modify this according to your OMR sheet layout

    # Step 4: Evaluate the OMR answers based on correct answers
    score, total_questions = evaluate_omr(processed_image, correct_answers)

    # Step 5: Display the results using Pandas and Matplotlib
    display_results(score, total_questions)
