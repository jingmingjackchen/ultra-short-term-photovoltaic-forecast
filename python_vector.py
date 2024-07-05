import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture("vid_h264.mp4")

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize a list to store the flow vectors
flow_vectors = []

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Store the flow vectors
    flow_vectors.append(flow)

    # Update previous frame
    prev_gray = gray

    # Display the result (optional)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = np.zeros_like(frame)
    mask[..., 1] = 255  # Set saturation to maximum
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    cv2.imshow("Dense Optical Flow", rgb)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save the flow vectors to a numpy file
np.savez_compressed("optical_flow_vectors.npz", flow_vectors)

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
