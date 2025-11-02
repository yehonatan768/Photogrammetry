import cv2
cap = cv2.VideoCapture("D:\\Projects\\Photogrammetry\\videos\\DJI0004.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total / fps if fps else 0
print(f"Frames: {total}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
cap.release()
