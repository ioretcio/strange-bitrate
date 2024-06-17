import cv2
import numpy as np
import sys

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    refresh = fps * 1
    grid = (151, 151)
    contour_size_thresh = 60

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, background = cap.read()
    frame_alpha = background
    ret, frame_beta = cap.read()
    frame_num = 0

    output_path = video_path.rsplit('.', 1)[0] + "_optimized_.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 24.0, (width, height))

    gray1 = cv2.cvtColor(frame_alpha, cv2.COLOR_BGR2GRAY)
    gray1_resized = cv2.resize(gray1, grid)

    gray2 = cv2.cvtColor(frame_beta, cv2.COLOR_BGR2GRAY)
    gray2_resized = cv2.resize(gray2, grid)

    history = []

    while True:
        frame_diff = cv2.absdiff(gray1_resized, gray2_resized)
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        motion_mask_restored = cv2.resize(motion_mask, (frame_alpha.shape[1], frame_alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(motion_mask_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_mask_restored = np.zeros(motion_mask_restored.shape, dtype=np.uint8)

        for contour in contours:
            if cv2.contourArea(contour) > contour_size_thresh:
                history.append([contour, fps])

        for i in range(len(history)):
            history[i][1] -= 1
            contour, lifetime = history[i]

            if len(contour) > 4 and lifetime > 0:
                points = np.array(contour)
                ellipse = cv2.minAreaRect(points)
                center_x, center_y = ellipse[0]
                angle = ellipse[2]
                scaled_width = ellipse[1][0] * 3
                scaled_height = ellipse[1][1] * 3
                scaled_ellipse = ((center_x, center_y), (scaled_width, scaled_height), angle)
                cv2.ellipse(motion_mask_restored, scaled_ellipse, (255, 255, 255), -1)

        history = [item for item in history if item[1] >= 0]

        cv2.bitwise_and(background, background, mask=cv2.bitwise_not(motion_mask_restored))
        background[motion_mask_restored != 0] = frame_beta[motion_mask_restored != 0]

        output_video.write(background)
        frame_alpha = frame_beta
        gray1_resized = gray2_resized
        ret, frame_beta = cap.read()

        if not ret:
            break

        gray2 = cv2.cvtColor(frame_beta, cv2.COLOR_BGR2GRAY)
        gray2_resized = cv2.resize(gray2, grid)

        if frame_num == refresh:
            background = frame_beta
            background = cv2.GaussianBlur(background, (11, 11), 0)
            frame_num = 0
        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python gpt_approach.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    process_video(video_path)

if __name__ == "__main__":
    main()