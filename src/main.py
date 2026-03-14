import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

model = YOLO('yolov8n-pose.pt')
video_path = "data/input/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("HATA: Video dosyası açılamadı!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- 1. BÖLÜM: KALİBRASYON ---
pts_src = []

def draw_points(event, x, y, flags, param):
    global pts_src
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:
        pts_src.append([x, y])
        print(f"Nokta {len(pts_src)} eklendi!")

success, first_frame = cap.read()
if not success:
    exit()

cv2.namedWindow("Kalibrasyon")
cv2.setMouseCallback("Kalibrasyon", draw_points)

print("\n=== KALİBRASYON ===")
print("Lütfen sahadaki referans dikdörtgenin 4 köşesine sırayla tıklayın (Sol Üst, Sağ Üst, Sağ Alt, Sol Alt).")

while len(pts_src) < 4:
    temp_frame = first_frame.copy()
    for i, pt in enumerate(pts_src):
        cv2.circle(temp_frame, tuple(pt), 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(temp_frame, tuple(pts_src[i-1]), tuple(pts_src[i]), (0, 255, 0), 2)
            
    if len(pts_src) == 4:
        cv2.line(temp_frame, tuple(pts_src[3]), tuple(pts_src[0]), (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon", temp_frame)
        cv2.waitKey(500) 
        
    cv2.imshow("Kalibrasyon", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Kalibrasyon")
print("VAR sistemi başlatılıyor... Lütfen bekleyin.")

pts_src = np.array(pts_src, dtype=np.float32)
pts_dst = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("data/output/output_var.mp4", fourcc, fps, (frame_width, frame_height))
# --- 2. BÖLÜM: ANA DÖNGÜ ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 
        
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # --- A) TAKIM AYRIMI (K-MEANS) ---
    player_boxes = []
    player_colors = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                player_boxes.append((x1, y1, x2, y2))
                
                h = y2 - y1
                y_start = max(0, y1 + int(h * 0.1))
                y_end = min(frame.shape[0], y1 + int(h * 0.4))
                
                if y_end > y_start and x2 > x1:
                    torso = frame[y_start:y_end, x1:x2]
                    avg_color = cv2.mean(torso)[:3]
                    player_colors.append(avg_color)
                else:
                    player_colors.append((255, 255, 255))
                    
    if len(player_colors) > 2:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(player_colors)
        labels = kmeans.labels_
        
        for i, (x1, y1, x2, y2) in enumerate(player_boxes):
            if labels[i] == 0:
                color = (0, 0, 255) 
                team_name = "Takim A"
            else:
                color = (255, 0, 0) 
                team_name = "Takim B"
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, team_name, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- B) VAR OFSAYT ÇİZGİSİ (ESNEK NOKTA TESPİTİ) ---
    last_defender_x = None 
    last_defender_y = None
    
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        print(f"DEBUG: Toplam keypoints = {len(keypoints)}")
        
        for idx, person in enumerate(keypoints):
            if len(person) >= 17: 
                # Ayak bilekleri (15,16), Dizler (13,14), Kalçalar (11,12), Başlık (0)
                body_parts = [person[15], person[16], person[13], person[14], person[11], person[12], person[0]]
                
                # Sadece koordinatı olan (0,0 olmayan) noktaları al
                valid_points = np.array([pt for pt in body_parts if pt[0] > 0 and pt[1] > 0])
                
                if len(valid_points) > 0:
                    # Bu oyuncunun en alçak noktasını bul (futbolda top durabilecek pozisyon)
                    max_y_idx = np.argmax(valid_points[:, 1])
                    lowest_point = valid_points[max_y_idx]
                    
                    print(f"DEBUG: Oyuncu {idx} - Valid points = {len(valid_points)}, En alçak X = {int(lowest_point[0])}")
                    
                    # İlk bulunan oyuncuyu (veya daha sağdaki oyuncuyu) seç
                    if last_defender_x is None or lowest_point[0] > last_defender_x:
                        last_defender_x = int(lowest_point[0])
                        last_defender_y = int(lowest_point[1])
    else:
        print("DEBUG: Keypoints bulunamadı. yolov8n-pose.pt modeli kullanılıyor mu?")

    # ÇİZGİYİ ÇİZ
    if last_defender_x is not None and last_defender_x > 0:
        print(f"DEBUG: Çizgi çiziliyor - X = {last_defender_x}, Y = {last_defender_y}")
        foot_point = np.array([[[last_defender_x, last_defender_y]]], dtype=np.float32)
        bird_eye_point = cv2.perspectiveTransform(foot_point, M)
        bird_x = bird_eye_point[0][0][0]
        
        line_point_top = np.array([[[bird_x, -1000]]], dtype=np.float32)
        line_point_bottom = np.array([[[bird_x, 2000]]], dtype=np.float32)
        
        camera_point_top = cv2.perspectiveTransform(line_point_top, M_inv)
        camera_point_bottom = cv2.perspectiveTransform(line_point_bottom, M_inv)
        
        pt1 = (int(camera_point_top[0][0][0]), int(camera_point_top[0][0][1]))
        pt2 = (int(camera_point_bottom[0][0][0]), int(camera_point_bottom[0][0][1]))
        
        cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 3)
        cv2.putText(annotated_frame, "VAR HATTI", (pt1[0] - 50, max(50, pt1[1] + 50)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        print("DEBUG: Çizgi çizilmeyecek - last_defender_x = None")

    cv2.imshow("VAR Final Projesi", annotated_frame)
    out.write(annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()