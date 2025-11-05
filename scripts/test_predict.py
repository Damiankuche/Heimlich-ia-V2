import base64, cv2, requests
from pathlib import Path

IMG = r"E:\PythonProject\HeimlichIA\test1 (4).jpg"  # <-- poné una imagen real
url = "http://127.0.0.1:8000/predictOne"

img = cv2.imread(IMG)
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
b64 = base64.b64encode(buf).decode("ascii")

resp = requests.post(url, json={"image_b64": b64}, timeout=30)
print(resp.status_code, resp.json())
