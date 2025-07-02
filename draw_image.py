import turtle
import cv2
import numpy as np
import time
import random
from sklearn.cluster import KMeans

# === Load dan Resize Gambar ===
img_path = "flower.jpeg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
max_height = 700
scale = max_height / img.shape[0]
new_width = int(img.shape[1] * scale)
new_height = int(img.shape[0] * scale)
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# === Outline Detection ===
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150)

# === Segmentasi Warna untuk Background dan Objek ===
pixel_data = img.reshape((-1, 3))
k = 5
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(pixel_data)
labels = kmeans.labels_.reshape((new_height, new_width))

# === Setup Turtle ===
pixel_size = 4
start_x = -new_width // 2
start_y = new_height // 2

screen = turtle.Screen()
screen.setup(width=new_width + 100, height=new_height + 100)
screen.title("Progressive Outline and Segment Fill")
screen.bgcolor("white")
screen.colormode(1.0)

t = turtle.Turtle()
t.hideturtle()
t.speed(0)
t.penup()
screen.tracer(0)

# === Buat Outline Path ===
t.pensize(1)
t.color("black")

outline_lines = []
visited = np.zeros_like(edges, dtype=bool)

for y in range(1, new_height - 1):
    for x in range(1, new_width - 1):
        if edges[y, x] > 0 and not visited[y, x]:
            path = [(x, y)]
            visited[y, x] = True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if edges[ny, nx] > 0 and not visited[ny, nx]:
                    path.append((nx, ny))
                    visited[ny, nx] = True
            if len(path) > 1:
                outline_lines.append(path)

random.shuffle(outline_lines)

# === Progressive Outline Gambar Perlahan ===
start_time = time.time()
outline_batch = 150  # jumlah garis per frame

for i, path in enumerate(outline_lines):
    t.penup()
    t.goto(start_x + path[0][0], start_y - path[0][1])
    t.pendown()
    for (x, y) in path:
        t.goto(start_x + x, start_y - y)
    t.penup()
    if i % outline_batch == 0:
        screen.update()

screen.update()

# === Segment Area ===
segment_coords = {i: [] for i in range(k)}
for y in range(0, new_height, pixel_size):
    for x in range(0, new_width, pixel_size):
        label = labels[y, x]
        segment_coords[label].append((x, y))

center_label = labels[new_height // 2, new_width // 2]
object_segments = [label for label in range(k) if label != center_label]
random.shuffle(segment_coords[center_label])

# === Urutan Gambar ===
draw_sequence = []
for label in object_segments:
    coords = segment_coords[label]
    random.shuffle(coords)
    draw_sequence.append((f"segment_{label}", coords))
draw_sequence.append(("background", segment_coords[center_label]))

# === Gambar Segmen  ===
start_time = time.time()
total_points = sum(len(coords) for _, coords in draw_sequence)
points_drawn = 0

for label_name, coords in draw_sequence:
    print(f"✏️ Filling {label_name} with {len(coords)} elements")
    for i, (x, y) in enumerate(coords):
        r, g, b = img[y, x]
        t.color(r / 255, g / 255, b / 255)
        t.goto(start_x + x, start_y - y)
        t.begin_fill()
        for _ in range(4):
            t.forward(pixel_size)
            t.right(90)
        t.end_fill()
        points_drawn += 1
        if points_drawn % 300 == 0:
            screen.update()
    screen.update()

end_time = time.time()
print(f"\n✅ Done in {end_time - start_time:.2f} seconds.")

turtle.done()