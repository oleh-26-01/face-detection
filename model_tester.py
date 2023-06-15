import cv2
import numpy as np
import pygame as pg
from tensorflow import keras

model = keras.models.load_model("face-recognition_s48_p1.2M.h5")
IMG_SIZE = model.input.shape[1]

max_size = (1920, 1080)
image_path = "my_recognition_data/image_n2.jpg"
original_image = cv2.imread(image_path)

# Scale image to fit the screen
if original_image.shape[0] > max_size[1] or original_image.shape[1] > max_size[0]:
    scale = min(max_size[0] / original_image.shape[1], max_size[1] / original_image.shape[0])
    original_image = cv2.resize(original_image, (0, 0), fx=scale, fy=scale)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.resize(image, (original_image.shape[1], original_image.shape[0]))

pg.init()
screen = pg.display.set_mode((original_image.shape[1], original_image.shape[0]))
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image = np.transpose(original_image, (1, 0, 2))
screen.blit(pg.surfarray.make_surface(original_image), (0, 0))
pg.display.set_caption("Face detection")
clock = pg.time.Clock()

font = pg.font.SysFont("Arial", 18)

running = True


class Window:
    def __init__(self):
        self.size = 40
        self.screen = screen
        self.pos = pg.Vector2(0, 0)

    def update(self, position):
        # Draw rectangle
        self.pos = pg.Vector2(position[0], position[1])
        pg.draw.rect(self.screen, (255, 0, 0),
                     (self.pos.x - self.size / 2, self.pos.y - self.size / 2, self.size, self.size), 2)

    def predict(self):
        # Return prediction on window area
        cut = image[int(self.pos.y - self.size / 2):int(self.pos.y + self.size / 2),
              int(self.pos.x - self.size / 2):int(self.pos.x + self.size / 2)]
        if cut.shape[0] == 0 or cut.shape[1] == 0:
            return np.array([[0, 0, 0, 1]])
        cut = cv2.resize(cut, (IMG_SIZE, IMG_SIZE))
        cut = np.array(cut)
        if len(cut.shape) == 2:
            cut = np.stack((cut,) * 3, axis=-1)
        cut = cut / 255.0
        if cut.shape != (IMG_SIZE, IMG_SIZE, 3):
            return np.array([[0, 0, 0, 1]])
        pred = model.predict(np.array([cut]), verbose=0)
        return pred


print("Press [W/S] or [UP/DOWN] to change window size")
print("Scroll mouse wheel to change window size")
print("Press ESC to exit")

window = Window()
size_change = 0
wheel_scroll = False
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False

            if event.key == pg.K_UP or event.key == pg.K_s:
                size_change = 1
            if event.key == pg.K_DOWN or event.key == pg.K_w:
                size_change = -1

        if event.type == pg.KEYUP:
            if event.key == pg.K_UP or event.key == pg.K_DOWN or event.key == pg.K_w or event.key == pg.K_s:
                size_change = 0

        # Check for mouse wheel scroll
        if event.type == pg.MOUSEWHEEL:
            wheel_scroll = True
            if event.y > 0:
                size_change = 1
            if event.y < 0:
                size_change = -1

    if size_change != 0:
        window.size += size_change * 10
        if window.size < 1:
            window.size = 1
        if window.size > min(original_image.shape[0], original_image.shape[1]):
            window.size = min(original_image.shape[0], original_image.shape[1])
        if wheel_scroll:
            wheel_scroll = False
            size_change = 0

    screen.fill((0, 0, 0))
    screen.blit(pg.surfarray.make_surface(original_image), (0, 0))
    mouse_pos = pg.mouse.get_pos()
    window.update(mouse_pos)
    prediction = window.predict()
    classes = ["away_from_face", "face", "close_to_face", "no_face"]

    # Draw rectangle background for text
    pg.draw.rect(screen, (0, 0, 0), (0, 0, 200, 25 * len(classes)))
    for i in range(len(classes)):
        text = font.render(classes[i] + ": " + str(round(prediction[0][i], 2)), True, (255, 255, 255))
        screen.blit(text, (10, 10 + 20 * i))

    pg.display.flip()
    clock.tick(60)