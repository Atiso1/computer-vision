import pygame
import random
import time
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# MEDIAPIPE TASK SETUP
# ==============================

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ==============================
# GAME VARIABLES
# ==============================

SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600

SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 10

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT = 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

pygame.mixer.init()

# ==============================
# BIRD
# ==============================

class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]

        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):

        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):

        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


# ==============================
# PIPE
# ==============================

class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image,(PIPE_WIDHT, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


# ==============================
# GROUND
# ==============================

class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image,(GROUND_WIDHT, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# ==============================
# HELPER
# ==============================

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):

    size = random.randint(100,300)

    pipe = Pipe(False,xpos,size)
    pipe_inverted = Pipe(True,xpos,SCREEN_HEIGHT-size-PIPE_GAP)

    return pipe,pipe_inverted


# ==============================
# COUNT FINGERS
# ==============================

def count_fingers(landmarks):

    tips = [4,8,12,16,20]
    fingers = 0

    for i in range(1,5):
        if landmarks[tips[i]].y < landmarks[tips[i]-2].y:
            fingers += 1

    return fingers


# ==============================
# PYGAME INIT
# ==============================

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDHT,SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird Hand Control")

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND,(SCREEN_WIDHT,SCREEN_HEIGHT))

BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

ground_group = pygame.sprite.Group()

for i in range(2):
    ground = Ground(GROUND_WIDHT*i)
    ground_group.add(ground)

pipe_group = pygame.sprite.Group()

for i in range(2):

    pipes = get_random_pipes(SCREEN_WIDHT*i+800)

    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

clock = pygame.time.Clock()

begin = True

# ==============================
# BEGIN SCREEN
# ==============================

while begin:

    clock.tick(15)

    ret,frame = cap.read()

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    result = landmarker.detect(mp_image)

    finger_count = 0

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        finger_count = count_fingers(landmarks)

    if finger_count == 2:

        bird.bump()
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()
        begin = False

    cv2.imshow("Hand Control",frame)
    cv2.waitKey(1)

    screen.blit(BACKGROUND,(0,0))
    screen.blit(BEGIN_IMAGE,(120,150))

    bird.begin()
    ground_group.update()

    bird_group.draw(screen)
    ground_group.draw(screen)

    pygame.display.update()


# ==============================
# GAME LOOP
# ==============================

while True:

    clock.tick(30)

    ret,frame = cap.read()

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    result = landmarker.detect(mp_image)

    finger_count = 0

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        finger_count = count_fingers(landmarks)

    if finger_count == 2:

        bird.bump()
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()

    cv2.imshow("Hand Control",frame)
    cv2.waitKey(1)

    screen.blit(BACKGROUND,(0,0))

    if is_off_screen(ground_group.sprites()[0]):

        ground_group.remove(ground_group.sprites()[0])

        new_ground = Ground(GROUND_WIDHT-20)

        ground_group.add(new_ground)

    if is_off_screen(pipe_group.sprites()[0]):

        pipe_group.remove(pipe_group.sprites()[0])
        pipe_group.remove(pipe_group.sprites()[0])

        pipes = get_random_pipes(SCREEN_WIDHT*2)

        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])

    bird_group.update()
    ground_group.update()
    pipe_group.update()

    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    pygame.display.update()

    if (pygame.sprite.groupcollide(bird_group,ground_group,False,False,pygame.sprite.collide_mask) or
        pygame.sprite.groupcollide(bird_group,pipe_group,False,False,pygame.sprite.collide_mask)):

        pygame.mixer.music.load(hit)
        pygame.mixer.music.play()
        time.sleep(1)
        break