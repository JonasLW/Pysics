import pygame as pg
import os
import numpy as np
import time

# Note to self: Position, velocity, acceleration should be integer vectors.
# However, calculations in between (such as normal vectors) should use floats.
# Calculating collision with entire terrain may be inefficient. Consider tiling
# Find a way to scale up size of pixels
# Consider using pygame vectors instead of numpy
# Figure out how to implement a camera
#----ASSETS-------
TEST_SPRITE = pg.image.load(os.path.join('Assets', 'TestSprite.png'))
TERRAIN = pg.image.load(os.path.join('Assets', 'Terrain.png'))
try:
    BACKGROUND = pg.image.load(os.path.join('Assets', 'Background.png'))
except:
    BACKGROUND = pg.image.load(os.path.join('Assets', 'Terrain.png'))
try:
    FOREGROUND = pg.image.load(os.path.join('Assets', 'Foreground.png'))
except:
    FOREGROUND = pg.image.load(os.path.join('Assets', 'Terrain.png'))
#TEST_SPRITE = pg.transform.rotate(TEST_SPRITE, 50)
#----WINDOW SETUP----
WIDTH, HEIGHT = TERRAIN.get_width(), TERRAIN.get_height()
WIN = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Test game")
#----CONSTANTS----
FPS = 60
VEL = 5
GRAVITY = 0.6
AIR_RESISTANCE = 0.005
INPUT_FORCE = 1
JUMP_FORCE = 20
#-----------------

class PhysicsSprite(pg.sprite.Sprite):
    def __init__(self, image, left, top):
        pg.sprite.Sprite.__init__(self)
        self.image = image
        self.mask = pg.mask.from_surface(self.image)
        self.rect = self.mask.get_rect(left=left, top=top)
        self.COM = np.array(self.mask.centroid())
        #self.density = density
        self.m = 1
        self.s = np.array([left,top])
        self.v = np.array([0, 0], dtype=float)
        self.I = 1
        self.angle = 0
        self.w = 0
        self.rebound = 0.5
        self.collision = None
        self.overlap_mask = None
        self.overlap_surface = None

    def update_pos(self, world, force):
        a = force/self.m
        ds = (self.v + 0.5*a).astype(int)
        normal = np.array([0, 0], dtype=float)
        """
        if self.s[0] + ds[0] < 0:
            normal += np.array([1, 0])
        elif self.s[0] + ds[0] > WIDTH - self.rect.width:
            normal += np.array([-1, 0 ])
        if self.s[1] + ds[1] < 0:
            normal += np.array([0, 1])
        elif self.s[1] + ds[1] > HEIGHT - self.rect.height:
            normal += np.array([0, -1])
        normal = normalize_2v(normal)
        """
        self.s += ds
        normal = self.detect_collision(world)
        self.s -= ds

        normal_acc = normal.dot(a)*normal
        normal_vel = normal.dot(self.v)*normal
        a -= normal_acc
        a -= (1 + self.rebound)*normal_vel
        ds = (self.v + 0.5*a).astype(int)
        self.s += ds
        self.rect.topleft = self.s
        self.v += a

    def detect_collision(self, world):
        #self.collision = np.array(pg.sprite.collide_mask(self, world))
        normal = np.array([0,0], dtype=float)
        self.overlap_mask = self.mask.overlap_mask(world.mask, (-1)*self.s)
        self.overlap_surface = self.overlap_mask.to_surface(unsetcolor=(0,0,0,0))
        if self.overlap_mask.count() > 0:
            POI = np.array(self.overlap_mask.centroid())
            normal = normalize_2v(self.COM - POI)
        return normal


class Terrain(pg.sprite.Sprite):
    def __init__(self, background, terrain, foreground):
        pg.sprite.Sprite.__init__(self)
        self.background = background
        self.terrain = terrain
        self.foreground = foreground
        self.mask = pg.mask.from_surface(self.terrain)
        self.rect = self.mask.get_rect(left=0, top=0)


def abs_2v(vector):
    return np.sqrt(vector.dot(vector))


def normalize_2v(vector):
    mod_squared = vector.dot(vector)
    if mod_squared < 0.0000000001:
        return np.array([0, 0], dtype=float)
    else:
        return vector/np.sqrt(mod_squared)


def draw_window(world, player):
    WIN.blit(world.background, (0,0))
    WIN.blit(world.terrain, (0,0))
    WIN.blit(player.image, player.s)
    WIN.blit(world.foreground, (0,0))
    if player.overlap_mask.count() > 0:
        WIN.blit(player.overlap_surface, player.s)
        print(player.overlap_mask.count())
    else:
        print("---")
    pg.display.update()


def get_input_dir(keys_pressed):
    input_dir = np.array([0, 0], dtype=float)
    if keys_pressed[pg.K_d]:
        input_dir[0] += 1
    if keys_pressed[pg.K_a]:
        input_dir[0] -= 1
    if keys_pressed[pg.K_w]:
        input_dir[1] -= 1
    if keys_pressed[pg.K_s]:
        input_dir[1] += 1
    return normalize_2v(input_dir)


def main():
    world = Terrain(BACKGROUND, TERRAIN, FOREGROUND)
    player = PhysicsSprite(TEST_SPRITE, 100, 100)

    clock = pg.time.Clock()
    running = True
    while running:
        clock.tick(FPS)
        force = np.array([0 ,0], dtype=float)
        keys_pressed = pg.key.get_pressed()
        input_dir = get_input_dir(keys_pressed)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    force += JUMP_FORCE*normalize_2v(np.array([0, -1]) + input_dir)
        force += INPUT_FORCE*input_dir
        force[1] += GRAVITY
        force -= AIR_RESISTANCE*player.v
        player.update_pos(world, force)
        player.detect_collision(world)
        draw_window(world, player)
    pg.quit()


if __name__ == "__main__":
    main()
