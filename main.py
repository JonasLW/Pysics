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
# Implement objects with variable density?
# Implement friction
# Find out why gravity is delayed
# Note: When object is clipping, it is rapidly bouncing back and forth.
#    Do something when force is not opposite normal vector
# Test alternate methods for calculating normal vector (circle method)
# Handle multiple points of impact
# Note: Rotation makes FPS drop to 50 for larger objects
#----ASSETS-------
TEST_SPRITE = pg.image.load(os.path.join('Assets', 'TestSprite3.png'))
TERRAIN = pg.image.load(os.path.join('Assets', 'Terrain3.png'))
try:
    BACKGROUND = pg.image.load(os.path.join('Assets', 'Background3.png'))
except:
    BACKGROUND = TERRAIN
try:
    FOREGROUND = pg.image.load(os.path.join('Assets', 'Foreground3.png'))
except:
    FOREGROUND = TERRAIN
#---------
#TEST_SPRITE = pg.transform.scale(TEST_SPRITE, (125, 108))
#----WINDOW SETUP----
WIDTH, HEIGHT = TERRAIN.get_width(), TERRAIN.get_height()
WIN = pg.display.set_mode((WIDTH, HEIGHT))
WIN.blit(BACKGROUND, (0,0))
WIN.blit(TERRAIN, (0,0))
WIN.blit(FOREGROUND, (0,0))
pg.display.set_caption("Test game")
"""
# This does not seem to help?...
TERRAIN.convert()
BACKGROUND.convert()
FOREGROUND.convert()
TEST_SPRITE.convert()
"""
#----CONSTANTS----
# Higher FPS gives better simulation, probably
# Consider making forces and such independent of FPS
FPS = 60
DT = 1/FPS
G = 0.5
GRAVITY = np.array([0, G])*DT*60
AIR_RESISTANCE = 0.0005*DT*60
INPUT_FORCE = 1*DT*60
JUMP_FORCE = 100*DT*60
STAT_THRESHOLD = 1  # Threshold speed for considering object static
#-----------------

class PhysicsSprite(pg.sprite.Sprite):
    def __init__(self, image, density, left, top):
        pg.sprite.Sprite.__init__(self)
        # surfaces and masks
        self.image = image
        self.boundary_image = get_outline(image)
        self.rot_image = image
        self.rot_boundary_image = self.boundary_image
        self.mask = pg.mask.from_surface(self.image)
        self.boundary_mask = pg.mask.from_surface(self.boundary_image)
        self.rot_mask = self.mask
        self.rot_boundary_mask = self.boundary_mask
        # related variables
        self.rect = self.mask.get_rect(left=left, top=top)
        self.prev_rect = self.mask.get_rect(left=left, top=top)
        self.COM = np.array([left, top], dtype=float) + np.array(self.mask.centroid(), dtype=float)
        self.ctc_vec = np.array(self.rect.center) - self.COM.astype(int)
        # kinetic variables
        self.density = density
        self.m, self.I = get_inertia(self.mask, self.density)
        self.s = np.array([left,top])
        self.v = np.array([0, 0], dtype=float)
        self.angle = 0
        self.ang_vel = 0
        self.rebound = 0.5
        self.friction_s = 0.1
        self.friction_d = 0.0008  # Uncertain what are appropriate ranges for these
        self.collision = None
        self.body_overlap = None
        self.overlap_surface = None

    def update(self, world, force):
        #start = pg.time.get_ticks()
        self.prev_rect = self.rect
        acc = force/self.m
        start_vel = self.v
        self.COM += start_vel + 0.5*acc
        self.angle += self.ang_vel    # mod 360?
        rot_ctc_vec = rotate_2v(self.ctc_vec, -self.angle)
        center_pos = self.COM + np.array(rot_ctc_vec)
        self.rot_image = pg.transform.rotate(self.image, self.angle)
        self.rot_boundary_image = pg.transform.rotate(self.boundary_image, self.angle)
        self.rect = self.rot_image.get_rect(center=center_pos)
        self.v += acc
        self.s = np.array(self.rect.topleft)
        self.rot_mask = pg.mask.from_surface(self.rot_image)
        self.rot_boundary_mask = pg.mask.from_surface(self.rot_boundary_image)
        # Detect and resolve collision.
        normal, POI, depth = self.detect_collision(world)
        n_tangent = rotate_2v(normal, -90)
        if not (normal[0] == 0 and normal[1] == 0):
            r = POI - self.COM
            r_tangent = rotate_2v(r, -90)
            r_para_sq = r.dot(r) - (r.dot(normal))*(r.dot(normal))  # comp. of r perp to normal
            POI_vel = self.v + r_tangent*self.ang_vel*np.pi/180
            POI_normal_spd = abs(POI_vel.dot(normal))  # Not sure if abs is necessary. If this is positive, something is wrong
            POI_tangent_spd = POI_vel.dot(n_tangent)
            # Find TOI
            t_rwnd = depth/POI_normal_spd
            TOI = 1 - t_rwnd
            if 0 < TOI < 0.9:
                # Move back to start of frame, then forward to TOI
                # Ideally, would re-calculate POI and normal at TOI
                self.COM -= start_vel + 0.5*acc
                self.angle -= self.ang_vel
                self.COM += start_vel*TOI + 0.5*acc*TOI*TOI
                self.angle += self.ang_vel*TOI
            # Identify stationary contact here. Apply "stopping impulse"
            # Apply the impulse
            e = self.rebound*world.rebound
            impulse = -(1+e)*POI_vel.dot(normal)/(1/self.m + r_para_sq/self.I)
            self.v += normal*impulse/self.m
            self.ang_vel -= cross_2v(r, normal)*impulse*180/(np.pi*self.I)
            # Apply friction
            #    trouble because normal vector isn't quite right. Makes friction messy
            tangent_force = force.dot(n_tangent)
            if abs(POI_tangent_spd) >= STAT_THRESHOLD or abs(tangent_force) > self.friction_s*impulse:
                friction_impulse = -np.sign(POI_tangent_spd)*self.friction_d*impulse
                stopping_impulse = -POI_tangent_spd/(1/self.m + r.dot(normal)*r.dot(normal)/self.I)
                if abs(friction_impulse) > abs(stopping_impulse):
                    friction_impulse = stopping_impulse
            else:
                friction_impulse = -tangent_force
            self.v += n_tangent*friction_impulse/self.m
            self.ang_vel -= cross_2v(r, n_tangent)*friction_impulse*180/(np.pi*self.I) # Not sure what is right sign here
            if 0 < TOI < 0.9:
                self.COM += self.v*t_rwnd + 0.5*acc*t_rwnd*t_rwnd
                self.angle += self.ang_vel*t_rwnd
                self.rot_image = pg.transform.rotate(self.image, self.angle)
                self.rot_boundary_image = pg.transform.rotate(self.boundary_image, self.angle)
                self.rect = self.rot_image.get_rect(center=center_pos)
            if TOI < 0 or TOI > 1:
                # Dirty hack to avoid clipping. Does not preserve energy
                if depth < 20:
                    self.COM += depth*normal
                else:
                    pass
        else:
            pass
        # Hacky air resistance against rotation
        self.ang_vel -= AIR_RESISTANCE*(self.rect.width/50)*(self.rect.height/50)*self.ang_vel
        #end = pg.time.get_ticks()
        #print(f"Player update time: {end-start:.2f} ms")

    def detect_collision(self, world):
        #self.collision = np.array(pg.sprite.collide_mask(self, world))
        #start = pg.time.get_ticks()
        normal = np.array([0,0], dtype=float)
        POI = np.array([0,0], dtype=float)
        depth = 0
        self.body_overlap = self.rot_mask.overlap_mask(world.mask, (-1)*self.s)
        boundary_overlap_1 = self.rot_boundary_mask.overlap_mask(world.mask, (-1)*self.s)
        boundary_overlap_2 = self.rot_mask.overlap_mask(world.boundary_mask, (-1)*self.s)
        self.overlap_surface = boundary_overlap_1.to_surface(unsetcolor=(0,0,0,0)) # For visualizing impact
        if boundary_overlap_1.count() > 0:
            overlap_center = np.array(self.body_overlap.centroid())
            boundary_1_center = np.array(boundary_overlap_1.centroid())
            boundary_2_center = np.array(boundary_overlap_2.centroid())
            if boundary_1_center[0] == boundary_2_center[0] and boundary_1_center[1] == boundary_2_center[1]:
                print("dang")
            displacement = boundary_2_center - boundary_1_center
            normal = normalize_2v(displacement)
            depth = abs_2v(displacement)
            POI = self.s + boundary_1_center
            if boundary_2_center[0] == 0 and boundary_2_center[1] == 0:
                print("Fuck")
                print(f"Center 2: {boundary_2_center}")
                normal = np.array([0,0], dtype=float)
                POI = np.array([0,0], dtype=float)
                depth = 0
        #end = pg.time.get_ticks()
        #print(f"Collision detection time: {end-start:.2f} ms")
        return normal, POI, depth


class Terrain(pg.sprite.Sprite):
    def __init__(self, background, terrain, foreground):
        pg.sprite.Sprite.__init__(self)
        self.background = background
        self.terrain = terrain
        self.foreground = foreground
        self.boundary = get_outline(terrain)
        self.boundary_mask = pg.mask.from_surface(self.boundary)
        self.mask = pg.mask.from_surface(self.terrain)
        self.rect = self.mask.get_rect(left=0, top=0)
        self.rebound = 1


def get_boundary(image):
    mask = pg.mask.from_surface(image)
    inner_mask = mask.overlap_mask(mask, offset=(1,0))
    inner_mask = inner_mask.overlap_mask(mask, offset=(-1,0))
    inner_mask = inner_mask.overlap_mask(mask, offset=(0,1))
    inner_mask = inner_mask.overlap_mask(mask, offset=(0,-1))
    mask.erase(inner_mask, (0,0))
    image_boundary = mask.to_surface(setcolor=(0,0,0,255), unsetcolor=(0,0,0,0))
    return image_boundary


def get_outline(image):
    # Make a padded mask, and erase original mask.
    # causes trouble since outline doesn't fit in original mask's rect.
    # Must find a way to deal with that before this can be implemented.
    # Image must have a transparent pixels along the border for this to work
    inner_mask = pg.mask.from_surface(image)
    outer_mask = pg.mask.from_surface(image)
    outer_mask_2 = pg.mask.from_surface(image)
    outer_mask.draw(inner_mask, (1,0))
    outer_mask.draw(inner_mask, (-1,0))
    outer_mask.draw(inner_mask, (0,1))
    outer_mask.draw(inner_mask, (0,-1))
    # Experiment to make outline further out:
    """
    outer_mask_2.draw(outer_mask, (2,0))
    outer_mask_2.draw(outer_mask, (-2,0))
    outer_mask_2.draw(outer_mask, (0,2))
    outer_mask_2.draw(outer_mask, (0,-2))
    outer_mask_2.erase(outer_mask, (0,0))
    """
    # Result: Object visibly floating. Quite jittery
    # --- end experiment
    outer_mask.erase(inner_mask, (0,0))
    outline = outer_mask.to_surface(setcolor=(0,0,0,255), unsetcolor=(0,0,0,0))
    return outline


def get_inertia(mask, density):
    width, height = mask.get_size()
    COM = np.array(mask.centroid())
    I = 0
    m = 0
    for y in range(height):
        for x in range(width):
            r = np.array([x, y]) - COM
            r_sq = r.dot(r)
            dm = density*mask.get_at((x, y))
            m += dm
            I += dm*r_sq
    return (m, I)


def abs_2v(vector):
    return np.sqrt(vector.dot(vector))


def rotate_2v(vector, angle_deg):
    angle = angle_deg*np.pi/180
    x = np.cos(angle)*vector[0] - np.sin(angle)*vector[1]
    y = np.sin(angle)*vector[0] + np.cos(angle)*vector[1]
    return np.array([x, y])


def normalize_2v(vector):
    mod_squared = vector.dot(vector)
    if mod_squared < 0.0000000001:
        return np.array([0, 0], dtype=float)
    else:
        return vector/np.sqrt(mod_squared)


def cross_2v(a, b):
    return a[0]*b[1] - a[1]*b[0]


def draw_window(world, player):
    #start = pg.time.get_ticks()
    WIN.blit(world.background, player.prev_rect, area=player.prev_rect)
    WIN.blit(world.terrain, player.prev_rect, area=player.prev_rect)
    WIN.blit(player.rot_image, player.rect)
    WIN.blit(world.foreground, player.prev_rect, area=player.prev_rect)
    if player.body_overlap.count() > 0:
        WIN.blit(player.overlap_surface, player.rect, area=player.rect)
    pg.display.update()
    #end = pg.time.get_ticks()
    #print(f"Draw time: {end-start:.2f} ms")


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
    player = PhysicsSprite(TEST_SPRITE, 0.01, 400, 100)

    clock = pg.time.Clock()
    running = True
    while running:
        clock.tick(FPS)
        #print(f"MSPF: {clock.get_time()}, FPS: {clock.get_fps():.2f}")
        force = np.array([0 ,0], dtype=float)
        keys_pressed = pg.key.get_pressed()
        input_dir = get_input_dir(keys_pressed)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    force += JUMP_FORCE*normalize_2v(np.array([0, -1]) + input_dir)
                if event.key == pg.K_q:
                    player.ang_vel += 1
                if event.key == pg.K_e:
                    player.ang_vel -= 1
        force += INPUT_FORCE*input_dir*player.m
        force += GRAVITY*player.m
        force -= AIR_RESISTANCE*(player.rect.width/10)*(player.rect.height/10)*player.v
        player.update(world, force)
        draw_window(world, player)
    pg.quit()


if __name__ == "__main__":
    main()
