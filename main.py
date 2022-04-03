import pygame as pg
import os
import numpy as np
from scipy import signal
import time

# Note: Rotation makes FPS drop to 50 for larger objects
# Calculating collision with entire terrain may be inefficient. Consider tiling
# Find a way to scale up size of pixels
# Consider using pygame vectors instead of numpy
# Figure out how to implement a camera
# Note: When object is clipping, it is rapidly bouncing back and forth.
#    Do something when force is not opposite normal vector
# Test alternate methods for calculating normal vector (circle method)
#    Normal map seems to work very well
# Implement objects with variable density?
# Implement multiple objects
#    Use rect collision before checking sprite collision
# Implement ability to draw sprites with multiple moving parts?
#    How, then, can I also have sprites with multiple static parts?
# Handle multiple points of impact
# Handle static contact ("sleeping")
#----ASSETS-------
STAGE = 'Level-Test2'
SOBEL_TEST = pg.image.load(os.path.join('Assets', 'SobelTest2.png'))
TEST_SPRITE = pg.image.load(os.path.join('Assets', 'TestSprite6.png'))
TERRAIN = pg.image.load(os.path.join('Assets', STAGE+'.png'))
try:
    BACKGROUND = pg.image.load(os.path.join('Assets', STAGE+'-bg.png'))
except:
    BACKGROUND = TERRAIN
try:
    FOREGROUND = pg.image.load(os.path.join('Assets', STAGE+'fg.png'))
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
INPUT_FORCE = 3*DT*60
JUMP_FORCE = 20*DT*60
MAX_SPEED = 8
STAT_THRESHOLD = 1  # Threshold speed for considering object static
PI = np.pi
#-----------------

class PhysicsSprite(pg.sprite.Sprite):
    def __init__(self, image, density, left, top):
        pg.sprite.Sprite.__init__(self)
        # surfaces and masks
        self.image = image
        self.boundary_image = get_boundary(image)
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
        self.radius = 0.5*np.sqrt(self.rect.width**2 + self.rect.height**2) \
                      + abs_2v(self.ctc_vec)
        # kinetic variables
        self.density = density
        self.m, self.I = get_inertia(self.mask, self.density)
        self.s = np.array([left,top])
        self.v = np.array([0, 0], dtype=float)
        self.angle = 0
        self.ang_vel = 0
        self.rebound = 0.2
        self.friction_s = 0.95
        self.friction_d = 0.9  # Uncertain what are appropriate ranges for these
        self.collision = None
        self.body_overlap = None
        self.overlap_surface = None
        self.stationary = False
        self.stationary_counter = 0
        self.contact_point = None
        self.contact_vector = None

    def get_point_velocity(self, point):
        r = point - self.COM
        r_tangent = rotate_2v(r, -0.5*PI)
        return self.v + r_tangent*self.ang_vel

    def update(self, world, force):
        # Current time: 2 - 12 ms per update
        # Assumes fairly small rotation per frame
        #start = pg.time.get_ticks()
        self.prev_rect = self.rect
        acc = force/self.m
        start_vel = self.v
        self.COM += start_vel + 0.5*acc
        self.angle += self.ang_vel    # mod 360?
        self.v += acc
        # Tentative sleeping implementation
        if (abs_2v(self.v) < STAT_THRESHOLD) and (abs(self.ang_vel*self.radius) < STAT_THRESHOLD):
            self.stationary_counter += 1
            if self.stationary_counter > 15:
                # Consider counting collisions instead of stationary frames
                self.v = np.array([0, 0], dtype=float)
                self.ang_vel = 0
                self.stationary = True
        else:
            self.stationary_counter = 0
            self.stationary = False
        if self.stationary:
            return
        # ---------------
        rot_ctc_vec = rotate_2v(self.ctc_vec, -self.angle)
        center_pos = self.COM + np.array(rot_ctc_vec)
        self.rot_image = pg.transform.rotate(self.image, self.angle*180/PI)
        self.rot_boundary_image = pg.transform.rotate(self.boundary_image, self.angle*180/PI)
        self.rect = self.rot_image.get_rect(center=center_pos)
        self.s = np.array(self.rect.topleft)
        self.rot_mask = pg.mask.from_surface(self.rot_image)
        self.rot_boundary_mask = pg.mask.from_surface(self.rot_boundary_image)
        # Detect and resolve collision.
        #normal, POI, depth = self.detect_collision(world)
        normal, POI, depth = self.detect_collision_map(world) # Maybe do this bit at TOI
        n_tangent = rotate_2v(normal, -0.5*PI)
        impulse = 0
        if not (normal[0] == 0 and normal[1] == 0):
            r = POI - self.COM
            r_para_sq = r.dot(r) - (r.dot(normal))*(r.dot(normal))  # comp. of r perp to normal
            POI_vel = self.get_point_velocity(POI)
            # Problem: depth is never smaller than 1. This causes trouble maybe
            POI_normal_spd = -POI_vel.dot(normal)  # Needs special handling when it gets very small
            POI_tangent_spd = POI_vel.dot(n_tangent)
            t_rwnd = depth/POI_normal_spd
            TOI = 1 - t_rwnd
            if TOI <= 1:  # Don't need to check TOI > 0 ???
                # Maybe don't rewind if depth is 0 or 1
                # Move back to start of frame, then forward to TOI
                # Ideally, would re-calculate POI and normal at TOI
                self.COM -= start_vel + 0.5*acc  # Can be more efficient
                self.angle -= self.ang_vel
                self.COM += start_vel*TOI + 0.5*acc*TOI*TOI
                self.angle += self.ang_vel*TOI
                self.v = start_vel + acc*TOI 
                # Experimental ---------------- Does not work, for unknown reason
                """
                POI -= (POI_vel*t_rwnd).astype(int)
                r = POI - self.COM
                r_para_sq = r.dot(r) - (r.dot(normal))*(r.dot(normal))  # comp. of r perp to normal
                POI_vel = self.get_point_velocity(POI)
                POI_normal_spd = -POI_vel.dot(normal)
                POI_tangent_spd = POI_vel.dot(n_tangent)
                """
                #----------------------------------
                # Identify stationary contact here. Apply "stopping impulse"
                # Find impulse
                e = self.rebound*world.rebound
                impulse = -(1+e)*POI_vel.dot(normal)/(1/self.m + r_para_sq/self.I)
                # Find friction
                tangent_force = force.dot(n_tangent)
                if abs(POI_tangent_spd) >= STAT_THRESHOLD or abs(tangent_force) > self.friction_s*impulse:
                    friction_impulse = -np.sign(POI_tangent_spd)*self.friction_d*impulse
                    stopping_impulse = -POI_tangent_spd/(1/self.m + r.dot(normal)*r.dot(normal)/self.I)
                    if abs(friction_impulse) > abs(stopping_impulse):
                        friction_impulse = stopping_impulse
                else:
                    friction_impulse = -tangent_force
                # Apply impulses
                self.v += normal*impulse/self.m 
                self.ang_vel -= cross_2v(r, normal)*impulse/(self.I)
                self.v += n_tangent*friction_impulse/self.m
                self.ang_vel -= cross_2v(r, n_tangent)*friction_impulse/(self.I) # Not sure what is right sign here
                # Forward to end of frame
                self.COM += self.v*t_rwnd + 0.5*acc*t_rwnd*t_rwnd
                self.angle += self.ang_vel*t_rwnd
                self.v += acc*t_rwnd
                # Apply to image
                self.rot_image = pg.transform.rotate(self.image, self.angle*180/PI)
                #self.rot_boundary_image = pg.transform.rotate(self.boundary_image, self.angle*180/PI)
                self.rect = self.rot_image.get_rect(center=center_pos)
            """
            if TOI < 0 or TOI > 1:
                # Dirty hack to avoid clipping. Does not preserve energy
                if depth < 20:
                    print(depth)
                    self.COM += depth*normal
                else:
                    pass
            """
        else:
            pass
        #end = pg.time.get_ticks()
        #if end-start > 10:
        #    print(f"MSPU: {end-start:.2f}")

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
            """
            if boundary_1_center[0] == boundary_2_center[0] and boundary_1_center[1] == boundary_2_center[1]:
                print("dang")
            """
            displacement = boundary_2_center - boundary_1_center
            normal = normalize_2v(displacement)
            depth = abs_2v(displacement)
            POI = self.s + boundary_1_center
            if boundary_2_center[0] == 0 and boundary_2_center[1] == 0:
                # The centroid function fucks up some times for some reason.
                # Maybe simply don't use it
                print("Fuck")
                print(f"Center 2: {boundary_2_center}")
                print(f"Count: {boundary_overlap_2.count()}")
                normal = np.array([0,0], dtype=float)
                POI = np.array([0,0], dtype=float)
                depth = 0
        #end = pg.time.get_ticks()
        #print(f"Collision detection time: {end-start:.2f} ms")
        return normal, POI, depth

    def detect_collision_map(self, world):
        # Current time: 1 - 3 ms per call
        #start = pg.time.get_ticks()
        normal = np.array([0,0], dtype=float)
        POI = np.array([0,0], dtype=float)
        depth = 0
        intersect = self.rot_boundary_mask.overlap(world.boundary_mask, (-1)*self.s)
        if intersect:
            # Experimental -------
            intersect_map = self.rot_mask.overlap_mask(world.boundary_mask, (-1)*self.s) # Or rot_boundary_mask
            bw_image = intersect_map.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,0))
            intersect_array = (pg.surfarray.array2d(bw_image) != 0)
            x_1, x_2 = self.s[0], self.s[0] + self.rect.width
            y_1, y_2 = self.s[1], self.s[1] + self.rect.height
            x_1 = max(x_1, 0)
            x_2 = min(x_2, world.rect.width)
            y_1 = max(y_1, 0)
            y_2 = min(y_2, world.rect.height)
            a_1 = -min(self.s[0], 0)
            a_2 = min(self.rect.width, world.rect.width - self.s[0])
            b_1 = -min(self.s[1], 0)
            b_2 = min(self.rect.height, world.rect.height - self.s[1])
            normal_array = np.zeros((self.rect.width, self.rect.height, 2), dtype=float)
            normal_array[a_1:a_2, b_1:b_2] = world.normal_map[x_1:x_2, y_1:y_2]
            normal[0] = np.sum(normal_array[:,:,0]*intersect_array)
            normal[1] = np.sum(normal_array[:,:,1]*intersect_array)
            # Trick to undo the effect of tiny bits of corners when calculating normal
            # 0.05 limits resolution of normal vector to 1 degree
            if abs(normal[0]) < abs(0.05*normal[1]):
                normal[0] = 0
            elif abs(normal[1]) < abs(0.05*normal[0]):
                normal[1] = 0
            normal = normalize_2v(normal)
            #----------
            object_edge_map = self.rot_boundary_mask.overlap_mask(world.mask, (-1)*self.s)
            POI_rel = np.array(object_edge_map.centroid())
            POI = self.s + POI_rel
            displacement = np.array(intersect_map.centroid()) - POI_rel
            depth = abs_2v(displacement)
        #end = pg.time.get_ticks()
        #print(f'MSPD: {end-start}')
        return normal, POI, depth


class Terrain(pg.sprite.Sprite):
    def __init__(self, background, terrain, foreground):
        pg.sprite.Sprite.__init__(self)
        self.background = background
        self.terrain = terrain
        self.foreground = foreground
        self.boundary = get_boundary(terrain)
        self.boundary_mask = pg.mask.from_surface(self.boundary)
        self.mask = pg.mask.from_surface(self.terrain)
        self.normal_map = get_normal_map(self.terrain)
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


def get_normal_map(image):
    # Consider smoothing the surface with a gaussian blur first
    print('Getting normal map...')
    mask = pg.mask.from_surface(image)
    bw_image = mask.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,255))
    img_array = pg.surfarray.array2d(bw_image)
    normal_array = np.zeros((image.get_width(), image.get_height(), 2), dtype=float)
    sobel_kernel_x = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    sobel_kernel_y = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    grad_x = signal.convolve2d(img_array, sobel_kernel_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(img_array, sobel_kernel_y, boundary='symm', mode='same')
    normal_array = np.transpose(np.array([grad_x, grad_y], dtype=float), (1,2,0))
    normal_array = normalize_2v_array(normal_array)
    """
    for x in range(image.get_width()):
        for y in range(image.get_height()):
            normal_array[x, y] = normalize_2v(np.array((grad_x[x,y], grad_y[x,y])))
    """
    print('Got normal map')
    return normal_array


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


def rotate_2v(vector, angle):
    x = np.cos(angle)*vector[0] - np.sin(angle)*vector[1]
    y = np.sin(angle)*vector[0] + np.cos(angle)*vector[1]
    return np.array([x, y])


def normalize_2v(vector):
    mod_squared = vector.dot(vector)
    if mod_squared < 0.00001:
        return np.array([0, 0], dtype=float)
    else:
        return vector/np.sqrt(mod_squared)


def normalize_2v_array(array):
    mod_squared = array[:,:,0]*array[:,:,0] + array[:,:,1]*array[:,:,1]
    mod_squared[mod_squared < 0.00001] = 1
    mod = np.sqrt(mod_squared)
    normalized = array
    normalized[:,:,0] = normalized[:,:,0]/mod
    normalized[:,:,1] = normalized[:,:,1]/mod
    normalized[(normalized > 0.00001)*(normalized < 0.00001)] = 0
    return normalized


def cross_2v(a, b):
    return a[0]*b[1] - a[1]*b[0]


def draw_window(world, player):
    # Current time use: 7 ms per call
    #start = pg.time.get_ticks()
    WIN.blit(world.background, player.prev_rect, area=player.prev_rect)
    WIN.blit(world.terrain, player.prev_rect, area=player.prev_rect)
    WIN.blit(player.rot_image, player.rect)
    WIN.blit(world.foreground, player.prev_rect, area=player.prev_rect)
    pg.display.update()
    #end = pg.time.get_ticks()
    #print(f"MSPB: {end-start:.2f} ms")


def get_input_dir(keys_pressed):
    input_dir = np.array([0, 0], dtype=float)
    if keys_pressed[pg.K_d]:
        input_dir[0] += 1
    if keys_pressed[pg.K_a]:
        input_dir[0] -= 1
    """
    if keys_pressed[pg.K_w]:
        input_dir[1] -= 1
    if keys_pressed[pg.K_s]:
        input_dir[1] += 1
    """
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
                    force += JUMP_FORCE*player.m*normalize_2v(np.array([0, -1]) + input_dir)
        if keys_pressed[pg.K_q]:
            player.ang_vel += PI/180
        if keys_pressed[pg.K_e]:
            player.ang_vel -= PI/180
        force += INPUT_FORCE*input_dir*player.m*(input_dir.dot(player.v) < MAX_SPEED)
        force += GRAVITY*player.m*(not player.stationary)
        force -= AIR_RESISTANCE*(player.rect.width/10)*(player.rect.height/10)*player.v
        # Hacky air resistance against rotation
        player.ang_vel -= AIR_RESISTANCE*(player.rect.width/50)*(player.rect.height/50)*player.ang_vel
        player.update(world, force)
        draw_window(world, player)
    pg.quit()


if __name__ == "__main__":
    main()
