from __future__ import annotations
from dataclasses import dataclass, field
import math
from operator import pos
import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import matrix44
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================
WIDTH, HEIGHT   = 1200, 800
TITLE           = "3D Gravity Simulation"

G               = 6.6740e-11
AU              = 149.6e9
DAY             = 3600 * 24
SIM_SPEED       = DAY * 1.0
SCALE           = 2 * AU

GRID_SIZE       = 10
GRID_DIVISIONS  = 120
BG_COLOR        = (0.01, 0.01, 0.025)
GRID_COLOR      = (0.18, 0.18, 0.30, 1.0)

MAX_LIGHTS      = 4

M_SUN           = 1.989e30
WELL_SCALE      = 0.90
WELL_SOFTENING  = 0.30

MAX_SUBSTEP_S   = DAY * 1.0
MAX_SUBSTEPS    = 512
TRAIL_MAX_JUMP  = 1.5
TRAIL_WIDTH     = 2.0 # pixels — change this freely

# Colors (RGBA 0–1)
YELLOW  = (1.0,  0.95, 0.3,  1.0)
BLUE    = (0.3,  0.6,  1.0,  1.0)
RED     = (1.0,  0.4,  0.4,  1.0)
ORANGE  = (1.0,  0.6,  0.1,  1.0)
CYAN    = (0.3,  1.0,  0.9,  1.0)
TAN     = (0.87, 0.72, 0.53, 1.0)
BUFF    = (0.76, 0.60, 0.42, 1.0)
CREAM   = (0.95, 0.90, 0.72, 1.0)
GREEN   = (0.4,  1.0,  0.4,  1.0)

# UI
PANEL_W, PANEL_H  = 300, 280
STATS_W, STATS_H  = 280,  90
PANEL_MARGIN      = 16
PICK_THRESHOLD_PX = 50

# ============================================================
# 1. Body
# ============================================================

@dataclass
class Body:
    pos:        np.ndarray = field(compare=False)
    vel:        np.ndarray = field(compare=False)
    mass:       float

    color:      tuple = (1.0, 1.0, 1.0, 1.0)
    name:       str   = "Body"
    is_star:    bool  = False
    radius:     float = 0.025
    rot_speed:  float = 1.0
    axial_tilt: float = 0.0
    has_rings:  bool  = False

    rotation:  float  = field(default=0.0,  repr=False)
    trail:     deque  = field(default_factory=lambda: deque(maxlen=600))
    trail_vbo: object = field(default=None, repr=False)
    trail_vao: object = field(default=None, repr=False)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)

    def merge(self, other: "Body") -> None:
        total_m = self.mass + other.mass
        self.vel = (self.vel * self.mass + other.vel * other.mass) / total_m
        self.vel *= 0.97
        self.pos = (self.pos * self.mass + other.pos * other.mass) / total_m
        if other.mass > self.mass:
            self.color = other.color
        self.radius = (self.radius**3 + other.radius**3) ** (1/3)
        self.mass = total_m
        self.rot_speed = (self.rot_speed + other.rot_speed) * 0.5
        other.release_gpu()

    def release_gpu(self):
        if self.trail_vbo:
            self.trail_vbo.release()
            self.trail_vao.release()
            self.trail_vbo = None
            self.trail_vao = None

    def clear_trail(self):
        self.trail.clear()
        self.release_gpu()

# ============================================================
# 2. Physics Engine
# ============================================================

class PhysicsEngine:
    def __init__(self, bodies: list[Body]):
        self.bodies = bodies

    def _compute_accels(self):
        if len(self.bodies) < 2:
            return [np.zeros(3) for _ in self.bodies]

        pos  = np.array([b.pos  for b in self.bodies])
        mass = np.array([b.mass for b in self.bodies])
        soft = np.array([b.radius * SCALE for b in self.bodies])

        diff  = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        dist2 = np.einsum('ijk,ijk->ij', diff, diff)

        soft_mat = (soft[:, np.newaxis] + soft[np.newaxis, :]) * 0.5
        dist2 = np.maximum(dist2, soft_mat ** 2)

        dist3 = dist2 ** 1.5
        np.fill_diagonal(dist3, 1.0)

        coeff = G * mass[np.newaxis, :] / dist3
        np.fill_diagonal(coeff, 0.0)
        accels = np.einsum('ij,ijk->ik', coeff, diff)
        return list(accels)

    def _estimate_max_accel(self):
        accels = self._compute_accels()
        if not accels: return 0.0
        return max(np.linalg.norm(a) for a in accels)

    def step(self, dt: float):
        if not self.bodies:
            return
        max_a = self._estimate_max_accel()
        if max_a > 0:
            safe_dt = min(MAX_SUBSTEP_S, math.sqrt(SCALE / max_a))
        else:
            safe_dt = MAX_SUBSTEP_S
        substeps = min(MAX_SUBSTEPS, max(1, math.ceil(dt / safe_dt)))
        sub_dt = dt / substeps
        for _ in range(substeps):
            self._verlet_step(sub_dt)
            self.handle_collisions(dt)

    def _verlet_step(self, dt: float):
        a0 = self._compute_accels()
        for i, b in enumerate(self.bodies):
            b.pos      += b.vel * dt + 0.5 * a0[i] * dt * dt
            b.rotation += b.rot_speed * (dt / DAY)
        a1 = self._compute_accels()
        for i, b in enumerate(self.bodies):
            b.vel += 0.5 * (a0[i] + a1[i]) * dt

    def handle_collisions(self, dt: float):
        to_remove = set()
        num_bodies = len(self.bodies)

        def r_m(b): return b.radius * SCALE

        for i in range(num_bodies):
            if i in to_remove: continue
            for j in range(i + 1, num_bodies):
                if j in to_remove: continue
                a, b = self.bodies[i], self.bodies[j]
                diff = a.pos - b.pos
                dist = np.linalg.norm(diff)
                rel_vel_vec = a.vel - b.vel
                next_dist = np.linalg.norm(diff + rel_vel_vec * dt)
                hitbox = r_m(a) + r_m(b)
                if dist < hitbox or next_dist < hitbox:
                    print(f"COLLISION: {a.name} hits {b.name}  dist={dist:.3e}  hitbox={hitbox:.3e}")
                    if a.mass >= b.mass:
                        a.merge(b); to_remove.add(j)
                    else:
                        b.merge(a); to_remove.add(i); break
        if to_remove:
            self.bodies = [b for idx, b in enumerate(self.bodies) if idx not in to_remove]

# ============================================================
# 3. Sphere mesh
# ============================================================

def build_sphere_mesh(stacks: int = 28, slices: int = 28):
    verts, indices = [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.extend([x, y, z, x, y, z])
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices += [a, b, a + 1, b, b + 1, a + 1]
    return np.array(verts, dtype='f4'), np.array(indices, dtype='i4')

SPHERE_VERTS, SPHERE_INDICES = build_sphere_mesh()

# ============================================================
# Ring mesh
# ============================================================

def build_ring_mesh(inner=1.45, outer=2.35, segments=120):
    verts = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        c, s = math.cos(angle), math.sin(angle)
        verts += [inner * c, 0.0, inner * s]
        verts += [outer * c, 0.0, outer * s]
    indices = []
    for i in range(segments):
        a = i * 2; b = i * 2 + 1
        indices += [a, b, a + 2, b, b + 2, a + 2]
    return np.array(verts, dtype='f4'), np.array(indices, dtype='i4')

RING_VERTS, RING_INDICES = build_ring_mesh()

# ============================================================
# Orbital Mechanics Helpers
# ============================================================

def radius_to_meters(r_gl: float) -> float:
    return r_gl * SCALE

def kepler_to_cartesian(a, e, M_central, inc=0.0, omega=0.0, Omega=0.0):
    f = 0.0
    r = a * (1 - e**2) / (1 + e * math.cos(f))
    x_orb = r * math.cos(f)
    y_orb = r * math.sin(f)
    mu = G * M_central
    v = math.sqrt(mu * (2/r - 1/a))
    vx_orb = 0.0
    vy_orb = v
    cosO, sinO = math.cos(Omega), math.sin(Omega)
    cosi, sini = math.cos(inc), math.sin(inc)
    cosw, sinw = math.cos(omega), math.sin(omega)
    R11 = cosO*cosw - sinO*sinw*cosi
    R12 = -cosO*sinw - sinO*cosw*cosi
    R21 = sinO*cosw + cosO*sinw*cosi
    R22 = -sinO*sinw + cosO*cosw*cosi
    R31 = sinw*sini
    R32 = cosw*sini
    pos = np.array([R11*x_orb + R12*y_orb, R31*x_orb + R32*y_orb, R21*x_orb + R22*y_orb])
    vel = np.array([R11*vx_orb + R12*vy_orb, R31*vx_orb + R32*vy_orb, R21*vx_orb + R22*vy_orb])
    return pos, vel

def star_luminosity(mass):
    return (mass / M_SUN) ** 3.5

def star_temperature(mass):
    return 5800 * ((mass / M_SUN) ** 0.5)

def temp_to_color(T):
    T = max(3000, min(12000, T))
    x = (T - 3000) / (12000 - 3000)
    return (1.0, 0.5 + 0.5 * x, 0.3 + 0.7 * x, 1.0)

# ============================================================
# 4. Scenarios
# ============================================================

def make_star(pos, vel, mass, name, axial_tilt=0.0, rot_speed=0.5):
    color = temp_to_color(star_temperature(mass))
    return Body(pos=pos, vel=vel, mass=mass, color=color, name=name,
                is_star=True, rot_speed=rot_speed, axial_tilt=axial_tilt)

def scenario_solar_system() -> list[Body]:
    sun = make_star(np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]),
                    M_SUN, "Sun", axial_tilt=7.25, rot_speed=0.5)

    def make_planet(a_au, e, inc_deg, mass, color, name, radius_km, has_rings=False):
        a   = a_au * AU
        inc = math.radians(inc_deg)
        p, v = kepler_to_cartesian(a, e, M_SUN, inc=inc)
        r = (radius_km / 70000) * 0.04
        return Body(p, v, mass, color=color, name=name, radius=r, has_rings=has_rings)

    return [
        sun,
        make_planet(0.387, 0.206,  7.0, 3.30e23, ORANGE, "Mercury", 2440),
        make_planet(0.723, 0.007,  3.4, 4.87e24, TAN,    "Venus",   6052),
        make_planet(1.000, 0.017,  0.0, 5.97e24, BLUE,   "Earth",   6371),
        make_planet(1.524, 0.093,  1.9, 6.39e23, RED,    "Mars",    3390),
        make_planet(5.204, 0.049,  1.3, 1.90e27, BUFF,   "Jupiter", 69911),
        make_planet(9.582, 0.057,  2.5, 5.68e26, CREAM,  "Saturn",  58232, has_rings=True),
    ]

def scenario_binary_stars() -> list[Body]:
    dist = 1 * AU
    M = 2e30
    v = math.sqrt(G * M / 4 / dist)
    return [
        make_star(np.array([ dist,0,0]), np.array([0,0, v]), M, "Star A"),
        make_star(np.array([-dist,0,0]), np.array([0,0,-v]), M, "Star B"),
    ]

def scenario_figure_eight() -> list[Body]:
    M  = 1.0e30
    L  = 0.5 * AU
    v0 = math.sqrt(G * M / L)
    px, pz = 0.97000436 * L, 0.24308753 * L
    vx, vz = 0.46620368 * v0, 0.43236573 * v0
    return [
        make_star(np.array([-px,0, pz]), np.array([ vx,0, vz]), M, "Body A"),
        make_star(np.array([ px,0,-pz]), np.array([ vx,0, vz]), M, "Body B"),
        make_star(np.array([  0,0,  0]), np.array([-2*vx,0,-2*vz]), M, "Body C"),
    ]

def scenario_star_system() -> list[Body]:
    dist = 0.5 * AU
    M = 2e30
    v = math.sqrt(G * M / dist)
    return [
        make_star(np.array([ dist,0,0]), np.array([0,0, v]), M, "Star A"),
        make_star(np.array([-dist,0,0]), np.array([0,0,-v]), M, "Star B"),
    ]

def scenario_RexPrime() -> list[Body]:
    M_A = 1.1 * M_SUN; M_B = 0.9 * M_SUN; M_tot = M_A + M_B
    a_bin = 3.5 * AU
    r_A = a_bin * (M_B / M_tot); r_B = a_bin * (M_A / M_tot)
    omega = math.sqrt(G * M_tot / (a_bin ** 3))
    starA = make_star(np.array([+r_A,0,0]), np.array([0,0,+omega*r_A]), M_A, "Rex Star A")
    starB = make_star(np.array([-r_B,0,0]), np.array([0,0,-omega*r_B]), M_B, "Rex Star B")
    def R(km): return (km / 70000) * 0.04
    pA1 = Body(starA.pos+np.array([0.35*AU,0,0]), starA.vel+np.array([0,0,math.sqrt(G*M_A/(0.35*AU))]),
               5.97e24, color=GREEN, name="Rex Prime",   radius=R(6371))
    pA2 = Body(starA.pos+np.array([0.65*AU,0,0]), starA.vel+np.array([0,0,math.sqrt(G*M_A/(0.65*AU))]),
               4.87e24, color=BLUE,  name="Rex Minor",   radius=R(6052))
    pB1 = Body(starB.pos+np.array([0.30*AU,0,0]), starB.vel+np.array([0,0,-math.sqrt(G*M_B/(0.30*AU))]),
               6.39e23, color=CYAN,  name="Rex Ember",   radius=R(3390))
    cb1 = Body(np.array([12*AU,0,0]), np.array([0,0, math.sqrt(G*M_tot/(12*AU))]),
               5e24, color=CREAM, name="Rex Outer I",  radius=R(7000), has_rings=True)
    cb2 = Body(np.array([-16*AU,0,0]), np.array([0,0,-0.85*math.sqrt(G*M_tot/(16*AU))]),
               5e24, color=RED,   name="Rex Outer II", radius=R(7000))
    return [starA, starB, pA1, pA2, pB1, cb1, cb2]

def myBigGapingBlackHole() -> list[Body]:
    BH_mass = 30 * M_SUN
    def R(km): return (km / 70000) * 0.04
    return [
        Body(np.array([0,0,0]), np.array([0,0,0]), BH_mass,
             color=GREEN, name="Blacky", is_star=True, radius=1.0),
        Body(np.array([5*AU,0,0]), np.array([0,0,14000]), 2e24,
             color=BLUE, name="Planety", radius=R(6000)),
    ]

def scenario_chaos_cluster() -> list[Body]:
    bodies = [Body(np.array([0,0,0]), np.array([0,0,0]), 0.8*M_SUN,
                   color=YELLOW, name="Cluster Star", is_star=True, radius=0.2)]
    rng = np.random.default_rng()
    for i in range(12):
        m = rng.uniform(1e22, 5e24)
        bodies.append(Body(
            rng.uniform(-0.8*AU, 0.8*AU, size=3),
            rng.uniform(-5000, 5000, size=3), m,
            color=(rng.uniform(0.2,1.0), rng.uniform(0.2,1.0), rng.uniform(0.2,1.0), 1.0),
            name=f"Astro-{i+1}", radius=0.01 + 0.02*(m/5e24)))
    return bodies

SCENARIOS = {
    "1": ("Solar System",            scenario_solar_system),
    "2": ("Binary Stars",            scenario_binary_stars),
    "3": ("Figure Eight",            scenario_figure_eight),
    "4": ("Star System",             scenario_star_system),
    "5": ("Rex Prime",               scenario_RexPrime),
    "6": ("Black Hole - Test",       myBigGapingBlackHole),
    "7": ("Chaos Cluster",           scenario_chaos_cluster),
}

# ============================================================
# 5. Overlay Renderer
# ============================================================

class OverlayRenderer:
    _VERT = '''
        #version 330
        in vec2 in_pos; in vec2 in_uv; out vec2 uv;
        void main() { gl_Position = vec4(in_pos, 0.0, 1.0); uv = in_uv; }
    '''
    _FRAG = '''
        #version 330
        in vec2 uv; out vec4 f_color; uniform sampler2D tex;
        void main() { f_color = texture(tex, uv); }
    '''
    def __init__(self, ctx, win_w, win_h):
        self.ctx, self.win_w, self.win_h = ctx, win_w, win_h
        self.prog = ctx.program(vertex_shader=self._VERT, fragment_shader=self._FRAG)
        self.vbo  = ctx.buffer(reserve=6 * 4 * 4)
        self.vao  = ctx.simple_vertex_array(self.prog, self.vbo, 'in_pos', 'in_uv')

    def _ndc(self, x, y):
        return (x / self.win_w) * 2 - 1, 1 - (y / self.win_h) * 2

    def draw(self, img: Image.Image, x: int, y: int):
        w, h = img.size
        x0, y0 = self._ndc(x, y)
        x1, y1 = self._ndc(x + w, y + h)
        self.vbo.write(np.array([
            x0,y1, 0,1,  x1,y1, 1,1,  x0,y0, 0,0,
            x1,y1, 1,1,  x1,y0, 1,0,  x0,y0, 0,0,
        ], dtype='f4'))
        tex = self.ctx.texture(img.size, 4, img.tobytes())
        tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.prog['tex'].value = 0
        tex.use(0)
        self.vao.render(moderngl.TRIANGLES)
        tex.release()

# ============================================================
# 6. Panel Helpers
# ============================================================

def _get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:    return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()

FONT_LARGE  = _get_font(17)
FONT_MEDIUM = _get_font(14)
FONT_SMALL  = _get_font(12)
PANEL_BG    = (10,  12,  20,  220)
BORDER_COL  = (80,  90, 120, 255)
TEXT_DIM    = (140, 150, 170, 255)
TEXT_BRIGHT = (220, 230, 255, 255)

def _rrect(draw, xy, r, fill, outline):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=1)

def build_inspect_panel(body: Body, is_focused: bool) -> Image.Image:
    img  = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _rrect(draw, [0, 0, PANEL_W-1, PANEL_H-1], 10, PANEL_BG, BORDER_COL)
    r, g, b, _ = body.color
    nc = (int(r*255), int(g*255), int(b*255), 255)
    _rrect(draw, [8, 8, PANEL_W-9, 36], 6, (*nc[:3], 40), (*nc[:3], 120))
    draw.text((16, 11), f"{'★ ' if body.is_star else ''}{body.name}", font=FONT_LARGE, fill=nc)
    speed_kms = np.linalg.norm(body.vel) / 1000.0
    pos_au    = body.pos / AU
    dist_au   = np.linalg.norm(body.pos) / AU
    rows = [
        ("TYPE",   "Star" if body.is_star else "Planet"),
        ("MASS",   f"{body.mass:.3e} kg"),
        ("SPEED",  f"{speed_kms:.2f} km/s"),
        ("DIST",   f"{dist_au:.3f} AU"),
        ("POS  X", f"{pos_au[0]:+.3f} AU"),
        ("POS  Y", f"{pos_au[1]:+.3f} AU"),
        ("POS  Z", f"{pos_au[2]:+.3f} AU"),
    ]
    y = 48
    for lbl, val in rows:
        draw.text((16, y),  lbl, font=FONT_SMALL,  fill=TEXT_DIM)
        draw.text((100, y), val, font=FONT_MEDIUM, fill=TEXT_BRIGHT)
        y += 26
    draw.line([(10, PANEL_H-46), (PANEL_W-10, PANEL_H-46)], fill=BORDER_COL, width=1)
    if is_focused:
        draw.text((16, PANEL_H-38), "⦿ FOCUSED  —  F to release",
                  font=FONT_SMALL, fill=(120, 220, 120, 255))
    else:
        draw.text((16, PANEL_H-38), "F to focus camera",
                  font=FONT_SMALL, fill=(*TEXT_DIM[:3], 180))
    draw.text((16, PANEL_H-18), "Click empty space to deselect",
              font=FONT_SMALL, fill=(*TEXT_DIM[:3], 140))
    return img

def build_stats_panel(day, speed, count, paused, focus_name) -> Image.Image:
    h = STATS_H + (20 if focus_name else 0)
    img  = Image.new("RGBA", (STATS_W, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _rrect(draw, [0, 0, STATS_W-1, h-1], 10, PANEL_BG, BORDER_COL)
    draw.text((14, 10), "3D GRAVITY SIMULATION", font=FONT_SMALL, fill=(180, 200, 255, 255))
    draw.line([(10, 28), (STATS_W-10, 28)], fill=BORDER_COL, width=1)
    status     = "  PAUSED" if paused else f"  Day {day:,.1f}"
    status_col = (255, 200, 80, 255) if paused else TEXT_BRIGHT
    draw.text((14, 34), status, font=FONT_MEDIUM, fill=status_col)
    draw.text((14, 56), f"  Speed {speed:.0f}×   Bodies {count}", font=FONT_MEDIUM, fill=TEXT_DIM)
    if focus_name:
        draw.line([(10, 76), (STATS_W-10, 76)], fill=BORDER_COL, width=1)
        draw.text((14, 80), f"  ⦿ {focus_name}", font=FONT_SMALL, fill=(120, 220, 120, 255))
    return img

def build_hint_panel() -> Image.Image:
    img  = Image.new("RGBA", (220, 34), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _rrect(draw, [0, 0, 219, 33], 8, (10,12,20,160), BORDER_COL)
    draw.text((12, 9), "Click a body to inspect it", font=FONT_SMALL, fill=(160,170,200,200))
    return img

# ============================================================
# 7. Shaders
# ============================================================

_SPHERE_VERT = '''
#version 330
in vec3 in_pos;
in vec3 in_normal;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normal_mat;
out vec3 frag_pos;
out vec3 frag_normal;
void main() {
    vec4 world = model * vec4(in_pos, 1.0);
    frag_pos    = world.xyz;
    frag_normal = normalize(normal_mat * in_normal);
    gl_Position = projection * view * world;
}
'''

_SPHERE_FRAG = f'''
#version 330
in vec3 frag_pos;
in vec3 frag_normal;
uniform vec4  body_color;
uniform bool  is_star;
uniform vec3  eye_pos;
uniform vec3  light_pos[{MAX_LIGHTS}];
uniform int   num_lights;
uniform float time;
out vec4 f_color;

float hash(vec3 p) {{
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}}

void main() {{
    vec3 rgb = body_color.rgb;
    if (is_star) {{
        vec3  V    = normalize(eye_pos - frag_pos);
        float limb = mix(0.72, 1.0, clamp(dot(frag_normal, V), 0.0, 1.0));
        // Subtle animated surface variation
        float n = hash(frag_normal * 8.0 + vec3(time * 0.05));
        n = mix(0.90, 1.0, n);
        f_color = vec4(rgb * limb * n, 1.0);
    }} else {{
        vec3  V = normalize(eye_pos - frag_pos);
        vec3  N = frag_normal;
        const float AMBIENT   = 0.07;
        const float SHININESS = 40.0;
        float diffuse  = 0.0;
        float specular = 0.0;
        for (int i = 0; i < num_lights; i++) {{
            vec3  L = normalize(light_pos[i] - frag_pos);
            float d = max(dot(N, L), 0.0);
            diffuse  += d;
            vec3  H   = normalize(L + V);
            specular += pow(max(dot(N, H), 0.0), SHININESS) * d;
        }}
        diffuse  = min(diffuse,  1.0);
        specular = min(specular, 0.7);
        // Enhanced atmospheric rim glow
        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.8;
        vec3 atmo = rgb * 0.4 + vec3(0.1, 0.3, 0.5) * 0.6;
        vec3 lit = rgb * (AMBIENT + (1.0 - AMBIENT) * diffuse)
                 + vec3(0.85, 0.92, 1.0) * specular * 0.45
                 + atmo * rim;
        f_color = vec4(clamp(lit, 0.0, 1.0), 1.0);
    }}
}}
'''

_GRID_VERT = '''
#version 330
in vec3 in_pos;
uniform mat4 projection;
uniform mat4 view;
void main() { gl_Position = projection * view * vec4(in_pos, 1.0); }
'''
_GRID_FRAG = '''
#version 330
out vec4 f_color;
uniform vec4 color;
void main() { f_color = color; }
'''

# ── Fading trail shader ──────────────────────────────────────
_TRAIL_VERT = '''
#version 330
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    v_color     = in_color;
    gl_Position = vec4(in_pos, 1.0);  // pass raw world pos to geometry stage
}
'''

_TRAIL_GEOM = '''
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
in  vec4 v_color[];
out vec4 g_color;
uniform mat4 projection;
uniform mat4 view;
uniform float line_width;
uniform vec2  viewport;

void main() {
    vec4 p0 = projection * view * gl_in[0].gl_Position;
    vec4 p1 = projection * view * gl_in[1].gl_Position;

    // NDC
    vec2 n0 = p0.xy / p0.w;
    vec2 n1 = p1.xy / p1.w;

    // Screen-space direction and perpendicular
    vec2 dir   = normalize((n1 - n0) * viewport);
    vec2 perp  = vec2(-dir.y, dir.x) * (line_width / viewport);

    g_color = v_color[0];
    gl_Position = vec4((n0 - perp) * p0.w, p0.z, p0.w); EmitVertex();
    gl_Position = vec4((n0 + perp) * p0.w, p0.z, p0.w); EmitVertex();

    g_color = v_color[1];
    gl_Position = vec4((n1 - perp) * p1.w, p1.z, p1.w); EmitVertex();
    gl_Position = vec4((n1 + perp) * p1.w, p1.z, p1.w); EmitVertex();

    EndPrimitive();
}
'''

_TRAIL_FRAG = '''
#version 330
in  vec4 g_color;
out vec4 f_color;
void main() { f_color = g_color; }
'''

# ── Star glow shader ─────────────────────────────────────────
_GLOW_VERT = '''
#version 330
in vec2 in_corner;
uniform mat4 projection;
uniform mat4 view;
uniform vec3 star_pos;
uniform float glow_size;
out vec2 uv;
void main() {
    vec4 clip_centre = projection * view * vec4(star_pos, 1.0);
    float ndc_radius = projection[0][0] * glow_size / clip_centre.w;
    uv = in_corner;
    gl_Position = vec4(
        clip_centre.xy / clip_centre.w + in_corner * ndc_radius,
        clip_centre.z  / clip_centre.w, 1.0);
}
'''
_GLOW_FRAG = '''
#version 330
in  vec2 uv;
out vec4 f_color;
uniform vec4 glow_color;
void main() {
    float d = length(uv);
    if (d > 1.0) discard;
    float alpha = exp(-d * d * 2.8) * 0.65;
    f_color = vec4(glow_color.rgb, alpha);
}
'''

# ── Starfield shader ─────────────────────────────────────────
_STAR_VERT = '''
#version 330
in vec3 in_pos;
in float in_brightness;
uniform mat4 projection;
uniform mat4 view;
out float brightness;
void main() {
    brightness  = in_brightness;
    gl_Position = projection * view * vec4(in_pos, 1.0);
    gl_PointSize = 2.0;
}
'''
_STAR_FRAG = '''
#version 330
in  float brightness;
out vec4  f_color;
void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float a = smoothstep(0.5, 0.1, d) * brightness;
    f_color = vec4(vec3(brightness), a);
}
'''

# ── Ring shader ───────────────────────────────────────────────
_RING_VERT = '''
#version 330
in vec3 in_pos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
out vec3 frag_pos_local;
void main() {
    frag_pos_local = in_pos;
    gl_Position = projection * view * model * vec4(in_pos, 1.0);
}
'''
_RING_FRAG = '''
#version 330
in  vec3 frag_pos_local;
out vec4 f_color;
uniform float inner_r;
uniform float outer_r;
void main() {
    float d = length(frag_pos_local.xz);
    float t = (d - inner_r) / (outer_r - inner_r);
    if (t < 0.0 || t > 1.0) discard;
    float alpha = sin(t * 3.14159) * 0.60;
    // Banded ring color: warm tan with subtle variation
    float band = fract(t * 6.0);
    vec3 col_a = vec3(0.78, 0.68, 0.50);
    vec3 col_b = vec3(0.58, 0.50, 0.38);
    vec3 color = mix(col_a, col_b, smoothstep(0.3, 0.7, band));
    f_color = vec4(color, alpha);
}
'''

# ============================================================
# 8. Main Window
# ============================================================

class GravitySim(mglw.WindowConfig):
    gl_version  = (3, 3)
    title       = TITLE
    window_size = (WIDTH, HEIGHT)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.time_step_multi  = 1.0
        self.is_paused        = False
        self.tracking_index   = None
        self.selected_index   = None
        self.sim_elapsed_days = 0.0
        self._trail_tick      = 0

        self.camera_target = np.array([0.0, 0.0, 0.0], dtype='f4')
        self.camera_yaw    = -math.pi / 2
        self.camera_pitch  = -math.pi / 6
        self.camera_radius = 5.0
        self.use_barycentric_camera = False

        self.physics = PhysicsEngine([])

        # ── Sphere ────────────────────────────────────────────
        self.sphere_prog = self.ctx.program(
            vertex_shader=_SPHERE_VERT, fragment_shader=_SPHERE_FRAG)
        vbo = self.ctx.buffer(SPHERE_VERTS)
        ibo = self.ctx.buffer(SPHERE_INDICES)
        self.sphere_vao = self.ctx.vertex_array(
            self.sphere_prog, [(vbo, '3f 3f', 'in_pos', 'in_normal')], ibo)

        # ── Grid ──────────────────────────────────────────────
        self.grid_prog = self.ctx.program(
            vertex_shader=_GRID_VERT, fragment_shader=_GRID_FRAG)
        self._init_grid(GRID_SIZE, GRID_DIVISIONS)

        # ── Fading trail ──────────────────────────────────────
        self.trail_prog = self.ctx.program(
            vertex_shader=_TRAIL_VERT, fragment_shader=_TRAIL_FRAG, geometry_shader=_TRAIL_GEOM)

        # ── Star glow ─────────────────────────────────────────
        self.glow_prog = self.ctx.program(
            vertex_shader=_GLOW_VERT, fragment_shader=_GLOW_FRAG)
        quad = np.array([-1,-1, 1,-1, -1,1, 1,-1, 1,1, -1,1], dtype='f4')
        self.glow_vao = self.ctx.simple_vertex_array(
            self.glow_prog, self.ctx.buffer(quad), 'in_corner')

        # ── Starfield ─────────────────────────────────────────
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.star_prog = self.ctx.program(
            vertex_shader=_STAR_VERT, fragment_shader=_STAR_FRAG)
        rng = np.random.default_rng(42)
        N   = 4000
        phi   = np.arccos(rng.uniform(-1, 1, N))
        theta = rng.uniform(0, 2*math.pi, N)
        R_sky = 800.0
        star_verts = np.column_stack([
            R_sky * np.sin(phi) * np.cos(theta),
            R_sky * np.sin(phi) * np.sin(theta),
            R_sky * np.cos(phi),
            rng.uniform(0.3, 1.0, N),
        ]).astype('f4')
        self.starfield_vbo = self.ctx.buffer(star_verts)
        self.starfield_vao = self.ctx.vertex_array(
            self.star_prog,
            [(self.starfield_vbo, '3f 1f', 'in_pos', 'in_brightness')])

        # ── Rings ─────────────────────────────────────────────
        self.ring_prog = self.ctx.program(
            vertex_shader=_RING_VERT, fragment_shader=_RING_FRAG)
        ring_vbo = self.ctx.buffer(RING_VERTS)
        ring_ibo = self.ctx.buffer(RING_INDICES)
        self.ring_vao = self.ctx.vertex_array(
            self.ring_prog, [(ring_vbo, '3f', 'in_pos')], ring_ibo)
        self.ring_prog['inner_r'].value = 1.45
        self.ring_prog['outer_r'].value = 2.35

        # ── 2-D overlay ───────────────────────────────────────
        self.overlay = OverlayRenderer(self.ctx, WIDTH, HEIGHT)

        self.load_scenario("1")
        self._print_controls()

    # ------------------------------------------------------------------
    def validate_indices(self):
        if self.selected_index is not None:
            if self.selected_index >= len(self.physics.bodies):
                self.selected_index = None
        if self.tracking_index is not None:
            if self.tracking_index >= len(self.physics.bodies):
                self.tracking_index = None

    # ------------------------------------------------------------------
    def _init_grid(self, size, divisions):
        n   = divisions + 1
        pts = np.linspace(-size, size, n, dtype='f4')
        XX, ZZ = np.meshgrid(pts, pts)
        self.grid_xz = np.column_stack([XX.ravel(), ZZ.ravel()])
        indices = []
        for iz in range(n):
            for ix in range(n - 1):
                indices += [iz*n+ix, iz*n+ix+1]
        for ix in range(n):
            for iz in range(n - 1):
                indices += [iz*n+ix, (iz+1)*n+ix]
        self.grid_line_indices = np.array(indices, dtype='i4')
        self.grid_vbo = self.ctx.buffer(reserve=len(self.grid_line_indices) * 3 * 4)
        self.grid_vao = self.ctx.simple_vertex_array(self.grid_prog, self.grid_vbo, 'in_pos')

    def _update_grid(self):
        xz = self.grid_xz
        y  = np.zeros(len(xz), dtype='f4')
        for body in self.physics.bodies:
            bx, _, bz = (body.pos / SCALE)
            dx   = xz[:, 0] - float(bx)
            dz   = xz[:, 1] - float(bz)
            dist = np.sqrt(dx*dx + dz*dz + WELL_SOFTENING**2)
            y   -= WELL_SCALE * (body.mass / M_SUN) / dist
        xyz = np.column_stack([xz[:, 0], y, xz[:, 1]]).astype('f4')
        self.grid_vbo.write(np.ascontiguousarray(xyz[self.grid_line_indices]))

    # ------------------------------------------------------------------
    def _clear_all_trails(self):
        for b in self.physics.bodies:
            b.clear_trail()

    # ------------------------------------------------------------------
    def _print_controls(self):
        print("=" * 52)
        print("  3D Gravity Simulator  –  Controls")
        print("=" * 52)
        print("  SPACE          Pause / Resume")
        print("  UP / DOWN      Speed ×2 / ÷2  (clears trails)")
        print("  Q / E          Zoom in / out")
        print("  Scroll         Smooth zoom")
        print("  Left-drag      Orbit camera")
        print("  Left-click     Inspect body")
        print("  F              Focus / unfocus selected body")
        print("  T              Cycle camera tracking")
        print("  R              Reset camera")
        print("  B              Toggle barycentric camera")
        for k, (name, _) in SCENARIOS.items():
            print(f"  {k}              Load: {name}")
        print("=" * 52)

    # ------------------------------------------------------------------
    def load_scenario(self, key: str):
        for b in self.physics.bodies:
            b.release_gpu()
        name, factory         = SCENARIOS[key]
        self.physics.bodies   = factory()
        self.sim_elapsed_days = 0.0
        self.tracking_index   = None
        self.selected_index   = None
        self.camera_target    = np.array([0.0, 0.0, 0.0], dtype='f4')
        print(f"Loaded: {name}")

    # ------------------------------------------------------------------
    def _get_matrices(self):
        if self.use_barycentric_camera and self.tracking_index is None:
            self.camera_target = self._barycenter().astype('f4')
        x = self.camera_radius * math.cos(self.camera_pitch) * math.cos(self.camera_yaw)
        y = self.camera_radius * math.sin(self.camera_pitch)
        z = self.camera_radius * math.cos(self.camera_pitch) * math.sin(self.camera_yaw)
        eye  = np.array([x, y, z], dtype='f4') + self.camera_target
        view = matrix44.create_look_at(eye, self.camera_target, [0,1,0], dtype='f4')
        proj = matrix44.create_perspective_projection(
            45.0, self.wnd.aspect_ratio, 0.001, 2000.0, dtype='f4')
        return proj, view, eye

    def _model_matrix(self, gl_pos, radius, rotation, axial_tilt_deg):
        tilt_rad = math.radians(axial_tilt_deg)
        T  = matrix44.create_from_translation(gl_pos.astype('f4'), dtype='f4')
        Zt = matrix44.create_from_z_rotation(tilt_rad,             dtype='f4')
        R  = matrix44.create_from_y_rotation(rotation,             dtype='f4')
        S  = matrix44.create_from_scale([radius]*3,                dtype='f4')
        return S @ Zt @ R @ T

    def _normal_matrix(self, model):
        m3 = model[:3, :3]
        try:    return np.ascontiguousarray(np.linalg.inv(m3).T, dtype='f4')
        except: return np.ascontiguousarray(np.eye(3, dtype='f4'))

    def _world_to_screen(self, gl_pos, proj, view):
        pos4 = np.array([*gl_pos, 1.0], dtype='f4')
        clip = pos4 @ view @ proj
        if abs(clip[3]) < 1e-6: return None
        ndc = clip[:3] / clip[3]
        if not (-1.1 <= ndc[0] <= 1.1 and -1.1 <= ndc[1] <= 1.1 and ndc[2] > 0): return None
        return (ndc[0]+1)*0.5*WIDTH, (1-ndc[1])*0.5*HEIGHT

    def _pick_body(self, mx, my, proj, view):
        best_idx, best_dist = None, PICK_THRESHOLD_PX
        for i, body in enumerate(self.physics.bodies):
            sc = self._world_to_screen((body.pos / SCALE).astype('f4'), proj, view)
            if sc is None: continue
            d = math.hypot(sc[0]-mx, sc[1]-my)
            if d < best_dist: best_dist, best_idx = d, i
        return best_idx

    def _focus_body(self, idx):
        body = self.physics.bodies[idx]
        self.tracking_index = idx
        self.camera_radius  = max(body.radius * 8.0, 0.3)
        print(f"Focused: {body.name}")

    def _unfocus(self):
        self.tracking_index = None
        print("Focus released")

    def _barycenter(self):
        if not self.physics.bodies:
            return np.zeros(3, dtype='f4')
        M   = sum(b.mass for b in self.physics.bodies)
        com = sum(b.mass * b.pos for b in self.physics.bodies) / M
        return (com / SCALE).astype('f4')

    # ------------------------------------------------------------------
    def on_key_event(self, key, action, modifiers):
        K = self.wnd.keys
        if action != K.ACTION_PRESS: return
        if key == K.UP:
            self.time_step_multi = min(self.time_step_multi * 2.0, 1024.0)
            self._clear_all_trails()
        if key == K.DOWN:
            self.time_step_multi = max(self.time_step_multi / 2.0, 0.0625)
            self._clear_all_trails()
        if key == K.SPACE: self.is_paused = not self.is_paused
        for k in SCENARIOS:
            if key == getattr(K, f"NUMBER_{k}", None): self.load_scenario(k)
        if key == K.Q: self.camera_radius = max(0.1, self.camera_radius * 0.5)
        if key == K.E: self.camera_radius = min(500.0, self.camera_radius / 0.5)
        if key == K.R:
            self.camera_target  = np.array([0.0,0.0,0.0], dtype='f4')
            self.camera_radius  = 5.0
            self.tracking_index = None
        if key == K.T:
            n = len(self.physics.bodies)
            if n == 0: return
            self.tracking_index = 0 if self.tracking_index is None \
                else (self.tracking_index + 1) % n
            if self.tracking_index == 0: self.tracking_index = None
        if key == K.F:
            if self.selected_index is not None and \
               self.selected_index < len(self.physics.bodies):
                if self.tracking_index == self.selected_index: self._unfocus()
                else: self._focus_body(self.selected_index)
            elif self.tracking_index is not None:
                self._unfocus()
        if key == K.B:
            self.use_barycentric_camera = not self.use_barycentric_camera
            print("Barycentric camera:", self.use_barycentric_camera)

    def on_mouse_press_event(self, x, y, button):
        if button != 1: return
        proj, view, _ = self._get_matrices()
        self.selected_index = self._pick_body(x, y, proj, view)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.camera_yaw   += dx * 0.005
        self.camera_pitch  = np.clip(self.camera_pitch + dy * 0.005, -1.5, 1.5)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera_radius *= 0.9 ** y_offset
        self.camera_radius  = np.clip(self.camera_radius, 0.1, 500.0)

    # ------------------------------------------------------------------
    def _update_trail_gpu(self, body: Body):
        if len(body.trail) < 2: return
        trail_data = np.array(list(body.trail), dtype='f4')
        n = len(trail_data)
        alphas = np.linspace(0.0, 1.65, n, dtype='f4')
        r, g, b, _ = body.color
        colors = np.column_stack([
            np.full(n, r, dtype='f4'),
            np.full(n, g, dtype='f4'),
            np.full(n, b, dtype='f4'),
            alphas,
        ])
        interleaved = np.hstack([trail_data, colors]).astype('f4')
        needed = interleaved.nbytes
        if body.trail_vbo is None or body.trail_vbo.size != needed:
            body.release_gpu()
            body.trail_vbo = self.ctx.buffer(interleaved)
            body.trail_vao = self.ctx.vertex_array(
                self.trail_prog,
                [(body.trail_vbo, '3f 4f', 'in_pos', 'in_color')])
        else:
            body.trail_vbo.write(interleaved)

    def _append_trail(self, body: Body, gl_pos: np.ndarray):
        if body.trail:
            jump = float(np.linalg.norm(gl_pos - body.trail[-1]))
            if jump > TRAIL_MAX_JUMP:
                body.clear_trail()
        body.trail.append(gl_pos.copy())

    # ------------------------------------------------------------------
    def _draw_sphere(self, gl_pos, radius, rotation, axial_tilt,
                     color, is_star, eye, light_positions, time):
        model      = self._model_matrix(gl_pos, radius, rotation, axial_tilt)
        normal_mat = self._normal_matrix(model)
        self.sphere_prog['model'].write(model)
        self.sphere_prog['normal_mat'].write(normal_mat)
        self.sphere_prog['body_color'].value = color
        self.sphere_prog['is_star'].value    = is_star
        self.sphere_prog['eye_pos'].value    = tuple(eye)
        self.sphere_prog['time'].value       = time
        n_lights = min(len(light_positions), MAX_LIGHTS)
        self.sphere_prog['num_lights'].value = n_lights
        padded = np.zeros((MAX_LIGHTS, 3), dtype='f4')
        for idx, lp in enumerate(light_positions[:MAX_LIGHTS]):
            padded[idx] = lp
        self.sphere_prog['light_pos'].write(np.ascontiguousarray(padded))
        self.sphere_vao.render(moderngl.TRIANGLES)

    def _draw_rings(self, gl_pos, radius, rotation, proj, view):
        """Draw Saturn-style rings around a body."""
        # Rings sit in the XZ plane, scaled by body radius, slightly tilted for looks
        tilt = math.radians(27.0)  # Saturn's ring tilt
        T  = matrix44.create_from_translation(gl_pos.astype('f4'), dtype='f4')
        Rx = matrix44.create_from_x_rotation(tilt, dtype='f4')
        Ry = matrix44.create_from_y_rotation(rotation * 0.3, dtype='f4')
        S  = matrix44.create_from_scale([radius]*3, dtype='f4')
        model = S @ Rx @ Ry @ T
        self.ring_prog['projection'].write(proj)
        self.ring_prog['view'].write(view)
        self.ring_prog['model'].write(model)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ring_vao.render(moderngl.TRIANGLES)

    # ----------------------------------------------------------------
    def on_render(self, time: float, frametime: float):
        self.ctx.clear(*BG_COLOR)

        if not self.is_paused:
            dt = SIM_SPEED * self.time_step_multi
            self.physics.step(dt)
            self.validate_indices()
            self.sim_elapsed_days += dt / DAY

        if self.tracking_index is not None and \
           self.tracking_index < len(self.physics.bodies):
            b = self.physics.bodies[self.tracking_index]
            target_gl = (b.pos / SCALE).astype('f4')
            self.camera_target += (target_gl - self.camera_target) * 0.12

        proj, view, eye = self._get_matrices()

        light_positions = [
            (b.pos / SCALE).astype('f4')
            for b in self.physics.bodies if b.is_star
        ]

        # ── Starfield (drawn first, no depth write) ───────────────────
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.star_prog['projection'].write(proj)
        self.star_prog['view'].write(view)
        self.starfield_vao.render(moderngl.POINTS)

        # ── 3-D scene ─────────────────────────────────────────────────
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.sphere_prog['projection'].write(proj)
        self.sphere_prog['view'].write(view)
        self.grid_prog['projection'].write(proj)
        self.grid_prog['view'].write(view)

        self._update_grid()
        self.grid_prog['color'].value = GRID_COLOR
        self.grid_vao.render(moderngl.LINES)

        # ── Fading trails ─────────────────────────────────────────────    
        self.trail_prog['projection'].write(proj)
        self.trail_prog['view'].write(view)
        self.trail_prog['line_width'].value  = TRAIL_WIDTH
        self.trail_prog['viewport'].value    = (float(WIDTH), float(HEIGHT))

        self.ctx.depth_mask = False          # ← read depth but don't write it

        for body in self.physics.bodies:
            gl_pos = (body.pos / SCALE).astype('f4')
            self._trail_tick += 1
            if self._trail_tick % 3 == 0:
                self._append_trail(body, gl_pos)
            self._update_trail_gpu(body)
            if body.trail_vao:
                body.trail_vao.render(moderngl.LINE_STRIP)

        self.ctx.depth_mask = True           # ← restore for spheres/rings
        # ── Star glows ────────────────────────────────────────────────
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.glow_prog['projection'].write(proj)
        self.glow_prog['view'].write(view)
        for body in self.physics.bodies:
            if not body.is_star: continue
            gl_pos = (body.pos / SCALE).astype('f4')
            self.glow_prog['star_pos'].value   = tuple(gl_pos)
            self.glow_prog['glow_size'].value  = body.radius * 5.5
            r, g, b_c, _ = body.color
            self.glow_prog['glow_color'].value = (r, g, b_c, 1.0)
            self.glow_vao.render(moderngl.TRIANGLES)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.DEPTH_TEST)

        # ── Spheres ───────────────────────────────────────────────────
        for i, body in enumerate(self.physics.bodies):
            gl_pos = (body.pos / SCALE).astype('f4')
            if i == self.selected_index:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = 'front'
                self._draw_sphere(gl_pos, body.radius * 1.09, body.rotation,
                                  body.axial_tilt, (1.0,1.0,1.0,1.0),
                                  True, eye, light_positions, time)
                self.ctx.cull_face = 'back'
                self.ctx.disable(moderngl.CULL_FACE)
            self._draw_sphere(gl_pos, body.radius, body.rotation,
                              body.axial_tilt, body.color, body.is_star,
                              eye, light_positions, time)

        # ── Rings (drawn after spheres, alpha blended) ────────────────
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.CULL_FACE)
        for body in self.physics.bodies:
            if not body.has_rings: continue
            gl_pos = (body.pos / SCALE).astype('f4')
            self._draw_rings(gl_pos, body.radius, body.rotation, proj, view)

        # ── 2-D overlay ───────────────────────────────────────────────
        self.ctx.disable(moderngl.DEPTH_TEST)
        focus_name = self.physics.bodies[self.tracking_index].name \
                     if self.tracking_index is not None else None
        self.overlay.draw(
            build_stats_panel(self.sim_elapsed_days, self.time_step_multi,
                              len(self.physics.bodies), self.is_paused, focus_name),
            PANEL_MARGIN, PANEL_MARGIN)
        if self.selected_index is not None and \
           self.selected_index < len(self.physics.bodies):
            is_focused = (self.tracking_index == self.selected_index)
            self.overlay.draw(
                build_inspect_panel(self.physics.bodies[self.selected_index], is_focused),
                WIDTH - PANEL_W - PANEL_MARGIN, HEIGHT - PANEL_H - PANEL_MARGIN)
        else:
            hint = build_hint_panel()
            self.overlay.draw(hint,
                WIDTH - hint.width - PANEL_MARGIN,
                HEIGHT - hint.height - PANEL_MARGIN)
        self.ctx.enable(moderngl.DEPTH_TEST)

# ============================================================
if __name__ == '__main__':
    mglw.run_window_config(GravitySim)