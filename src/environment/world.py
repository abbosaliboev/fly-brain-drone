"""Assembles the MJCF scene: ground, walls, visible food, and drone."""
from __future__ import annotations

import mujoco

from src.drone.drone import drone_xml, DRONE_ACTUATORS_XML

# Phase 6: a simple bounded square arena. Real MuJoCo collision geometry --
# the drone can physically hit these, not just a signal used for the vision
# pathway (src/vision/encoder.py's encode_looming_from_walls uses this same
# ARENA_HALF_SIZE so the neural avoidance signal and the physical walls agree).
ARENA_HALF_SIZE = 4.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.2
FOOD_POSITION = (3.0, 3.0, 1.0)
FOOD_RADIUS = 0.4

# solref/solimp soften the contact response considerably (MuJoCo defaults
# are tuned for rigid contact, which combined with our external-force-based
# forward locomotion occasionally produced a violent, numerically unstable
# collision response -- the drone briefly launching hundreds of meters
# into the air, per Drone.reset_if_unstable's docstring). This is a
# mitigation, not a proven fix for the root cause.
_WALL_CONTACT = 'solref="0.05 1.5" solimp="0.8 0.95 0.01"'

WALLS_XML = f"""
<geom name="wall_east" type="box" pos="{ARENA_HALF_SIZE} 0 {WALL_HEIGHT/2}" size="{WALL_THICKNESS} {ARENA_HALF_SIZE} {WALL_HEIGHT/2}" rgba="0.5 0.4 0.35 1" {_WALL_CONTACT}/>
<geom name="wall_west" type="box" pos="{-ARENA_HALF_SIZE} 0 {WALL_HEIGHT/2}" size="{WALL_THICKNESS} {ARENA_HALF_SIZE} {WALL_HEIGHT/2}" rgba="0.5 0.4 0.35 1" {_WALL_CONTACT}/>
<geom name="wall_north" type="box" pos="0 {ARENA_HALF_SIZE} {WALL_HEIGHT/2}" size="{ARENA_HALF_SIZE} {WALL_THICKNESS} {WALL_HEIGHT/2}" rgba="0.5 0.4 0.35 1" {_WALL_CONTACT}/>
<geom name="wall_south" type="box" pos="0 {-ARENA_HALF_SIZE} {WALL_HEIGHT/2}" size="{ARENA_HALF_SIZE} {WALL_THICKNESS} {WALL_HEIGHT/2}" rgba="0.5 0.4 0.35 1" {_WALL_CONTACT}/>
"""

def food_xml(position=FOOD_POSITION, rgba=(1.0, 0.35, 0.02, 1.0)) -> str:
    x, y, z = position
    return f"""
<geom name="food" group="1" type="ellipsoid" pos="{x} {y} {z}" size="0.38 0.36 0.40"
      rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0"/>
<geom name="food_lobe" group="1" type="ellipsoid" pos="{x + 0.10} {y} {z + 0.03}" size="0.31 0.32 0.34"
      rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0"/>
<geom name="food_stem" group="1" type="cylinder" pos="{x} {y} {z + 0.43}" size="0.035 0.13"
      rgba="0.24 0.12 0.04 1" contype="0" conaffinity="0"/>
<geom name="food_leaf" group="1" type="ellipsoid" pos="{x + 0.13} {y} {z + 0.48}" size="0.14 0.055 0.018"
      quat="0.9239 0 0.3827 0" rgba="0.10 0.48 0.20 1" contype="0" conaffinity="0"/>
"""


def distractors_xml(distractors=()) -> str:
    geoms = []
    for index, (position, rgba) in enumerate(distractors):
        geoms.append(
            f'<geom name="distractor_{index}" type="sphere" '
            f'pos="{position[0]} {position[1]} {position[2]}" size="{FOOD_RADIUS}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" '
            'contype="0" conaffinity="0"/>'
        )
    return "\n".join(geoms)

WORLD_XML_TEMPLATE = """
<mujoco model="fly_brain_drone_world">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.11 0.14 0.17" rgb2="0.18 0.21 0.24"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="8 8" reflectance="0.12"/>
  </asset>
  <visual>
    <quality shadowsize="256"/>
    <headlight ambient="{ambient} {ambient} {ambient}"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="{diffuse} {diffuse} {diffuse}"/>
    <light pos="-3 -3 3" dir="1 1 -0.7" diffuse="0.28 0.32 0.38"/>
    <geom name="ground" type="plane" size="20 20 0.1" material="floor_mat" rgba="0.85 0.88 0.92 1"/>
    <camera name="world_view" pos="6 -7 5.5" xyaxes="0.75 0.65 0 -0.35 0.40 0.85" fovy="52"/>
    <geom name="room_trim_north" type="box" pos="0 3.78 0.12" size="3.75 0.035 0.06" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
    <geom name="room_trim_south" type="box" pos="0 -3.78 0.12" size="3.75 0.035 0.06" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
    <geom name="room_trim_east" type="box" pos="3.78 0 0.12" size="0.035 3.75 0.06" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
    <geom name="room_trim_west" type="box" pos="-3.78 0 0.12" size="0.035 3.75 0.06" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
    {challenge}
    {walls}
    {food}
    {distractors}
    {drone}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""

CHALLENGE_ARENA_XML = """
<geom name="platform_nw" type="box" pos="-2.55 2.25 0.16" size="0.65 0.55 0.16" rgba="0.18 0.23 0.27 1"/>
<geom name="platform_se" type="box" pos="2.45 -2.20 0.20" size="0.75 0.62 0.20" rgba="0.16 0.21 0.25 1"/>
<geom name="island_center" type="cylinder" pos="0 0 0.14" size="0.72 0.14" rgba="0.21 0.25 0.28 1"/>
<geom name="pylon_left" type="cylinder" pos="-2.25 -0.70 0.78" size="0.14 0.78" rgba="0.25 0.29 0.32 1"/>
<geom name="pylon_right" type="cylinder" pos="1.85 1.20 0.78" size="0.14 0.78" rgba="0.25 0.29 0.32 1"/>
<geom name="arch_left" type="box" pos="-1.45 -2.35 1.05" size="0.10 0.10 1.05" rgba="0.22 0.27 0.31 1"/>
<geom name="arch_right" type="box" pos="0.15 -2.35 1.05" size="0.10 0.10 1.05" rgba="0.22 0.27 0.31 1"/>
<geom name="arch_top" type="box" pos="-0.65 -2.35 2.05" size="0.90 0.10 0.10" rgba="0.22 0.27 0.31 1"/>
<geom name="beacon_left" type="sphere" pos="-1.45 -2.35 2.18" size="0.07" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
<geom name="beacon_right" type="sphere" pos="0.15 -2.35 2.18" size="0.07" rgba="0.12 0.55 0.72 1" contype="0" conaffinity="0"/>
"""


def build_model(drone_start_pos=(0.0, 0.0, 1.0), drone_start_yaw_deg: float = 0.0,
                food_position=FOOD_POSITION,
                food_rgba=(1.0, 0.35, 0.02, 1.0),
                achromatic_background: bool = False,
                distractors=(), light_level: float = 1.0,
                challenge_arena: bool = False, game_assets: bool = False) -> mujoco.MjModel:
    if light_level < 0.0:
        raise ValueError("light_level must be non-negative")
    xml = WORLD_XML_TEMPLATE.format(
        drone=drone_xml(drone_start_pos, drone_start_yaw_deg), actuators=DRONE_ACTUATORS_XML,
        walls=WALLS_XML, food=food_xml(food_position, food_rgba),
        distractors=distractors_xml(distractors),
        challenge=CHALLENGE_ARENA_XML if challenge_arena else "",
        ambient=0.4 * light_level, diffuse=0.8 * light_level,
    )
    if game_assets:
        # One free body and a small static room need little contact workspace.
        # Bound compiler/runtime scratch allocation on memory-limited Windows.
        xml = xml.replace('<option gravity=', '<size memory="8M"/>\n  <option gravity=', 1)
        from src.environment.lab_details import lab_details_xml
        extras = '''
<geom name="hard_pillar_a" type="cylinder" pos="-1 1 0.8" size="0.18 0.8" rgba="0.3 0.3 0.3 1"/>
<geom name="hard_pillar_b" type="cylinder" pos="1 -1 0.8" size="0.18 0.8" rgba="0.3 0.3 0.3 1"/>
<geom name="hard_distractor_a" group="1" type="sphere" pos="-2.5 -2.7 1" size="0.22" rgba="0.8 0.25 0.03 1" contype="0" conaffinity="0"/>
<geom name="hard_distractor_b" group="1" type="sphere" pos="2.7 0 1" size="0.22" rgba="0.1 0.6 0.2 1" contype="0" conaffinity="0"/>
'''
        xml = xml.replace('</worldbody>', extras + lab_details_xml() + '</worldbody>')
        xml = xml.replace('shadowsize="256"', 'shadowsize="512"')
        xml = xml.replace('offwidth="640" offheight="480"', 'offwidth="320" offheight="240"')
    if achromatic_background:
        xml = xml.replace('rgba="0.25 0.3 0.28 1"', 'rgba="0.28 0.28 0.28 1"')
        xml = xml.replace('rgba="0.5 0.4 0.35 1"', 'rgba="0.4 0.4 0.4 1"')
        # Keep the polished geometry while removing its colour signal.
        xml = xml.replace(
            'rgb1="0.11 0.14 0.17" rgb2="0.18 0.21 0.24"',
            'rgb1="0.12 0.12 0.12" rgb2="0.21 0.21 0.21"',
        )
        xml = xml.replace('rgba="0.12 0.55 0.72 1"', 'rgba="0.34 0.34 0.34 1"')
        for colour, grey in (
            ("0.18 0.23 0.27", "0.23 0.23 0.23"),
            ("0.16 0.21 0.25", "0.21 0.21 0.21"),
            ("0.21 0.25 0.28", "0.25 0.25 0.25"),
            ("0.25 0.29 0.32", "0.29 0.29 0.29"),
            ("0.22 0.27 0.31", "0.27 0.27 0.27"),
        ):
            xml = xml.replace(f'rgba="{colour} 1"', f'rgba="{grey} 1"')
    return mujoco.MjModel.from_xml_string(xml)
