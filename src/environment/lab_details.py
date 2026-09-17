"""Lightweight neutral lab architecture used only in the interactive arena."""


def lab_details_xml():
    parts = ['<camera name="gallery_view" pos="7 -9 11" xyaxes="0.789 0.614 0 -0.47 0.605 0.644" fovy="48"/>']
    # Raised wall ribs, inset panels and light strips: neutral colours preserve
    # the warm-colour food task. Wall-adjacent objects are physical.
    for side in (-1, 1):
        for index, coord in enumerate((-3, -1.5, 0, 1.5, 3)):
            for axis in (0, 1):
                x, y = (side*3.76, coord) if axis == 0 else (coord, side*3.76)
                size = '.035 .045 .8' if axis == 0 else '.045 .035 .8'
                parts.append(f'<geom name="lab_rib_{side}_{axis}_{index}" type="box" pos="{x} {y} .85" size="{size}" rgba=".21 .21 .21 1"/>')
        for axis in (0, 1):
            x, y = (side*3.74, 0) if axis == 0 else (0, side*3.74)
            size = '.018 3.55 .025' if axis == 0 else '3.55 .018 .025'
            parts.append(f'<geom name="lab_light_{side}_{axis}" type="box" pos="{x} {y} 1.8" size="{size}" rgba=".85 .85 .85 1"/>')
    # Inlaid runway markers below flight height and decorative island rings.
    for i in range(-6, 7):
        for y in (-1.4, 1.4):
            parts.append(f'<geom type="box" pos="{i*.5} {y} .012" size=".12 .025 .008" rgba=".58 .58 .58 1" contype="0" conaffinity="0" group="2"/>')
    for x, y in ((-2.55, 2.25), (2.45, -2.2)):
        parts.append(f'<geom type="cylinder" pos="{x} {y} .44" size=".27 .025" rgba=".55 .55 .55 1"/>')
        parts.append(f'<geom type="cylinder" pos="{x} {y} .53" size=".16 .065" rgba=".24 .24 .24 1"/>')
    return '\n'.join(parts)
