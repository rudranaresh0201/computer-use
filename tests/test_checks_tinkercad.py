"""The Tinkercad checker's geometry, on STL files built here: a box, a cylinder resting on it, one sunk
into it, and one beside it. The real export was checked by hand: a scripted drop that looked stacked
from the camera was sunk 10 mm into the box, and the checker said so."""

import math
import struct

from computeruse.agent.checks import read_stl, solids, stacked_verdict


def box(x0, y0, z0, w, d, h):
    p = [(x0 + dx * w, y0 + dy * d, z0 + dz * h) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [t for a, b, c, d in faces for t in ((p[a], p[b], p[c]), (p[a], p[c], p[d]))]


def cylinder(cx, cy, z0, r, h, sides=24):
    ring = [(cx + r * math.cos(2 * math.pi * i / sides), cy + r * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    tris = []
    for i in range(sides):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % sides]
        tris += [((cx, cy, z0), (x2, y2, z0), (x1, y1, z0)), ((cx, cy, z0 + h), (x1, y1, z0 + h), (x2, y2, z0 + h)),
                 ((x1, y1, z0), (x2, y2, z0), (x2, y2, z0 + h)), ((x1, y1, z0), (x2, y2, z0 + h), (x1, y1, z0 + h))]
    return tris


def binary_stl(tris):
    out = b"\0" * 80 + struct.pack("<I", len(tris))
    for t in tris:
        out += struct.pack("<12f", 0, 0, 0, *t[0], *t[1], *t[2]) + b"\0\0"
    return out


def test_binary_round_trip_and_two_separate_solids_that_touch():
    tris = box(0, 0, 0, 20, 20, 20) + cylinder(10, 10, 20, 8, 20)
    parts = solids(read_stl(binary_stl(tris)))
    assert [p["triangles"] for p in parts] == [12, 96]
    assert stacked_verdict(parts)[0]


def test_a_cylinder_sunk_into_the_box_is_not_stacked():
    ok, detail = stacked_verdict(solids(box(0, 0, 0, 20, 20, 20) + cylinder(10, 10, 10, 8, 20)))
    assert not ok and "z 10.0 to 30.0" in detail


def test_a_cylinder_beside_the_box_is_not_stacked():
    assert not stacked_verdict(solids(box(0, 0, 0, 20, 20, 20) + cylinder(60, 10, 20, 8, 20)))[0]


def test_ascii_stl_and_an_empty_design():
    tris = box(0, 0, 0, 10, 10, 10)
    ascii_ = "solid t\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
                                   + "endloop\nendfacet\n" for t in tris) + "endsolid t\n"
    assert len(read_stl(ascii_.encode())) == 12
    assert stacked_verdict([]) == (False, "0 solids, nothing round stands on a box: empty design")
