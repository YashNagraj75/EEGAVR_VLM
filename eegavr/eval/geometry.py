from ..generator.export import _order
from ..generator.scene import PANEL_PX, UNIT_R, poly

DECIMALS = 3
LINE_DECIMALS = 4
PANEL_KEYS = ("line_width", "objects")
OBJECT_KEYS = ("shape", "fill", "vertices", "centre", "radius")


def _q(v, decimals):
    return float(v) if decimals is None else round(float(v), decimals)


def complete_object(o, decimals=DECIMALS):
    # Outline from the generator's own polygon, panel fractions (x right, y down); circles as centre + radius.
    e = {"shape": o["shape"], "fill": "filled" if o["fill"] else "outline"}
    if o["shape"] == "circle":
        e["centre"] = [_q(o["x"], decimals), _q(o["y"], decimals)]
        e["radius"] = _q(o["size"] * UNIT_R["circle"], decimals)
    else:
        e["vertices"] = [[_q(x, decimals), _q(y, decimals)] for x, y in list(poly(o).exterior.coords)[:-1]]
    return e


def complete_scene(scene, decimals=DECIMALS):
    # Display order is the hashed panel-id permutation of the exported symbolic list; the id itself is never shown.
    objs = scene["objects"]
    strokes = {round(float(o["stroke_px"]), 6) for o in objs}
    if len(strokes) > 1:
        raise ValueError(f"{scene.get('panel_id')}: stroke width differs between objects, which could encode roles")
    width = (strokes.pop() / PANEL_PX) if strokes else 0.0
    order = _order(scene["panel_id"], len(objs))
    return {"line_width": _q(width, LINE_DECIMALS if decimals is not None else None),
            "objects": [complete_object(objs[i], decimals) for i in order]}
