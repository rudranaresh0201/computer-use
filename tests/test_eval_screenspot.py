"""ScreenSpot loading, image sizing and the report table (eval/screenspot.py).
The model-running part is exercised by the CPU dry run of the notebook."""

import pytest

from computeruse.eval import screenspot
from computeruse.eval.vlm_grounder import ArmResult


def row(i, source="windows", dtype="icon", bbox=(0.94, 0.14, 0.99, 0.2)):
    return {"file_name": f"pc_{i}.png", "bbox": list(bbox), "instruction": "close",
            "data_type": dtype, "data_source": source, "image": {"bytes": b"png", "path": None}}


def test_rows_become_items_in_our_coordinate_space():
    (item,) = screenspot.items_from_rows([row(7)], start_index=12)
    assert item.id == "0012_pc_7"
    assert item.bbox_norm == (940, 140, 990, 200)
    assert (item.platform, item.slice) == ("desktop", "desktop/icon")


def test_every_source_maps_to_one_of_the_three_platforms():
    assert set(screenspot.PLATFORM_OF.values()) == set(screenspot.PLATFORMS)
    assert screenspot.items_from_rows([row(0, source="android", dtype="text")])[0].slice == "mobile/text"
    assert screenspot.items_from_rows([row(0, source="gitlab")])[0].slice == "web/icon"


def test_a_bbox_that_is_not_normalized_xyxy_is_refused():
    with pytest.raises(ValueError, match="not normalized"):
        screenspot.items_from_rows([row(0, bbox=(100, 50, 200, 80))])
    with pytest.raises(ValueError, match="not normalized"):
        screenspot.items_from_rows([row(0, bbox=(0.5, 0.5, 0.2, 0.9))])  # x2 < x1: that's (x, y, w, h)


def test_big_images_shrink_to_the_training_budget_small_ones_stay():
    from PIL import Image

    phone = screenspot.fit_to_budget(Image.new("RGB", (1170, 2532)))
    assert phone.size[0] * phone.size[1] <= screenspot.TRAINING_PIXEL_BUDGET
    assert abs(phone.size[0] / phone.size[1] - 1170 / 2532) < 0.01
    small = Image.new("RGB", (960, 540))
    assert screenspot.fit_to_budget(small) is small


def result(id_, slice_, hit):
    return ArmResult(id_, "windows", slice_, "(1,1)", (1, 1), (0, 0, 2, 2), hit, 1.0)


def test_report_has_the_usual_table_and_paired_differences():
    zero = [result("a", "desktop/text", False), result("b", "desktop/icon", False), result("c", "web/text", True)]
    tuned = [result("a", "desktop/text", True), result("b", "desktop/icon", True), result("c", "web/text", True)]
    out = screenspot.report({"zero_shot": zero, "fine_tuned": tuned})
    table = out["arms"]["fine_tuned"]
    assert table["desktop"]["accuracy"] == 1.0 and table["desktop"]["n"] == 2
    assert table["overall"]["n"] == 3
    assert table["mobile"]["n"] == 0 and table["mobile"]["accuracy"] is None
    assert out["vs_zero_shot"]["fine_tuned"]["desktop"]["diff"] == 1.0
    assert out["vs_zero_shot"]["fine_tuned"]["web"]["diff"] == 0.0


def test_v2_pixel_xywh_boxes_become_our_coordinates_and_platform_comes_from_the_file():
    # the first desktop row of the real screenspot_desktop_v2.json, on its 960x540 image
    rows = [{"img_filename": "pc_ede36f9b.png", "bbox": [910, 78, 44, 34], "instruction": "close this window",
             "data_type": "icon", "data_source": "shop"}]
    items = screenspot.items_from_v2("desktop", rows, lambda f: (960, 540), lambda f: b"png", start_index=7)
    item = items[0]
    assert item.bbox_norm == (948, 144, 994, 207)
    assert item.platform == "desktop" and item.slice == "desktop/icon"  # not PLATFORM_OF["shop"] == "web"
    assert item.id == "v2_0007_pc_ede36f9b"


def test_v2_box_outside_its_image_is_refused():
    rows = [{"img_filename": "a.png", "bbox": [900, 10, 200, 10], "instruction": "x", "data_type": "text", "data_source": "ios"}]
    with pytest.raises(ValueError, match="outside"):
        screenspot.items_from_v2("mobile", rows, lambda f: (960, 540), lambda f: b"")
