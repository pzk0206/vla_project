"""测试相机像素、世界坐标与工作平面之间的确定性几何转换。"""

import unittest

import numpy as np

from camera_geometry import (
    compute_camera_matrices,
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
    world_to_pixel,
)


CAMERA_CONFIG = {
    "workspace_center": [0.0, 0.4, 0.0],
    "up_vector": [0.0, 1.0, 0.0],
    "image_width": 448,
    "image_height": 448,
    "fov": 45,
    "near_val": 0.1,
    "far_val": 100.0,
}
CAMERA_EYE = [0.0, 0.4, 3.0]


class NormalizedBoxCenterTests(unittest.TestCase):
    def test_maps_normalized_box_center_to_float_pixel(self):
        pixel = normalized_box_center_to_pixel([400, 300, 600, 500], 448, 448)

        np.testing.assert_allclose(pixel, [223.5, 178.8], atol=1e-9)

    def test_rejects_invalid_normalized_box(self):
        invalid_boxes = (
            [600, 300, 400, 500],
            [-1, 300, 400, 500],
            [100, 200, 300],
        )
        for box in invalid_boxes:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    normalized_box_center_to_pixel(box, 448, 448)


class CameraBackprojectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view, cls.projection = compute_camera_matrices(
            CAMERA_CONFIG, CAMERA_EYE
        )

    def test_image_center_hits_workspace_center_on_ground_plane(self):
        world = pixel_to_world_on_plane(
            (223.5, 223.5), 448, 448, self.view, self.projection
        )

        np.testing.assert_allclose(world, [0.0, 0.4, 0.0], atol=1e-6)

    def test_world_pixel_world_round_trip(self):
        expected = [0.12, 0.51, 0.0]
        pixel = world_to_pixel(expected, 448, 448, self.view, self.projection)
        actual = pixel_to_world_on_plane(
            pixel, 448, 448, self.view, self.projection
        )

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_world_positive_y_projects_above_image_center(self):
        _, pixel_y = world_to_pixel(
            [0.0, 0.5, 0.0], 448, 448, self.view, self.projection
        )

        self.assertLess(pixel_y, 223.5)

    def test_rejects_singular_matrix(self):
        with self.assertRaisesRegex(ValueError, "不可逆"):
            pixel_to_world_on_plane(
                (223.5, 223.5), 448, 448, [0.0] * 16, self.projection
            )

    def test_rejects_parallel_ray(self):
        horizontal_config = dict(CAMERA_CONFIG)
        horizontal_config["workspace_center"] = [0.0, 1.4, 1.0]
        horizontal_config["up_vector"] = [0.0, 0.0, 1.0]
        horizontal_view, horizontal_projection = compute_camera_matrices(
            horizontal_config, [0.0, 0.4, 1.0]
        )

        with self.assertRaisesRegex(ValueError, "平行"):
            pixel_to_world_on_plane(
                (223.5, 223.5),
                448,
                448,
                horizontal_view,
                horizontal_projection,
            )

    def test_rejects_intersection_behind_camera_ray(self):
        with self.assertRaisesRegex(ValueError, "反方向"):
            pixel_to_world_on_plane(
                (223.5, 223.5),
                448,
                448,
                self.view,
                self.projection,
                plane_z=4.0,
            )

    def test_rejects_boolean_plane_height(self):
        with self.assertRaisesRegex(ValueError, "工作平面高度"):
            pixel_to_world_on_plane(
                (223.5, 223.5),
                448,
                448,
                self.view,
                self.projection,
                plane_z=True,
            )


if __name__ == "__main__":
    unittest.main()
