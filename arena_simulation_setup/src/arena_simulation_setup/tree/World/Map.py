import io
import itertools
import math
import typing
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.tree import PathView


class Map(PathView):
    @property
    def map_yaml(self) -> Path:
        return self.path / 'map.yaml'

    @property
    def map_png(self) -> Path:
        return self.path / 'map.png'

    @classmethod
    def generate_png(
        cls,
        rooms: shapely.MultiPolygon,
        doors: shapely.MultiPolygon,
        walls: shapely.MultiLineString,
        resolution: float = 0.01,
        padding: int = 5,
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Generate a PNG image of the map with the given elements.
        """
        min_x, min_y, max_x, max_y = rooms.bounds

        width = max_x - min_x
        height = max_y - min_y

        img = PIL.Image.new(
            'RGB',
            (
                math.ceil(width / resolution) + 2 * padding,
                math.ceil(height / resolution) + 2 * padding,
            ),
            color='black'
        )

        scaling_factor = 1 / resolution

        def tf(shape):
            shape = shapely.affinity.translate(shape, -min_x, -min_y)
            shape = shapely.affinity.scale(shape, scaling_factor, -scaling_factor, origin=(0, 0))  # type: ignore
            shape = shapely.affinity.translate(shape, 0, height * scaling_factor)
            shape = shapely.set_precision(shape, 0.01)
            shape = shapely.make_valid(shape)
            shape = shapely.remove_repeated_points(shape)
            return shape

        def as_int(coords):
            return [(int(math.trunc(x) + padding), int(math.trunc(y) + padding)) for (x, y, *_) in coords]

        draw = PIL.ImageDraw.Draw(img)
        for cutout in itertools.chain(rooms.geoms, doors.geoms):
            poly = tf(shapely.Polygon(cutout))
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        for wall in walls.geoms:
            line = tf(shapely.LineString(wall))
            draw.line(as_int(line.coords), fill='black', width=1)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue(), (min_x + padding * resolution, min_y + padding * resolution)

    @classmethod
    def generate_map_yaml(cls, resolution: float, filename: str, origin: tuple[float, float]) -> str:
        return typing.cast(
            str,
            yaml.safe_dump({
                'free_thresh': 0.1,
                'image': filename,
                'negate': 0,
                'occupied_thresh': 0.9,
                'origin': [*origin, 0],
                'resolution': resolution,
            })
        )
