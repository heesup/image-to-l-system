import math
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional
from .lsystem import LSystem

class TurtleRenderer:
    """Fast 2D Turtle Graphics renderer for L-System plant structures."""

    def __init__(self, image_size: Tuple[int, int] = (256, 256), margin: int = 20, bg_color: str = "white", fg_color: str = "forestgreen"):
        self.width, self.height = image_size
        self.margin = margin
        self.bg_color = bg_color
        self.fg_color = fg_color

    def compute_bounds(self, expanded_str: str, angle: float, step_size: float) -> Tuple[float, float, float, float]:
        """First pass to compute bounding box (min_x, min_y, max_x, max_y) of drawn segments."""
        x, y = 0.0, 0.0
        heading = 90.0  # Pointing upwards
        angle_rad = math.radians(angle)
        
        stack = []
        min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0

        for char in expanded_str:
            if char in ('F', 'G'):
                rad = math.radians(heading)
                nx = x + step_size * math.cos(rad)
                ny = y + step_size * math.sin(rad)
                min_x = min(min_x, nx)
                max_x = max(max_x, nx)
                min_y = min(min_y, ny)
                max_y = max(max_y, ny)
                x, y = nx, ny
            elif char == 'f':
                rad = math.radians(heading)
                x += step_size * math.cos(rad)
                y += step_size * math.sin(rad)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            elif char == '+':
                heading -= angle
            elif char == '-':
                heading += angle
            elif char == '[':
                stack.append((x, y, heading))
            elif char == ']':
                if stack:
                    x, y, heading = stack.pop()

        return min_x, min_y, max_x, max_y

    def render(self, lsystem: LSystem, return_tensor: bool = False) -> Image.Image:
        """Render L-System into a PIL Image auto-centered on canvas."""
        expanded = lsystem.expand()
        min_x, min_y, max_x, max_y = self.compute_bounds(expanded, lsystem.angle, lsystem.step_size)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        canvas_w = self.width - 2 * self.margin
        canvas_h = self.height - 2 * self.margin

        scale_x = canvas_w / bbox_w if bbox_w > 1e-5 else 1.0
        scale_y = canvas_h / bbox_h if bbox_h > 1e-5 else 1.0
        scale = min(scale_x, scale_y)

        # Center on canvas
        offset_x = self.margin + (canvas_w - bbox_w * scale) / 2.0 - min_x * scale
        offset_y = self.margin + (canvas_h - bbox_h * scale) / 2.0 - min_y * scale

        image = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(image)

        x, y = 0.0, 0.0
        heading = 90.0  # Upwards
        stack = []

        line_width = max(1, int(lsystem.line_width))

        for char in expanded:
            if char in ('F', 'G'):
                rad = math.radians(heading)
                nx = x + lsystem.step_size * math.cos(rad)
                ny = y + lsystem.step_size * math.sin(rad)

                # Transform to canvas coords (flip Y axis so up is up)
                px1 = x * scale + offset_x
                py1 = self.height - (y * scale + offset_y)
                px2 = nx * scale + offset_x
                py2 = self.height - (ny * scale + offset_y)

                draw.line([(px1, py1), (px2, py2)], fill=self.fg_color, width=line_width)
                x, y = nx, ny
            elif char == 'f':
                rad = math.radians(heading)
                x += lsystem.step_size * math.cos(rad)
                y += lsystem.step_size * math.sin(rad)
            elif char == '+':
                heading -= lsystem.angle
            elif char == '-':
                heading += lsystem.angle
            elif char == '[':
                stack.append((x, y, heading))
            elif char == ']':
                if stack:
                    x, y, heading = stack.pop()

        return image
