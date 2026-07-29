import math
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from .lsystem import LSystem
from .renderer import TurtleRenderer

class PlantGraphExtractor:
    """Extracts 2D Plant Organ Primitives (Stem Base x, y, angle theta, length l, width w),
    Adjacency Matrix A, and Existence Masks directly from L-System Turtle traces.
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 256), max_nodes: int = 64):
        self.image_size = image_size
        self.max_nodes = max_nodes
        self.renderer = TurtleRenderer(image_size=image_size)

    def extract_graph(self, lsystem: LSystem) -> Dict[str, Any]:
        """Extract organ primitive array (N, 5) [norm_x, norm_y, theta_rad, norm_length, norm_width],
        adjacency matrix (N, N), and existence mask (N,).
        """
        expanded = lsystem.expand()
        min_x, min_y, max_x, max_y = self.renderer.compute_bounds(expanded, lsystem.angle, lsystem.step_size)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        canvas_w = self.renderer.width - 2 * self.renderer.margin
        canvas_h = self.renderer.height - 2 * self.renderer.margin

        scale_x = canvas_w / bbox_w if bbox_w > 1e-5 else 1.0
        scale_y = canvas_h / bbox_h if bbox_h > 1e-5 else 1.0
        scale = min(scale_x, scale_y)

        offset_x = self.renderer.margin + (canvas_w - bbox_w * scale) / 2.0 - min_x * scale
        offset_y = self.renderer.margin + (canvas_h - bbox_h * scale) / 2.0 - min_y * scale

        def transform_pt(x_val: float, y_val: float) -> Tuple[float, float]:
            px = x_val * scale + offset_x
            py = self.renderer.height - (y_val * scale + offset_y)
            return px / float(self.renderer.width), py / float(self.renderer.height)

        # Track unique nodes & directed edges
        node_map: Dict[Tuple[int, int], int] = {}
        nodes: List[Tuple[float, float, float, float, float]] = []  # (norm_x, norm_y, theta, length, width)
        edges: List[Tuple[int, int]] = []

        norm_step_len = (lsystem.step_size * scale) / float(self.renderer.width)

        x, y = 0.0, 0.0
        heading = 90.0
        stack = []
        base_width = max(1.0, float(lsystem.line_width))
        curr_width = base_width

        def get_node_idx(curr_x: float, curr_y: float, curr_h: float, w_val: float) -> int:
            nx, ny = transform_pt(curr_x, curr_y)
            grid_key = (int(round(nx * 1000)), int(round(ny * 1000)))
            if grid_key in node_map:
                return node_map[grid_key]
            else:
                idx = len(nodes)
                norm_theta = (math.radians(curr_h) + math.pi) / (2.0 * math.pi)
                nodes.append((nx, ny, norm_theta, norm_step_len, w_val / 20.0))
                node_map[grid_key] = idx
                return idx

        parent_idx = get_node_idx(x, y, heading, curr_width)

        for char in expanded:
            if char in ('F', 'G', 'A', 'B'):
                rad = math.radians(heading)
                nx = x + lsystem.step_size * math.cos(rad)
                ny = y + lsystem.step_size * math.sin(rad)
                child_idx = get_node_idx(nx, ny, heading, curr_width)
                edges.append((parent_idx, child_idx))
                x, y = nx, ny
                parent_idx = child_idx
            elif char == 'f':
                rad = math.radians(heading)
                x += lsystem.step_size * math.cos(rad)
                y += lsystem.step_size * math.sin(rad)
                parent_idx = get_node_idx(x, y, heading, curr_width)
            elif char == '+':
                heading -= lsystem.angle
            elif char == '-':
                heading += lsystem.angle
            elif char == '[':
                stack.append((x, y, heading, parent_idx, curr_width))
                curr_width = max(1.0, curr_width * 0.75)
            elif char == ']':
                if stack:
                    x, y, heading, parent_idx, curr_width = stack.pop()

        num_nodes = min(len(nodes), self.max_nodes)
        node_attr = np.zeros((self.max_nodes, 5), dtype=np.float32)
        adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        existence_mask = np.zeros((self.max_nodes,), dtype=np.float32)

        for i in range(num_nodes):
            node_attr[i] = np.array(nodes[i], dtype=np.float32)
            existence_mask[i] = 1.0

        parent_indices = np.arange(self.max_nodes, dtype=np.int64)
        for u, v in edges:
            if u < self.max_nodes and v < self.max_nodes:
                adj_matrix[u, v] = 1.0
                adj_matrix[v, u] = 1.0
                parent_indices[v] = u

        return {
            "nodes": torch.from_numpy(node_attr),
            "adj_matrix": torch.from_numpy(adj_matrix),
            "parent_indices": torch.from_numpy(parent_indices),
            "existence_mask": torch.from_numpy(existence_mask),
            "num_nodes": num_nodes
        }
