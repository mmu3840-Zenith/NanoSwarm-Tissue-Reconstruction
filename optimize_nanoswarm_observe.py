from pathlib import Path

path = Path("nanoswarm_v2/__main__.py")
source = path.read_text(encoding="utf-8")

old = """    def observe(self, x, y, include_pheromone=False):
        # Local 3x3 observation; out-of-bounds cells are zero.
        def patch(grid):
            out = np.zeros((3, 3), dtype=np.float32)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < self.size and 0 <= yy < self.size:
                        out[dy + 1, dx + 1] = grid[yy, xx]
            return out.ravel()
        obs = patch(self.damage)
        if include_pheromone:
            obs = np.concatenate((obs, patch(self.pheromone)))
        return obs
"""

new = """    def observe(self, x, y, include_pheromone=False):
        # Vectorized 3x3 observation; out-of-bounds cells remain zero.
        def patch(grid):
            out = np.zeros((3, 3), dtype=np.float32)

            x0 = max(0, x - 1)
            x1 = min(self.size, x + 2)
            y0 = max(0, y - 1)
            y1 = min(self.size, y + 2)

            if x0 < x1 and y0 < y1:
                ox0 = x0 - (x - 1)
                ox1 = ox0 + (x1 - x0)
                oy0 = y0 - (y - 1)
                oy1 = oy0 + (y1 - y0)
                out[oy0:oy1, ox0:ox1] = grid[y0:y1, x0:x1]

            return out.ravel()

        obs = patch(self.damage)
        if include_pheromone:
            obs = np.concatenate((obs, patch(self.pheromone)))
        return obs
"""

if old not in source:
    raise SystemExit("Expected observe() implementation not found. No source changes made.")

path.write_text(source.replace(old, new, 1), encoding="utf-8")
print("Optimized Tissue.observe() successfully.")
