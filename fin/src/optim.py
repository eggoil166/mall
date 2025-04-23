import numpy as np

class TileOptimizer:
    def __init__(self, heatmap):
        self.heatmap = heatmap
    
    def compute_pareto_frontier(self, tile_size=30, alpha=0.27):
        """Optimize tile placement (paper Eq. 7-9)."""
        tiles = self._discretize(tile_size)
        sorted_tiles = sorted(tiles, key=lambda x: -x[1])
        cumulative = np.cumsum([score for _, score in sorted_tiles])
        optimal_idx = np.argmax(cumulative >= alpha * cumulative[-1])
        return sorted_tiles[:optimal_idx]
    
    def _discretize(self, tile_size):
        """Discretize heatmap into tiles."""
        h, w = self.heatmap.shape
        return [((x, y), np.sum(self.heatmap[y:y+tile_size, x:x+tile_size]))
                for y in range(0, h, tile_size) 
                for x in range(0, w, tile_size)]