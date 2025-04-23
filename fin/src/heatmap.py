import numpy as np

class HeatmapGenerator:
    def __init__(self, shape=(600, 600), sigma=15):
        self.heatmap = np.zeros(shape, dtype=np.float64)
        self.sigma = sigma
    
    def add_trajectory(self, p1, p2):
        """Accumulate Gaussian probabilities (paper Eq. 5-6)."""
        xx, yy = np.meshgrid(np.arange(self.heatmap.shape[1]), 
                           np.arange(self.heatmap.shape[0]))
        px, py = p2[0]-p1[0], p2[1]-p1[1]
        norm = px**2 + py**2
        u = ((xx - p1[0])*px + (yy - p1[1])*py) / norm
        u = np.clip(u, 0, 1)
        closest_x = p1[0] + u * px
        closest_y = p1[1] + u * py
        dist = np.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)
        self.heatmap += np.exp(-dist**2 / (2 * self.sigma**2)) 
    
    def get_heatmap(self):
        hm = self.heatmap * (1/(self.sigma*np.sqrt(2*np.pi)))
        return hm / hm.sum()