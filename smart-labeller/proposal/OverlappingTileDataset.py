# data/OverlappingTileDataset.py
import torch, os, math
from torch.utils.data import Dataset
from PIL import Image

class OverlappingTileDataset(Dataset):
    def __init__(self, image_paths, tile_size=960, overlap_ratio=0.2, transforms=None):
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap_ratio))
        self.transforms = transforms
        self.tiles = self._create_tiles()

    def _create_tiles(self):
        tiles_list = []
        for idx, path in enumerate(self.image_paths):
            with Image.open(path) as img:
                w, h = img.size
            
            cols = math.ceil((w - self.tile_size) / self.stride) + 1
            rows = math.ceil((h - self.tile_size) / self.stride) + 1

            for r in range(rows):
                for c in range(cols):
                    x1, y1 = c * self.stride, r * self.stride
                    x2, y2 = min(x1 + self.tile_size, w), min(y1 + self.tile_size, h)
                    # Snap to edge to keep size consistent
                    if x2 == w: x1 = max(0, w - self.tile_size)
                    if y2 == h: y1 = max(0, h - self.tile_size)
                    
                    tiles_list.append({'img_idx': idx, 'path': path, 'coords': [x1, y1, x2, y2]})
        return tiles_list

    def __getitem__(self, idx):
        t = self.tiles[idx]
        img = Image.open(t['path']).convert("RGB").crop(t['coords'])
        if self.transforms: img = self.transforms(img)
        
        # This structure is required by your ot_main.py
        return {
            "image": img,
            "metadata": {
                "img_idx": t['img_idx'],
                "coords": torch.tensor(t['coords'])
            }
        }

    def __len__(self):
        return len(self.tiles)