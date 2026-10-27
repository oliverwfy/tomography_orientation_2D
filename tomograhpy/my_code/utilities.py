import re
import numpy as np
from pathlib import Path
from collections import namedtuple
import dataclasses
import salvus.namespace as sn
from salvus.flow.simple_config.source.cartesian import VectorPoint2D, VectorPoint3D,ScalarPoint2D
from salvus.flow.simple_config.receiver.cartesian import Point2D, Point3D
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple

import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy.typing as npt


import sys

def get_script_dir() -> Path:
    # PyInstaller / frozen exe
    if getattr(sys, "frozen", False):
        # directory of the executable
        exe_dir = Path(sys.executable).resolve().parent
        # if you bundled data with PyInstaller, they live under sys._MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else exe_dir

    # Path of the top-level script being executed (works with `python script.py` and `python -m pkg.module`)
    main = sys.modules.get("__main__")
    if main and hasattr(main, "__file__"):
        return Path(main.__file__).resolve().parent

    # Fallback: inside a module file (use this if calling from that module)
    return Path(__file__).resolve().parent



def add_events_to_Project(Project, events):
    for event in events:
        Project.add_to_project(event)
    return None

def add_inversion(Project, inv_config):
    Project+= inv_config
    return None



def reorder_list(lst, order):
    return [lst[i] for i in order]


def extract_numbers(lst):
    return [int(num) for s in lst for num in re.findall(r'\d+', s)]


def reorder_events_list(lst):

    e_nums = extract_numbers(lst)
    order = []
    for i in range(len(e_nums)): 
        order.append(int(np.where(np.array(e_nums) == i)[0][0]))
    return reorder_list(lst, order)

    
def source_location(event_data):
    return [(src.location[0], src.location[1]) for e in event_data for src in e.sources] 

def receriver_location(event_data):
    return [(rx.location[0], rx.location[1]) for rx in event_data[0].receivers]


def time_from_ed(event_data, temporal_interpolation=False):
    
    if temporal_interpolation:
        time = None
    else:
        time = event_data[0].get_waveform_data_xarray('displacement').time.values        
    return time


def fmc_data_from_ed(event_data, save_dir=None):
    time = time_from_ed(event_data)
    time = time.reshape(len(time), -1)
    srcs_loc = source_location(event_data)
    rxs_loc = receriver_location(event_data)
    fmc_data = np.zeros([len(time), 2,len(srcs_loc)*len(rxs_loc)])

    for i, _ in enumerate(event_data):
        fmc_data[:, 0,i*len(rxs_loc):(i+1)*len(rxs_loc)] = \
        event_data[0].get_data_cube(receiver_field='displacement', component='X')[1].T

        fmc_data[:, 1,i*len(rxs_loc):(i+1)*len(rxs_loc)] = \
        event_data[0].get_data_cube(receiver_field='displacement', component='Y')[1].T

    if save_dir:
        np.save(Path(save_dir, "time.npy"), time)
        np.save(Path(save_dir, "fmc_data.npy"), fmc_data)   
        np.save(Path(save_dir, "rxs_loc.npy"), rxs_loc)
        np.save(Path(save_dir, "srcs_loc.npy"), srcs_loc)
        
    return (fmc_data, time, rxs_loc, srcs_loc) 


class Vector:
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z
    
    def distance(self, other):
        """Calculates the Euclidean distance to another Vector."""
        return np.linalg.norm(self.array() - other.array(), ord=2)

    def array(self):
        """Returns the vector as a NumPy array."""
        if self.z: 
            return np.array([self.x, self.y, self.z])
        else:
            return np.array([self.x, self.y])



def compute_slowness(C, rho, theta):
    # Define propagation direction (sin(theta), 0, cos(theta))
    n3 = np.cos(np.radians(theta))
    n1 = np.sin(np.radians(theta))
    # VTI elasticity tensor components
    C11, C12, C13, C33, C44, C66 = C

    # Construct the Christoffel matrix
    Gamma = np.array([
        [C11 * n1**2 + C66 * n3**2, 0, (C13 + C44) * n1 * n3],
        [0, C66 * n1**2 + C44 * n3**2, 0],
        [(C13 + C44) * n1 * n3, 0, C33 * n3**2 + C44 * n1**2],
    ])

    # Solve the eigenvalue problem for phase velocities squared
    eigenvalues, _ = np.linalg.eig(Gamma)
    velocities = np.sqrt(eigenvalues / rho)

    # Compute slowness (s = 1/v)
    slowness = 1 / velocities

    # reorder slowness 
    sorted_indices = np.argsort(slowness)
    slowness_ordered = slowness[sorted_indices]
    
    return slowness_ordered


def phase_velocity_SH(C, rho, theta):
    C11, C12, C13, C33, C44, C66 = C
    n3 = np.cos(np.radians(theta))
    return np.sqrt( (C66*(1-n3**2) + C44*n3**2)/rho )



def generate_within_std(mean, std, size):


    values = np.random.exponential(mean, size)
    lower = mean / 10

    # Keep regenerating only the invalid entries
    while np.any((values < lower)):
        mask = (values < lower)
        values[mask] = np.random.exponential(mean, np.sum(mask))
    
    return values




def generate_random_layer(L, l_mean, n_layer, seed=None):
    np.random.seed(seed)  # Set the random seed
    l_ls = generate_within_std(l_mean, l_mean / 2, n_layer).round(6)  # Generate normally distributed values
    l_ls = np.abs(l_ls)  # Ensure all values are positive
    l_ls = L * l_ls / l_ls.sum()  # Normalize to sum to L
    
    # generate random orientation angles
    theta_ls = np.random.uniform(low=-np.pi/6, high=np.pi/6, size=n_layer).round(6)  

    return np.round(l_ls, 6), theta_ls



def generate_layer(L, l_mean, seed=None):
    
    np.random.seed(seed)  # Set the random seed
    n_layer = int(L//l_mean)
    l_ls = np.full(n_layer, l_mean)  # Equal thicknesses
    # Generate random orientation angles
    theta_ls = np.random.uniform(low=-np.pi/6, high=np.pi/6, size=n_layer).round(6)

    return l_ls, theta_ls

def generate_random_layer_v2(L, l_mean, n_layer, seed=None, orientation=np.pi/6):
    np.random.seed(seed)  # Set the random seed
    l_ls = generate_within_std(l_mean, l_mean / 2, n_layer).round(6)  # Generate normally distributed values
    l_ls = np.abs(l_ls)  # Ensure all values are positive
    l_ls = L * l_ls / l_ls.sum()  # Normalize to sum to L
    
    # generate random orientation angles
    # Generate data for a normal distribution
    x = np.linspace(-3, 3, n_layer)

    mean = 0
    std_dev = 1
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    y = y/y.max()
    
    
    theta_ls = [ np.round( np.random.uniform(low=-orientation*w, high=orientation*w), 6) for w in y]

    return np.round(l_ls, 6), theta_ls



@dataclasses.dataclass
class Transducers_2D:
    n_tx :  int
    n_rx :  int
    edge_gap :  float
    domain : tuple
    f_dir : str = "y"
    recording_fields: list[str] = dataclasses.field(
        default_factory=lambda: ["displacement"])
    
    def create_salvus_source_receivers(self):
        if self.f_dir == 'y':
            src_pos = [(np.round(x, 5) , np.round(self.domain[3] * self.edge_gap, 5))
                       for x in np.linspace(self.domain[0], self.domain[1], self.n_tx+2)[1:-1]]
            rx_pos = [(np.round(x, 5) , np.round(self.domain[3] * (1-self.edge_gap), 5))
                       for x in np.linspace(self.domain[0], self.domain[1], self.n_rx+2)[1:-1]]
            
            srcs = [VectorPoint2D(x=s[0],y=s[1], fx=0, fy=1e9) for s in src_pos]
            rxs = [Point2D(x=r[0], y=r[1], station_code=f"REC{i + 1}",
                           fields=self.recording_fields,) 
                   for i, r in enumerate(rx_pos)
                   ]
        # elif self.f_dir == 'x':
            # src_pos = [(np.round(x, 5) , np.round(self.domain[3] * self.edge_gap, 5))
            #            for x in np.linspace(self.domain[0], self.domain[1], self.n_tx+2)[1:-1]]
            # rx_pos = [(np.round(x, 5) , np.round(self.domain[3] * self.edge_gap, 5))
            #            for x in np.linspace(self.domain[0], self.domain[1], self.n_rx+2)[1:-1]]
            # rx_pos += [(np.round(x, 5) , np.round(self.domain[3] * (1-self.edge_gap), 5))
            #            for x in np.linspace(self.domain[0], self.domain[1], self.n_rx+2)[1:-1]]
            
            # srcs = [VectorPoint2D(x=s.x,y=s.y, fx=0, fy=1) for s in src_pos]
            # rxs = [Point2D(x=r.x, y=r.y, station_code=f"REC{i + 1}",
            #                fields=self.recording_fields,) 
            #        for i, r in enumerate(rx_pos)
            #        ]
            
        return srcs, rxs


@dataclasses.dataclass
class ArrayTransducer2D:
    nx: int
    dx: float
    x0: float
    f_dir: tuple
    source_y : list[float] = dataclasses.field(
        default_factory=lambda: [0.001, 0.009]
    )
    array_name: str = "array_0"
    
    recording_fields: list[str] = dataclasses.field(
        default_factory=lambda: ["phi"]
    )
    

    
    def test_within_domain(self, domain: sn.domain.Domain) -> bool:
        x_bounds = domain.bounds.hc["x"]
        array_x_bounds = (self.x0, self.x0 + (self.nx - 1) * self.dx)
        return (x_bounds[0] <= array_x_bounds[0] and array_x_bounds[1] <= x_bounds[1])
    
    
    def create_salvus_source_receivers(self, source_index: int, source_y: float, f_source: int = 1):
        if source_index < 0 or source_index >= self.nx:
            raise ValueError("Source index out of range.")
        
        array_coordinates = np.zeros((self.nx))
        for i in range(self.nx):
            array_coordinates[i] = self.x0 + i * self.dx
            
        # source position
        source_x = array_coordinates[source_index]
        
        receivers = []
        source = VectorPoint2D(x=source_x, y=source_y, fx= self.f_dir[0], fy= self.f_dir[1]) 
        # sn.simple_config.source.cartesian.SideSetScalarPoint2D(
        # # Note that this 0 for Y is used for starting the projection
        # # on the side set, it is not the coordinate of the source.
        # point=(source_x, 0),
        # f=f_source,
        # direction="y",
        # side_set_name=source_side,
        # )    
        

        
                            
        receivers += [
            Point2D(x=array_coordinates[i], y= y,
                station_code = f"{self.array_name}_y{j}_x{i:03d}",
                fields=self.recording_fields,
            )
            for i in range(self.nx)
            for j,y in enumerate(self.source_y)
        ]
            
            # [
            # sn.simple_config.receiver.cartesian.SideSetPoint2D(
            #     direction="y",
            #     point=(
            #         array_coordinates[i],
            #         0,
            #     ),
            #     # Note that we're using leading zeros, but if nx/ny is
            #     # really high this needs to be scaled up.
            #     station_code = f"{self.array_name}_{side_set}_x{i:03d}",
            #     fields=self.recording_fields,
            #     side_set_name=side_set,
            # )
            # for i in range(self.nx)
            # for y in self.source_y
            # ]
        
        return source, receivers
        
        






@dataclass
class VoronoiGrainIndexer:
    """
    Index Voronoi regions from a color PNG and query the grain ID for a point.

    - Coordinates are normalized: (x, y) in [0,1] x [0,1], origin at bottom-left.
    - Grain IDs are assigned ONLY to regions that pass the percentage threshold,
      and are sorted by area descending: gid=0 is the largest kept region.
    - Pixels not belonging to a kept region are labelled -1.
    """
    image_path: str
    min_percent: float = 0.1       # keep regions with area >= this percent
    dilate_iters: int = 1          # boundary dilation iterations for robust edge detection

    # Internal state (populated by process())
    img_array: Optional[np.ndarray] = None
    H: Optional[int] = None
    W: Optional[int] = None
    areas_df: Optional[pd.DataFrame] = None
    kept_colors_sorted: Optional[np.ndarray] = None
    color_to_gid: Optional[dict] = None
    seg_labels: Optional[np.ndarray] = None
    boundary_mask_dilated: Optional[np.ndarray] = None

    def process(self) -> None:
        """Load image, segment colors, compute areas, build grain labels and boundaries."""
        # ---- Load image ----
        img = Image.open(self.image_path).convert("RGB")
        self.img_array = np.array(img)
        self.H, self.W, _ = self.img_array.shape

        # ---- Unique colors & counts ----
        unique_colors, counts = np.unique(
            self.img_array.reshape(-1, 3), axis=0, return_counts=True
        )
        total_pixels = self.H * self.W
        percentages = counts / total_pixels * 100.0

        # ---- Keep regions by percentage threshold ----
        keep_mask = percentages >= self.min_percent
        kept_colors = unique_colors[keep_mask]
        kept_counts = counts[keep_mask]
        kept_percentages = percentages[keep_mask]

        # ---- Area table (unsorted) ----
        areas_df = pd.DataFrame({
            "Color (RGB)": [tuple(c) for c in kept_colors],
            "Pixel Count": kept_counts,
            "Percentage (%)": kept_percentages
        })

        # ---- Sort by area (descending) and build color->gid map ----
        order = np.argsort(-kept_counts)  # largest first
        self.kept_colors_sorted = kept_colors[order]
        kept_counts_sorted = kept_counts[order]
        kept_percentages_sorted = kept_percentages[order]

        self.areas_df = areas_df.iloc[order].reset_index(drop=True)
        self.color_to_gid = {tuple(color): idx for idx, color in enumerate(self.kept_colors_sorted)}

        # ---- Build seg_labels restricted to kept regions (others = -1) ----
        self.seg_labels = np.full((self.H, self.W), fill_value=-1, dtype=int)
        for color, gid in self.color_to_gid.items():
            mask = np.all(self.img_array == color, axis=-1)
            self.seg_labels[mask] = gid

        # ---- Build boundary mask from seg_labels ----
        boundary_mask = np.zeros((self.H, self.W), dtype=bool)
        # vertical boundaries (row-wise change)
        boundary_mask[1:, :] |= (self.seg_labels[1:, :] != self.seg_labels[:-1, :])
        # horizontal boundaries (col-wise change)
        boundary_mask[:, 1:] |= (self.seg_labels[:, 1:] != self.seg_labels[:, :-1])

        # Dilate (optional) to make boundary robust
        kernel = np.ones((3, 3), np.uint8)
        self.boundary_mask_dilated = cv2.dilate(
            boundary_mask.astype(np.uint8) * 255, kernel, iterations=self.dilate_iters
        ).astype(bool)

    # ---------- Queries & utilities ----------

    def point_to_grain_id(self, x: float, y: float) -> int:
        """
        Return the grain ID for normalized (x,y) in [0,1]x[0,1], origin at bottom-left.
        Grain IDs are sorted by area (0 = largest). Returns -1 if outside kept regions.
        """
        if self.seg_labels is None:
            raise RuntimeError("Call .process() before querying.")

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"coords must be in [0,1], got ({x},{y})")

        col = int(round(x * (self.W - 1)))
        row = int(round((1 - y) * (self.H - 1)))  # flip y to image coords

        gid = int(self.seg_labels[row, col])

        # Handle boundary/unknown by local voting for stability
        if gid == -1 or self.boundary_mask_dilated[row, col]:
            neigh_ids = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                rr, cc = row + dr, col + dc
                if 0 <= rr < self.H and 0 <= cc < self.W and self.seg_labels[rr, cc] != -1:
                    neigh_ids.append(int(self.seg_labels[rr, cc]))
            if neigh_ids:
                gid = Counter(neigh_ids).most_common(1)[0][0]

        return gid

    def show_boundary_overlay(self, save_path: str | Path = None) -> None:
        """Show or save an image with boundaries overlaid in black."""
        if self.img_array is None or self.boundary_mask_dilated is None:
            raise RuntimeError("Call .process() before saving overlays.")

        overlay = self.img_array.copy()
        overlay[self.boundary_mask_dilated] = (0, 0, 0)

        if save_path:
            Image.fromarray(overlay).save(str(save_path))
        else:
            plt.figure(figsize=(6,6))
            plt.title("Image with Boundaries")
            plt.imshow(overlay)
            plt.axis("off")
            plt.show()

    def plot_bar_chart(self, save_path: str | Path = None,
                    title: str = "Equivalent Circle Radius of Regions") -> None:
        """
        Show or save bar chart of equivalent radii (normalized true area=1).
        """
        if self.areas_df is None or self.kept_colors_sorted is None:
            raise RuntimeError("Call .process() before plotting.")

        true_area = 1.0
        pct = self.areas_df["Percentage (%)"].to_numpy() / 100.0
        equi_r = np.sqrt(pct * true_area / np.pi)

        labels = list(range(len(equi_r)))
        colors = (self.kept_colors_sorted / 255.0)

        plt.figure(figsize=(8, 6))
        plt.bar(labels, equi_r, color=colors)
        plt.ylabel("Equivalent Radius (normalized units)")
        plt.xlabel("Region Index (sorted by area)")
        plt.title(title)
        plt.xticks(labels)
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path))
            plt.close()
        else:
            plt.show()

    def get_areas_table(self) -> pd.DataFrame:
        """
        Return a copy of the areas table (sorted by area).
        Columns: 'Color (RGB)', 'Pixel Count', 'Percentage (%)'
        """
        if self.areas_df is None:
            raise RuntimeError("Call .process() before accessing areas.")
        return self.areas_df.copy()

    def color_for_gid(self, gid: int) -> Tuple[int, int, int]:
        """Return the RGB tuple for a given grain id (sorted by area)."""
        if self.kept_colors_sorted is None:
            raise RuntimeError("Call .process() first.")
        if not (0 <= gid < len(self.kept_colors_sorted)):
            raise ValueError(f"gid {gid} out of range [0, {len(self.kept_colors_sorted)-1}]")
        c = self.kept_colors_sorted[gid]
        return (int(c[0]), int(c[1]), int(c[2]))
    




# convert nodel field to elemental field 
def nodal_to_elemental_field(
    nodal_field: npt.NDArray, 
    mesh: sn.UnstructuredMesh
) -> npt.NDArray:
    
    elemental_nodal_field = nodal_field[mesh.connectivity]  # convert nodal field to elemental nodel field 
    
    # mass matrix with the same shape as elemental nodal field
    mass_mat = mesh.get_mass_matrix()
    weighted_sum = (mass_mat[:, :, None, None] * elemental_nodal_field).sum(axis=1)
    normalization = mass_mat.sum(axis=1)[:, None, None]
    return weighted_sum / normalization






def elemental_nodal_to_nodal_field(
    elemental_nodal_field: npt.NDArray,
    connectivity: npt.NDArray
) -> npt.NDArray:
    
    """
    Convert elemental nodal field -> global nodal field by averaging
    overlapping element contributions.

    Parameters
    ----------
    elemental_nodal_field : (n_elements, n_nodes_per_element, ...)
        Field values defined at each element node.
    connectivity : (n_elements, n_nodes_per_element)
        Global node indices of each element.

    Returns
    -------
    nodal_field : (n_global_nodes, ...)
        Averaged nodal field.
    """

    n_elements, n_nodes_per_element = connectivity.shape
    n_global_nodes = connectivity.max() + 1

    # Prepare arrays for accumulation
    nodal_sum = np.zeros((n_global_nodes,) + elemental_nodal_field.shape[2:], dtype=float)
    nodal_count = np.zeros(n_global_nodes, dtype=int)

    # Loop over elements
    for e in range(n_elements):
        nodes = connectivity[e]
        nodal_sum[nodes] += elemental_nodal_field[e]
        nodal_count[nodes] += 1

    # Avoid division by zero
    nodal_count[nodal_count == 0] = 1

    return nodal_sum / nodal_count[:, None] if elemental_nodal_field.ndim > 2 else nodal_sum / nodal_count


