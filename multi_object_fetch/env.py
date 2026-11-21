from abc import ABC, abstractmethod
import random
import os
import gym
import numpy as np
from fetch_block_construction.envs.robotics.fetch.construction import FetchBlockConstructionEnv
from fetch_block_construction.envs.robotics import fetch_env
from gym import utils as gym_utils
from typing import Dict, Any, Optional, Sequence, Tuple
from multi_object_fetch.assets.generate_multi_camera_xml import generate_multi_camera_xml
from mujoco_py import functions
from mujoco_py.generated import const
from mujoco_py.modder import TextureModder
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import shutil
import tempfile
from pathlib import Path
from importlib.resources import files
from mujoco_py.builder import MujocoException
from multi_object_fetch.language.instructions import (
    COLOR_NAMES, MESH_SHAPE_NAMES, MULTICOLORSHAPE_HELDOUT, MULTISHAPE_NAMES, SHAPE_NAMES, build_instruction,
)

MESH_DIR = Path(__file__).resolve().parent / "assets" / "meshes"
# Model geom/site group that the default render options hide (groups 0-2 are shown).
HIDDEN_GROUP = 3


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    q[0], q[1 + axis] = np.cos(angle / 2), np.sin(angle / 2)
    return q


def mesh_radius(path: Path) -> float:
    """Distance from the origin to the farthest vertex of a binary STL."""
    data = path.read_bytes()
    num_triangles = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(data, dtype=record, count=num_triangles, offset=84)
    return float(np.linalg.norm(triangles["vertices"].reshape(-1, 3), axis=1).max())


def prepare_fetch_assets() -> Path:
    cache_dir = Path.home() / ".cache" / "multi_object_fetch"
    fetch_assets_src = files("fetch_block_construction.envs.robotics") / "assets"
    if not (cache_dir / "fetch").is_dir():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fetch_assets_src, cache_dir, dirs_exist_ok=True)
    return cache_dir


class ColorsMixin(ABC):
    @property
    def colors(self) -> np.ndarray:
        return np.array([(0, 255, 0),
                         (47, 79, 79),
                         (139, 69, 19),
                         (25, 25, 112),
                         (0, 100, 0),
                         (189, 183, 107),
                         (72, 209, 204),
                         (255, 0, 0),
                         (255, 165, 0),
                         (255, 255, 0),
                         (199, 21, 133),
                         (0, 0, 205),
                         (0, 250, 154),
                         (216, 191, 216),
                         (255, 0, 255),
                         (30, 144, 255)]) / 255

    @property
    def color_names(self) -> Tuple[str, ...]:
        """English names aligned with ``colors`` indices (see COLOR_NAMES)."""
        return COLOR_NAMES

    @property
    def shape_names(self) -> Tuple[str, ...]:
        """English names aligned with ``shape_geoms`` indices (see SHAPE_NAMES)."""
        return SHAPE_NAMES

    @property
    def shape_geoms(self) -> Tuple[int, ...]:
        return (const.GEOM_SPHERE, const.GEOM_BOX)

    @staticmethod
    def delta_e_cie2000(lab1, lab2, kL=1, kC=1, kH=1):
        L1, a1, b1 = lab1.lab_l, lab1.lab_a, lab1.lab_b
        L2, a2, b2 = lab2.lab_l, lab2.lab_a, lab2.lab_b

        C1 = np.sqrt(a1 ** 2 + b1 ** 2)
        C2 = np.sqrt(a2 ** 2 + b2 ** 2)
        C_ave = (C1 + C2) / 2

        G = 0.5 * (1 - np.sqrt(C_ave ** 7 / (C_ave ** 7 + 25 ** 7)))

        L1_, L2_ = L1, L2
        a1_, a2_ = (1 + G) * a1, (1 + G) * a2
        b1_, b2_ = b1, b2

        C1_ = np.sqrt(a1_ ** 2 + b1_ ** 2)
        C2_ = np.sqrt(a2_ ** 2 + b2_ ** 2)

        h1_ = np.degrees(np.arctan2(b1_, a1_)) % 360
        h2_ = np.degrees(np.arctan2(b2_, a2_)) % 360

        delta_L = L2_ - L1_
        delta_C = C2_ - C1_
        delta_h = h2_ - h1_
        if abs(delta_h) > 180:
            if h2_ <= h1_:
                delta_h += 360
            else:
                delta_h -= 360
        delta_H = 2 * np.sqrt(C1_ * C2_) * np.sin(np.radians(delta_h) / 2)

        L_ave = (L1_ + L2_) / 2
        C_ave = (C1_ + C2_) / 2

        h_ave = h1_ + h2_
        if abs(h1_ - h2_) > 180:
            h_ave += 360
        h_ave /= 2

        T = (1 - 0.17 * np.cos(np.radians(h_ave - 30)) +
             0.24 * np.cos(np.radians(2 * h_ave)) +
             0.32 * np.cos(np.radians(3 * h_ave + 6)) -
             0.20 * np.cos(np.radians(4 * h_ave - 63)))

        delta_theta = 30 * np.exp(-((h_ave - 275) / 25) ** 2)

        R_C = 2 * np.sqrt(C_ave ** 7 / (C_ave ** 7 + 25 ** 7))
        S_L = 1 + (0.015 * (L_ave - 50) ** 2) / np.sqrt(20 + (L_ave - 50) ** 2)
        S_C = 1 + 0.045 * C_ave
        S_H = 1 + 0.015 * C_ave * T
        R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

        delta_E = np.sqrt(
            (delta_L / (kL * S_L)) ** 2 +
            (delta_C / (kC * S_C)) ** 2 +
            (delta_H / (kH * S_H)) ** 2 +
            R_T * (delta_C / (kC * S_C)) * (delta_H / (kH * S_H))
        )

        return delta_E

    def delta_e_2000_to_red(self, color):
        red = sRGBColor(1.0, 0.0, 0.0)
        color_rgb = sRGBColor(color[0], color[1], color[2])
        color_lab = convert_color(color_rgb, LabColor)
        red_lab = convert_color(red, LabColor)
        return self.delta_e_cie2000(red_lab, color_lab)


class MultiObjectFetchEnv(fetch_env.FetchEnv, gym_utils.EzPickle, ColorsMixin, ABC):
    def __init__(self, initial_qpos: Dict[str, Any], obs_type: str = "dictstate", object_size: float = 0.045,
                 target_size: float = 0.04, robot_configuration: str = "default", viewpoint: str = "frontview",
                 randomize_background_color: bool = False, randomize_table_color: bool = False,
                 environment: str = "Reach", task: str = "Red") -> None:
        """Initializes a new multi-object Fetch environment.

        Args:
            initial_qpos (dict): A dictionary that maps joint-names to their initial configurations.
            object_size (float): The size of the objects in the environment.
            target_size (float): The size of the target object in the environment.
            robot_configuration (str): The variant of the robot to use.
            viewpoint (str): The camera viewpoint to use for rendering.
            environment (str): High-level motor skill: Reach, Push, Pick, or Stack.
            task (str): Relational color task: Red, Reddest, Odd, OddGroups, Multicolor, or
                MulticolorShape (Reach only). Stack uses Onto1, Onto1Fixed, Onto2, Onto2Fixed.
        """
        self.obs_type = obs_type
        self.object_size = object_size
        self.target_size = target_size
        self.robot = robot_configuration
        self.viewpoint = viewpoint
        self.randomize_background_color = randomize_background_color
        self.randomize_table_color = randomize_table_color
        self.environment = environment
        self.task = task
        # Multicolor instruction depends on the episode's target color (set in _reset_sim).
        self.instruction = (
            ""
            if task in ("Multicolor", "MulticolorShape") or environment == "Stack"
            else build_instruction(environment, task)
        )

    def render(self, mode='human', size=None):
        self._render_callback()
        size = size if size is not None else (self.width, self.height)
        self._bind_offscreen_buffer()
        if mode == "seg":
            seg = self.sim.render(size[0], size[1], camera_name=self.viewpoint, segmentation=True)
            ids = seg[:, :, 1]

            # table to background
            id = self.sim.model.geom_name2id("table0")
            ids[ids == id] = -1

            for id in np.unique(ids):
                if id == -1:
                    continue

                name = self.sim.model.geom_id2name(id)
                if name is not None and ('forearm' in name or 'wrist' in name or 'elbow' in name or
                                         'upperarm' in name or 'gripper' in name):
                    ids[ids == id] = 254
                # robot to background
                elif name is not None and 'robot' in name:
                    ids[ids == id] = -1

            return (ids[::-1, :] + 1).astype(np.uint8)
        else:
            # original image is upside-down, so flip it
            return self.sim.render(size[0], size[1], camera_name=self.viewpoint)[::-1, :, :].copy()

    def _bind_offscreen_buffer(self):
        # mujoco_py's EGL backend shares one OpenGL context among all simulations on a GPU, and MjSim.render draws into
        # whichever framebuffer is bound: that of the last simulation rendered. Without binding this simulation's own
        # buffer, its first frame after another environment rendered is its previous frame.
        context = self.sim._render_context_offscreen
        if context is not None:
            functions.mjr_setBuffer(const.FB_OFFSCREEN, context.con)

    SUCCESS_THRESHOLD = 0.05

    @abstractmethod
    def goal_distance(self) -> float:
        """Distance that defines success: gripper to target (Reach) or object0 to target (Push, Pick)."""
        pass

    def success(self) -> bool:
        return self.goal_distance() < self.SUCCESS_THRESHOLD

    def episode_info(self) -> Dict[str, Any]:
        """Values that are constant within an episode (see docs/metrics.md)."""
        return {"num_distractors": self.num_distractors, "target_color": self.target_color,
                "target_shape": self.target_shape}

    def step_info(self) -> Dict[str, Any]:
        """Per-step physical quantities used by the evaluation metrics (see docs/metrics.md)."""
        return {"dist": float(self.goal_distance())}

    def reset(self):
        obs = super().reset()
        # Move the target0 site to the new goal; otherwise only render() does it, and success() reads the site.
        self._render_callback()
        return obs

    def step(self, action):
        try:
            obs, reward, done, info = super().step(action)
        except MujocoException:
            # Random / large actions can blow up contacts; end the episode cleanly.
            # Per-step quantities are omitted: after the reset they no longer describe the episode.
            self.sim.set_state(self.initial_state)
            self.sim.forward()
            obs = self._get_obs()
            info = {"success": False, "unstable": True, **self.episode_info()}
            return obs, 0.0, True, info
        info["success"] = self.success()
        info["unstable"] = False
        info.update(self.episode_info())
        info.update(self.step_info())
        return obs, reward, done, info

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # Index into COLOR_NAMES of the instructed color; -1 for tasks without a named color.
        self.target_color = -1
        # Index into MULTISHAPE_NAMES of the instructed shape; -1 for tasks without a named shape.
        self.target_shape = -1

        if self.randomize_table_color:
            table_color = np.random.uniform(0, 1, size=3)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('table0')][:3] = table_color

        if self.randomize_background_color:
            if not hasattr(self, "texture_modder"):
                self.texture_modder = TextureModder(self.sim)
            self.texture_modder.rand_noise('skybox')


class ReachEnv(MultiObjectFetchEnv):
    # MulticolorShape: random twist of each mesh shape away from facing the frontview camera, for variety.
    # Up to MESH_MAX_ROLL in the outline's plane, up to MESH_MAX_TILT out of it (the outline keeps >= cos(tilt)
    # of its width, so it stays readable). Rotations are about the shape's center: bounding spheres do not change.
    MESH_MAX_ROLL = np.radians(30)
    MESH_MAX_TILT = np.radians(20)
    # Cubes get any turn about the vertical (+-45 degrees covers all, a cube repeats every 90) and the same tilt.
    CUBE_MAX_YAW = np.radians(45)
    CUBE_MAX_TILT = MESH_MAX_TILT

    def __init__(self, initial_qpos: Dict[str, Any], obs_type: str = "dictstate", object_size: float = 0.045,
                 target_size: float = 0.04, robot_configuration: str = "default", viewpoint: str = "frontview",
                 num_distractors: int = 1, reward_type: str = "Dense", task: str = "Red", width: int = 512,
                 height: int = 512, environment: str = "Reach", split: Optional[str] = None,
                 heldout_pairs: Optional[Sequence[Tuple[str, str]]] = None) -> None:
        """split (MulticolorShape only): None uses every (color, shape) pair; "train" never shows a held-out pair,
        on any object; "val" always instructs a held-out pair (distractors may use any pair). heldout_pairs:
        (color name, shape name) pairs to hold out, MULTICOLORSHAPE_HELDOUT by default."""
        super().__init__(
            initial_qpos, obs_type, object_size, target_size, robot_configuration, viewpoint,
            environment=environment, task=task,
        )
        self.num_distractors = num_distractors
        if split not in (None, 'train', 'val'):
            raise ValueError(f"Invalid split: {split}")
        if split is not None and task != 'MulticolorShape':
            raise ValueError("split is only supported by MulticolorShape")
        self.split = split
        heldout = {(COLOR_NAMES.index(c), MULTISHAPE_NAMES.index(s))
                   for c, s in (MULTICOLORSHAPE_HELDOUT if heldout_pairs is None else heldout_pairs)}
        pairs = [(c, s) for c in range(len(COLOR_NAMES)) for s in range(len(MULTISHAPE_NAMES))]
        # (color index, MULTISHAPE_NAMES index) pairs the target / any object may use.
        self.target_pairs = [p for p in pairs if (p in heldout) == (split == 'val')] if split else pairs
        self.object_pairs = [p for p in pairs if p not in heldout] if split == 'train' else pairs
        self.heldout_pairs = sorted(heldout)
        self.width = width
        self.height = height

        # MulticolorShape: Language Table shapes as mesh geoms next to the sites (see generate_multi_camera_xml).
        shape_meshes = ({shape: MESH_DIR / f"{shape}.stl" for shape in MESH_SHAPE_NAMES}
                        if task == 'MulticolorShape' else None)
        # Bounding-sphere radius of each mesh shape, in target sizes.
        self.mesh_radii = {shape: mesh_radius(path) for shape, path in (shape_meshes or {}).items()}
        # Site name -> mesh shape name shown at it this episode (set in _reset_sim).
        self.mesh_shapes = {}

        fetch_cache = prepare_fetch_assets()
        with tempfile.NamedTemporaryFile(mode='wt', dir=fetch_cache / "fetch", delete=False, suffix=".xml") as fp:
            fp.write(generate_multi_camera_xml(num_distractors, self.robot, task='reach', target_size=target_size,
                                               shape_meshes=shape_meshes))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0, obj_range=0, target_range=0,
            distance_threshold=target_size + 0.015, initial_qpos=initial_qpos, reward_type=reward_type,
            obs_type=obs_type, render_size=0)

        gym_utils.EzPickle.__init__(self, initial_qpos, obs_type, object_size, target_size, robot_configuration,
                                    viewpoint, num_distractors, reward_type, task, width, height, environment,
                                    split, heldout_pairs)
        os.remove(MODEL_XML_PATH)



    def site_radius(self, name):
        """Bounding-sphere radius of a target/distractor site: a cube's corners reach sqrt(3) half-sizes out."""
        if name in self.mesh_shapes:
            return self.target_size * self.mesh_radii[self.mesh_shapes[name]]
        site_id = self.sim.model.site_name2id(name)
        size = self.sim.model.site_size[site_id][0]
        return size * np.sqrt(3) if self.sim.model.site_type[site_id] == const.GEOM_BOX else size

    def sample_pos(self, object_size):
        pos = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        pos[2] = self.height_offset + self.table_size[2] + object_size if pos[2] < self.height_offset + self.table_size[2] + object_size else pos[2]
        return pos.copy()

    def _sample_goal(self):
        return self.sample_pos(self.target_size)

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        self.height_offset = self.sim.data.get_body_xpos('table0')[2]
        self.table_size = self.sim.model.geom_size[self.sim.model.geom_name2id('table0')]
        if self.task == 'MulticolorShape':
            # Untwisted orientation of the mesh shapes (facing the frontview camera), the same for every site.
            self.mesh_base_quat = self.sim.model.body_quat[self.sim.model.body_name2id('target0:shapes')].copy()

    def _reset_sim(self):
        super()._reset_sim()

        if self.task == 'Odd':
            colors = self.np_random.choice(range(len(self.colors)), size=2, replace=False)
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = self.colors[colors[0]]

            for i in range(self.num_distractors):
                self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = self.colors[colors[1]]

        elif self.task == 'OddGroups':
            colors = self.np_random.choice(range(len(self.colors)), size=3, replace=False)
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = self.colors[colors[0]]

            for i in range(self.num_distractors):
                self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = self.colors[colors[1]] if i % 2 == 0 else self.colors[colors[2]]

        elif self.task == 'Red':
           target_color = np.array([255, 0, 0])
           self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = target_color / 255

           def generate_number_excluding_7(max_number):
               numbers = np.arange(max_number)  # Create an array with numbers from 0 to max_number
               numbers = np.delete(numbers, 7)  # Remove the number 7 from the array
               return self.np_random.choice(numbers)  # Randomly choose a number from the array

           for i in range(self.num_distractors):
               self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = self.colors[generate_number_excluding_7(len(self.colors))]

        elif self.task == 'Reddest':
           # Sample as many colors as there are targets (real target + distractors)
           num_colors = 1 + self.num_distractors
           sampled_colors = self.colors[self.np_random.choice(range(len(self.colors)), size=num_colors, replace=False)]

           # Find the reddest color using Delta E 2000
           reddest_color = min(sampled_colors, key=lambda color: self.delta_e_2000_to_red(color))

           # Assign the reddest color to the target
           self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = reddest_color

           # Assign the other colors to distractors
           distractor_colors = [c for c in sampled_colors if not np.array_equal(c, reddest_color)]
           for i in range(self.num_distractors):
               self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = distractor_colors[i]

        elif self.task == 'Multicolor':
            assert len(self.color_names) == len(self.colors)
            num_colors = 1 + self.num_distractors
            color_indices = self.np_random.choice(
                len(self.colors), size=num_colors, replace=False
            )
            target_idx = int(color_indices[0])
            self.target_color = target_idx
            self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][:3] = self.colors[target_idx]
            self.instruction = build_instruction(
                self.environment, self.task, color_name=self.color_names[target_idx]
            )
            for i in range(self.num_distractors):
                distractor_idx = int(color_indices[i + 1])
                self.sim.model.site_rgba[self.sim.model.site_name2id(f'distractor{i}')][:3] = (
                    self.colors[distractor_idx]
                )

        elif self.task == 'MulticolorShape':
            assert len(self.color_names) == len(self.colors) and len(SHAPE_NAMES) == len(self.shape_geoms)

            def pick(candidates, why):
                if not candidates:
                    raise ValueError(f"No (color, shape) pair left for {why} (split={self.split}); "
                                     "hold out fewer pairs")
                return candidates[int(self.np_random.choice(len(candidates)))]

            # Pairs come from self.target_pairs / self.object_pairs (see split in __init__).
            color, shape = pick(self.target_pairs, "the target")
            objects = [(color, shape)]
            if self.num_distractors > 0:
                # The target's color with another shape, so color alone is not enough.
                objects.append(pick([(c, s) for c, s in self.object_pairs if c == color and s != shape],
                                    "a same-color distractor"))
            if self.num_distractors > 1:
                # The target's shape with another color, so shape alone is not enough.
                objects.append(pick([(c, s) for c, s in self.object_pairs if s == shape and c != color],
                                    "a same-shape distractor"))
            # The rest get unused (color, shape) pairs, so the instruction always names exactly one object.
            unused = [p for p in self.object_pairs if p not in objects]
            if len(unused) < 1 + self.num_distractors - len(objects):
                raise ValueError(f"Not enough (color, shape) pairs for the distractors (split={self.split})")
            for k in self.np_random.choice(len(unused), size=1 + self.num_distractors - len(objects), replace=False):
                objects.append(unused[int(k)])

            self.target_color, self.target_shape = color, shape
            self.instruction = build_instruction(
                self.environment, self.task, color_name=self.color_names[color], shape_name=MULTISHAPE_NAMES[shape]
            )
            # Primitive shapes are drawn by the site; mesh shapes by the site's mesh geom, with the site hidden.
            self.mesh_shapes = {}
            site_names = ['target0'] + [f'distractor{i}' for i in range(self.num_distractors)]
            for name, (c, s) in zip(site_names, objects):
                site_id = self.sim.model.site_name2id(name)
                is_mesh = s >= len(SHAPE_NAMES)
                self.sim.model.site_rgba[site_id] = (*self.colors[c], 0 if is_mesh else 1)
                self.sim.model.site_group[site_id] = HIDDEN_GROUP if is_mesh else 0
                if not is_mesh:
                    self.sim.model.site_type[site_id] = self.shape_geoms[s]
                else:
                    self.mesh_shapes[name] = MULTISHAPE_NAMES[s]
                for mesh_shape in MESH_SHAPE_NAMES:
                    geom_id = self.sim.model.geom_name2id(f'{name}:{mesh_shape}')
                    shown = mesh_shape == self.mesh_shapes.get(name)
                    self.sim.model.geom_rgba[geom_id] = (*self.colors[c], 1 if shown else 0)
                    self.sim.model.geom_group[geom_id] = 0 if shown else HIDDEN_GROUP
                # Twist in the shape's frame (z: outline normal, toward the camera): tilt about x and y, roll about z.
                tilt_x, tilt_y = self.np_random.uniform(-self.MESH_MAX_TILT, self.MESH_MAX_TILT, size=2)
                roll = self.np_random.uniform(-self.MESH_MAX_ROLL, self.MESH_MAX_ROLL)
                twist = quat_mul(quat_mul(axis_quat(0, tilt_x), axis_quat(1, tilt_y)), axis_quat(2, roll))
                self.sim.model.body_quat[self.sim.model.body_name2id(f'{name}:shapes')] = quat_mul(
                    self.mesh_base_quat, twist)
                # Cube twist in the world frame (z up): yaw, then tilt about x and y; other sites stay untwisted.
                yaw = self.np_random.uniform(-self.CUBE_MAX_YAW, self.CUBE_MAX_YAW)
                cube_tilt_x, cube_tilt_y = self.np_random.uniform(-self.CUBE_MAX_TILT, self.CUBE_MAX_TILT, size=2)
                cube_twist = quat_mul(quat_mul(axis_quat(2, yaw), axis_quat(0, cube_tilt_x)), axis_quat(1, cube_tilt_y))
                is_cube = not is_mesh and self.shape_geoms[s] == const.GEOM_BOX
                self.sim.model.site_quat[site_id] = cube_twist if is_cube else (1, 0, 0, 0)

        else:
            raise ValueError("Invalid task: {}".format(self.task))

        # Randomize start position of distractors.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        ball_positions = [self.goal]
        radii = [self.site_radius('target0')]
        for i in range(self.num_distractors):
            radius = self.site_radius(f'distractor{i}')
            pos_not_valid = True
            while pos_not_valid:
                distractor_xpos = self.sample_pos(self.target_size)

                pos_not_valid = False
                for position, other_radius in zip(ball_positions, radii):
                    pos_not_valid = np.linalg.norm(distractor_xpos - position) < radius + other_radius
                    if pos_not_valid:
                        break

            ball_positions.append(distractor_xpos)
            radii.append(radius)

            site_id = self.sim.model.site_name2id(f'distractor{i}')
            self.sim.model.site_pos[site_id] = distractor_xpos - sites_offset[0]

        self.sim.forward()
        return True

    def _render_callback(self):
        super()._render_callback()
        if self.task == 'MulticolorShape':
            # Mesh shapes follow their sites (both are placed relative to floor0).
            for name in ['target0'] + [f'distractor{i}' for i in range(self.num_distractors)]:
                body_id = self.sim.model.body_name2id(f'{name}:shapes')
                self.sim.model.body_pos[body_id] = self.sim.model.site_pos[self.sim.model.site_name2id(name)]
            self.sim.forward()

    def reset(self):
        self.goal = self._sample_goal().copy()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        # Move the target0 site to the new goal; otherwise only render() does it, and success() reads the site.
        self._render_callback()
        obs = self._get_obs()
        return obs

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'Sparse':
            return -np.array(not self.success()).astype(np.float32)
        else:
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            goal_dist = np.linalg.norm(grip_pos.copy() - self.sim.data.get_site_xpos(f'target0').copy())
            return np.exp(-20 * goal_dist)

    def goal_distance(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        return np.linalg.norm(grip_pos.copy() - self.sim.data.get_site_xpos(f'target0').copy())


class ManipulateEnv(MultiObjectFetchEnv):
    # Subtask definitions for the evaluation metrics (see docs/metrics.md).
    FINGERS = ('robot0:l_gripper_finger_link', 'robot0:r_gripper_finger_link')
    LIFT_THRESHOLD = 0.01  # Touching nothing but the fingers and this far above its resting height.
    NEEDS_LIFT_HEIGHT = 0.05  # Goals at least this high cannot be reached by pushing (= SUCCESS_THRESHOLD).
    GRASP_GAP_TOLERANCE = 0.005  # Finger gap may differ this much from the object width (soft contacts).
    GRASP_CENTER_TOLERANCE = 0.01  # Object center this close to the midpoint between the fingers.
    PLACED_SPEED = 0.01  # A placed object moves slower than this (m/s).
    NUM_TASK_OBJECTS = 1  # Blocks the task needs (object0, ...); the distractors come after them.

    def __init__(self, initial_qpos: Dict[str, Any], obs_type: str = "dictstate", object_size: float = 0.045,
                 target_size: float = 0.04, robot_configuration: str = "default", viewpoint: str = "frontview",
                 num_distractors: int = 1, reward_type: str = "Dense", task: str = "Red", width: int = 512,
                 height: int = 512, target_in_the_air=True, block_gripper=False,
                 environment: str = "Pick") -> None:
        super().__init__(
            initial_qpos, obs_type, object_size, target_size, robot_configuration, viewpoint,
            environment=environment, task=task,
        )
        self.num_distractors = num_distractors
        self.num_blocks = num_distractors + self.NUM_TASK_OBJECTS
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.width = width
        self.height = height

    
        fetch_cache = prepare_fetch_assets()
        with tempfile.NamedTemporaryFile(mode='wt', dir=fetch_cache / "fetch", delete=False, suffix=".xml") as fp:
            fp.write(self._model_xml())
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=block_gripper, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=target_in_the_air, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=self.target_size + self.object_size,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=0)
        gym_utils.EzPickle.__init__(self, initial_qpos, obs_type, object_size, target_size, robot_configuration,
                                    viewpoint, num_distractors, reward_type, task, width, height, target_in_the_air,
                                    block_gripper, environment)
        os.remove(MODEL_XML_PATH)

    def _model_xml(self):
        return generate_multi_camera_xml(self.num_blocks, self.robot, task='pick_place', target_size=self.target_size,
                                         object_size=self.object_size)

    def _reset_sim(self):
        super()._reset_sim()

        # Randomize color of objects
        if self.task == 'Odd':
            colors = self.np_random.choice(range(len(self.colors)), size=2, replace=False)

            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = self.colors[colors[0]]

            for object_name in self.object_names[1:]:
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = self.colors[colors[1]]

        elif self.task == 'OddGroups':
            colors = self.np_random.choice(range(len(self.colors)), size=3, replace=False)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = self.colors[colors[0]]

            for i, object_name in enumerate(self.object_names[1:]):
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = self.colors[colors[1]] if i % 2 == 0 else self.colors[colors[2]]

        # Randomize colors of distractors
        elif self.task == 'Red':
            target_color = np.array([255, 0, 0])
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = target_color / 255

            def generate_number_excluding_7(max_number):
                numbers = np.arange(max_number)  # Create an array with numbers from 0 to max_number
                numbers = np.delete(numbers, 7)  # Remove the number 7 from the array
                return self.np_random.choice(numbers)  # Randomly choose a number from the array

            for i in range(self.num_blocks - 1):
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[i+1])][:3] = self.colors[
                    generate_number_excluding_7(len(self.colors))]

        elif self.task == 'Reddest':
           # Sample as many colors as there are blocks
           sampled_colors = self.colors[self.np_random.choice(range(len(self.colors)), size=self.num_blocks, replace=False)]

           # Find the reddest color using Delta E 2000
           reddest_color = min(sampled_colors, key=lambda color: self.delta_e_2000_to_red(color))

           # Assign the reddest color to the target block
           self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = reddest_color

           # Assign the other colors to distractors
           distractor_colors = [c for c in sampled_colors if not np.array_equal(c, reddest_color)]
           for i, object_name in enumerate(self.object_names[1:]):
               self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = distractor_colors[i]

        elif self.task == 'Multicolor':
            assert len(self.color_names) == len(self.colors)
            color_indices = self.np_random.choice(
                len(self.colors), size=self.num_blocks, replace=False
            )
            target_idx = int(color_indices[0])
            self.target_color = target_idx
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(self.object_names[0])][:3] = (
                self.colors[target_idx]
            )
            self.instruction = build_instruction(
                self.environment, self.task, color_name=self.color_names[target_idx]
            )
            for i, object_name in enumerate(self.object_names[1:]):
                distractor_idx = int(color_indices[i + 1])
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = (
                    self.colors[distractor_idx]
                )

        else:
            raise ValueError("Invalid task: {}".format(self.task))

        self._place_objects()
        return True

    def _place_objects(self, object_names=None):
        """Randomize start position of objects (default: all blocks), at least a block diagonal apart."""
        object_names = self.object_names if object_names is None else object_names
        prev_obj_xpos = []

        valid_configuration = False
        failed_attempts = 0
        while not valid_configuration:
            for obj_name in object_names:
                for block_attempt in range(0, 100):
                    object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    if np.all([np.linalg.norm(object_xypos - other_xpos) >= 2 * np.sqrt(2) * self.object_size for other_xpos in prev_obj_xpos]):
                        break

                if not np.all([np.linalg.norm(object_xypos - other_xpos) >= 2 * np.sqrt(2) * self.object_size for other_xpos in prev_obj_xpos]):
                    failed_attempts += 1
                    if failed_attempts > 100:
                        raise ValueError("Failed to find a valid configuration after 100 attempts.")
                    prev_obj_xpos.clear()
                    break
                else:
                    # If a valid configuration has been found for the last object.
                    failed_attempts = 0
                    if len(prev_obj_xpos) == len(object_names) - 1:
                        valid_configuration = True

                prev_obj_xpos.append(object_xypos)
                self._set_object_xy(obj_name, object_xypos)

    def _set_object_xy(self, obj_name, xy):
        """Put a block on the table at xy."""
        object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = xy
        object_qpos[2] = self.height_offset
        self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
        self.sim.forward()

    def _sample_goal(self):
        goal = self.sim.data.get_joint_qpos(F"{self.object_names[0]}:joint")[:3]
        object_positions = [self.sim.data.get_joint_qpos(F"{obj_name}:joint") for obj_name in self.object_names]
        while not (np.all([np.linalg.norm(goal - other_xpos[:3]) >= self.target_size + np.sqrt(2) * self.object_size for other_xpos in object_positions])):
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.3)
        return goal.copy()

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'Sparse':
            return -np.array(not self.success()).astype(np.float32)
        else:
            grip_dist = np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy() - np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy())
            goal_dist = np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy() - np.squeeze(self.sim.data.get_site_xpos('target0')).copy())
            return 0.9 * np.exp(-20 * goal_dist) + 0.1 * np.exp(-10 * grip_dist)

    def goal_distance(self):
        return np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos(f'target0').copy()) - np.squeeze(self.sim.data.get_site_xpos('object0'))).copy()

    def episode_info(self):
        info = super().episode_info()
        # Height of the goal above a block resting on the table (resting blocks sit within 1 mm of height_offset).
        info["goal_height"] = float(self.goal[2] - self.height_offset)
        info["goal_needs_lift"] = info["goal_height"] >= self.NEEDS_LIFT_HEIGHT
        return info

    def step_info(self):
        """Per-step subtask flags for object0.

        grasped: both fingers are in contact with object0 (MuJoCo contacts) and object0 is between
            them: the finger gap equals its width along the closing axis and its center lies on
            the line between the fingers. Two fingers pressed on one face (pushing) do not count.
        grasped_geometric: stricter cross-check from geometry only: the gap and center conditions,
            and the center also lies within the finger pads' extent (rejects edge pinches).
        lifted: object0 is above its resting height and touches nothing except the fingers (no
            table, no other block): carried, but also falling or knocked into the air.
        held: grasped and lifted.
        placed: success and, for a goal that needs lifting, held at the goal; otherwise
            released (not grasped), resting on the table and still.
        """
        info = super().step_info()
        object_pos = np.squeeze(self.sim.data.get_site_xpos('object0')).copy()
        grip_pos = np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy()
        info["object_lift"] = float(object_pos[2] - self.height_offset)
        info["grip_object_dist"] = float(np.linalg.norm(grip_pos - object_pos))
        info["object_speed"] = float(np.linalg.norm(self.sim.data.get_site_xvelp('object0')))

        contacts = self._object0_contacts()
        fingers = contacts & set(self.FINGERS)
        others = contacts - set(self.FINGERS)
        info["object_contacts"] = sorted(str(name) for name in contacts)
        geometry = self._grasp_geometry()
        info.update(geometry)
        info["grasped"] = len(fingers) == 2 and geometry["between_fingers"]
        info["lifted"] = not others and info["object_lift"] > self.LIFT_THRESHOLD
        info["held"] = info["grasped"] and info["lifted"]
        if self.episode_info()["goal_needs_lift"]:
            info["placed"] = info["dist"] < self.SUCCESS_THRESHOLD and info["held"]
        else:
            info["placed"] = (info["dist"] < self.SUCCESS_THRESHOLD and not info["grasped"]
                              and "table0" in others and info["object_speed"] < self.PLACED_SPEED)
        return info

    def _object0_contacts(self):
        """Names of the geoms in contact with object0 (None for unnamed geoms)."""
        model, data = self.sim.model, self.sim.data
        obj = model.geom_name2id('object0')
        names = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == obj:
                names.add(model.geom_id2name(contact.geom2))
            elif contact.geom2 == obj:
                names.add(model.geom_id2name(contact.geom1))
        return names

    @staticmethod
    def _half_extent(size, xmat, axis):
        """Half-extent of a box geom along a world axis (accounts for the box's rotation)."""
        return float(np.sum(size * np.abs(xmat.reshape(3, 3).T @ axis)))

    def _grasp_geometry(self):
        model, data = self.sim.model, self.sim.data
        left, right = (model.geom_name2id(name) for name in self.FINGERS)
        obj = model.geom_name2id('object0')
        p_left, p_right = data.geom_xpos[left].copy(), data.geom_xpos[right].copy()
        axis = (p_left - p_right) / np.linalg.norm(p_left - p_right)  # Closing axis.
        gap = (np.linalg.norm(p_left - p_right) - self._half_extent(model.geom_size[left], data.geom_xmat[left], axis)
               - self._half_extent(model.geom_size[right], data.geom_xmat[right], axis))
        width = 2 * self._half_extent(model.geom_size[obj], data.geom_xmat[obj], axis)
        offset = data.geom_xpos[obj] - (p_left + p_right) / 2
        pad = data.geom_xmat[left].reshape(3, 3)
        closing = int(np.argmax(np.abs(pad.T @ axis)))
        within_pads = all(abs(offset @ pad[:, i]) <= model.geom_size[left][i] for i in range(3) if i != closing)
        between = abs(gap - width) < self.GRASP_GAP_TOLERANCE and abs(offset @ axis) < self.GRASP_CENTER_TOLERANCE
        return {"finger_gap": float(gap), "object_width": float(width), "grasp_center_offset": float(offset @ axis),
                "between_fingers": bool(between), "grasped_geometric": bool(between and within_pads)}


class StackEnv(ManipulateEnv):
    """Put object0 on top of a tower object1, ..., objectN (object1 at the bottom; N = 1 for Onto1, 2 for Onto2).

    The base is the tower's top block, the one the instruction names. Blocks after the tower are loose distractors.
    Fixed tasks: the whole tower is welded in place. Otherwise every tower block is free and can be pushed or
    knocked over; that does not end the episode. There is no target location: the goal
    is the top of the base, wherever the base is. The episode ends (failed) when object0 or a tower block falls
    off the table.
    """
    TOWER_HEIGHTS = {'Onto1': 1, 'Onto1Fixed': 1, 'Onto2': 2, 'Onto2Fixed': 2}
    ON_TOP_TOLERANCE = 0.01  # object0 sits one block height (2 * object_size) above the base, within this.
    TOWER_GRIPPER_CLEARANCE = 0.12  # A tower of 2+ blocks starts at least this far (xy) from the gripper.

    @property
    def tower_height(self):
        if self.task not in self.TOWER_HEIGHTS:
            raise ValueError("Invalid task: {}".format(self.task))
        return self.TOWER_HEIGHTS[self.task]

    @property
    def NUM_TASK_OBJECTS(self):
        return 1 + self.tower_height

    @property
    def tower_names(self):
        return ['object{}'.format(i) for i in range(1, self.tower_height + 1)]

    @property
    def base_name(self):
        return self.tower_names[-1]

    @property
    def fixed_base(self):
        """The tower blocks are welded in place."""
        return self.task.endswith('Fixed')

    def _model_xml(self):
        fixed_blocks = tuple(range(1, self.tower_height + 1)) if self.fixed_base else ()
        return generate_multi_camera_xml(self.num_blocks, self.robot, task='pick_place', target_size=self.target_size,
                                         object_size=self.object_size, fixed_blocks=fixed_blocks)

    def _reset_sim(self):
        MultiObjectFetchEnv._reset_sim(self)

        color_indices = self.np_random.choice(len(self.colors), size=self.num_blocks, replace=False)
        self.target_color, self.base_color = int(color_indices[0]), int(color_indices[self.tower_height])
        for object_name, color_idx in zip(self.object_names, color_indices):
            self.sim.model.geom_rgba[self.sim.model.geom_name2id(object_name)][:3] = self.colors[color_idx]
        self.instruction = build_instruction(self.environment, self.task, color_name=self.color_names[self.target_color],
                                             base_color_name=self.color_names[self.base_color])
        # No target location to show.
        self.sim.model.site_rgba[self.sim.model.site_name2id('target0')][3] = 0

        self._place_objects()
        self.base_start_xy = np.squeeze(self.sim.data.get_site_xpos(self.base_name))[:2].copy()
        return True

    def _place_objects(self):
        """Scatter object0, the tower's bottom block and the distractors, then stack the tower on its bottom block.

        A tower of two or more blocks reaches the gripper's start height, so its bottom block is kept
        TOWER_GRIPPER_CLEARANCE (xy) away from the gripper's start; otherwise the first steps sweep it over.
        """
        upper = self.tower_names[1:]
        for _ in range(100):
            super()._place_objects([name for name in self.object_names if name not in upper])
            bottom_xy = np.squeeze(self.sim.data.get_site_xpos('object1'))[:2].copy()
            if not upper or np.linalg.norm(bottom_xy - self.initial_gripper_xpos[:2]) >= self.TOWER_GRIPPER_CLEARANCE:
                break
        else:
            raise ValueError("Failed to place the tower clear of the gripper after 100 attempts.")
        for level, name in enumerate(upper, start=1):
            self._set_object_xy(name, bottom_xy, level)

    def _set_object_xy(self, obj_name, xy, level=0):
        """Put a block at xy, level block heights above the table (0: on the table)."""
        z = self.height_offset + 2 * self.object_size * level
        if obj_name in self.tower_names and self.fixed_base:
            self.sim.model.body_pos[self.sim.model.body_name2id(obj_name)] = [xy[0], xy[1], z]
        else:
            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            object_qpos[:3] = [xy[0], xy[1], z]
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
        self.sim.forward()

    def _sample_goal(self):
        """Where object0's center is when it sits squarely on the base."""
        return np.squeeze(self.sim.data.get_site_xpos(self.base_name)).copy() + [0, 0, 2 * self.object_size]

    def _step_callback(self):
        super()._step_callback()
        self.goal = self._sample_goal()  # A free base can move.

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # A block the stack needs fell off the table: the task cannot be finished any more.
        return obs, reward, done or info.get("dropped", False), info

    def _on_top(self, upper_pos, lower_pos):
        """upper's center is within the half-diagonal (sqrt(2) * object_size) of lower's center horizontally, so
        anywhere over its top face including the corners, and one block height above it."""
        offset = upper_pos - lower_pos
        return (np.linalg.norm(offset[:2]) < np.sqrt(2) * self.object_size
                and abs(offset[2] - 2 * self.object_size) < self.ON_TOP_TOLERANCE)

    def stack_info(self):
        """on_base: object0 is on top of the base (see _on_top; it may still be held). Offsets past an edge fit
            the circle but cannot rest: placed also needs contact and stillness.
        base_resting: the tower stands: its bottom block rests on the table (welded, or a free block on any face,
            not lifted) and every block above it is on top of the one below.
        placed: on_base, base_resting, object0 touches the base, is released (not grasped) and still.
        dropped: object0 or a tower block fell off the table (center below the table top).
        """
        object_pos = np.squeeze(self.sim.data.get_site_xpos('object0')).copy()
        tower_pos = [np.squeeze(self.sim.data.get_site_xpos(name)).copy() for name in self.tower_names]
        contacts = self._object0_contacts()
        grasped = len(contacts & set(self.FINGERS)) == 2 and self._grasp_geometry()["between_fingers"]
        on_base = self._on_top(object_pos, tower_pos[-1])
        base_resting = (abs(tower_pos[0][2] - self.height_offset) < self.ON_TOP_TOLERANCE
                        and all(self._on_top(upper, lower) for lower, upper in zip(tower_pos, tower_pos[1:])))
        speed = np.linalg.norm(self.sim.data.get_site_xvelp('object0'))
        table_top = self.height_offset - self.object_size
        return {"on_base": bool(on_base), "base_resting": bool(base_resting), "grasped": bool(grasped),
                "placed": bool(on_base and base_resting and self.base_name in contacts and not grasped
                               and speed < self.PLACED_SPEED),
                "dropped": bool(min(pos[2] for pos in [object_pos] + tower_pos) < table_top)}

    def success(self):
        return self.stack_info()["placed"]

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'Sparse':
            return -np.array(not self.success()).astype(np.float32)
        if self.reward_type == 'Staged':
            return self.staged_reward()
        if self.reward_type == 'ManiSkill':
            return self.maniskill_reward()
        # Dense: 1 once placed; until then at most 0.9, so holding object0 on the base is worth less than letting go.
        if self.success():
            return 1.0
        grip_dist = np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy() - np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy())
        return 0.8 * np.exp(-20 * self.goal_distance()) + 0.1 * np.exp(-10 * grip_dist)

    def staged_reward(self):
        """Stepped dense reward (reward_type Staged): each stage of the stack sets a floor above the previous one.

        reach object0: 0..0.2 (1 - tanh(5 * gripper distance)); grasped: 0.4; grasped and lifted to at least the
        stacking height: 0.5, plus up to 0.2 for horizontal alignment over the base (1 - tanh(10 * xy distance)) and,
        once over the top face, up to 0.1 for lowering onto it (1 - tanh(10 * height above the stacked position));
        on_base: 0.8 while held, 0.9 once released (so letting go never lowers the reward); placed: 1.
        """
        state = self.stack_info()
        if state["placed"]:
            return 1.0
        if state["on_base"]:
            return 0.8 if state["grasped"] else 0.9
        object_pos = np.squeeze(self.sim.data.get_site_xpos('object0')).copy()
        if not state["grasped"]:
            grip_dist = np.linalg.norm(object_pos - np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy())
            return 0.2 * (1 - np.tanh(5 * grip_dist))
        reward = 0.4
        offset = object_pos - np.squeeze(self.sim.data.get_site_xpos(self.base_name)).copy()
        height = offset[2] - 2 * self.object_size  # Above the stacked position.
        if height > -self.ON_TOP_TOLERANCE:
            xy_dist = np.linalg.norm(offset[:2])
            reward += 0.1 + 0.2 * (1 - np.tanh(10 * xy_dist))
            if xy_dist < np.sqrt(2) * self.object_size:
                reward += 0.1 * (1 - np.tanh(10 * max(height, 0.0)))
        return reward

    FINGER_JOINTS = ('robot0:l_gripper_finger_joint', 'robot0:r_gripper_finger_joint')

    def maniskill_reward(self):
        """ManiSkill StackCube-v1 normalized dense reward (compute_dense_reward / 8), ported (reward_type ManiSkill).

        Later cases override earlier ones, as in ManiSkill:
        reach: 2 * (1 - tanh(5 * gripper-to-object0 distance));
        grasped: 4 + (1 - tanh(5 * object0-to-goal distance)), goal = one block height above the base;
        on_base: 6 + (ungrasp + static) / 2, ungrasp = finger opening / maximum opening (1 once released),
            static = 1 - tanh(10 * linear speed + angular speed);
        placed: 8.
        Grasped, on_base and placed are this env's definitions (see stack_info), not ManiSkill's thresholds.
        """
        state = self.stack_info()
        if state["placed"]:
            return 1.0
        if state["on_base"]:
            if state["grasped"]:
                opening = sum(self.sim.data.get_joint_qpos(name) for name in self.FINGER_JOINTS)
                ungrasp = opening / sum(self.sim.model.jnt_range[self.sim.model.joint_name2id(name)][1]
                                        for name in self.FINGER_JOINTS)
            else:
                ungrasp = 1.0
            speed = np.linalg.norm(self.sim.data.get_site_xvelp('object0'))
            angular_speed = np.linalg.norm(self.sim.data.get_site_xvelr('object0'))
            static = 1 - np.tanh(10 * speed + angular_speed)
            return (6 + (ungrasp + static) / 2) / 8
        if state["grasped"]:
            return (4 + 1 - np.tanh(5 * self.goal_distance())) / 8
        grip_dist = np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy()
                                   - np.squeeze(self.sim.data.get_site_xpos('robot0:grip')).copy())
        return 2 * (1 - np.tanh(5 * grip_dist)) / 8

    def goal_distance(self):
        return float(np.linalg.norm(np.squeeze(self.sim.data.get_site_xpos('object0')).copy() - self._sample_goal()))

    def episode_info(self):
        info = super().episode_info()
        info["goal_height"] = 2 * self.object_size * self.tower_height  # Always on top of a standing tower.
        info["goal_needs_lift"] = True
        info["base_color"] = self.base_color
        info["fixed_base"] = self.fixed_base
        info["tower_height"] = self.tower_height
        return info

    def step_info(self):
        """ManipulateEnv's flags for object0, with placed meaning placed on the base (see stack_info)."""
        info = super().step_info()
        info.update(self.stack_info())
        base_pos = np.squeeze(self.sim.data.get_site_xpos(self.base_name)).copy()
        info["base_shift"] = float(np.linalg.norm(base_pos[:2] - self.base_start_xy))
        return info
