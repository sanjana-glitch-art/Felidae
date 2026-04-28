import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

#  CELL TYPE CONSTANTS  (what each grid cell can be)
EMPTY       = 0
WALL        = 1
SUNNY_SPOT  = 2   # window / warm patch 
FOOD_BOWL   = 3   # eating station
TOY         = 4   
TABLE_OBJ   = 5   # knockable objects on tables
BOX         = 6   
VET_CARRIER = 7   # episode-ending danger zone
HUMAN       = 8   # human cell (moves or stays fixed)

#  COLOUR MAP  for pygame rendering
CELL_COLORS = {
    EMPTY:       (245, 245, 220),   # beige floor
    WALL:        (80,  80,  80),    # dark grey
    SUNNY_SPOT:  (255, 230, 100),   # warm yellow
    FOOD_BOWL:   (100, 200, 100),   # green
    TOY:         (255, 150, 50),    # orange
    TABLE_OBJ:   (180, 140, 100),   # brown
    BOX:         (210, 170, 90),    # cardboard tan
    VET_CARRIER: (200, 60,  60),    # alarming red
    HUMAN:       (150, 180, 255),   # soft blue
}
CAT_COLOR   = (80,  60,  120)       # purple cat
CELL_SIZE   = 60                    # pixels per grid cell


class CatEnv(gym.Env):
    """
    CatRL — a 12×12 grid-world where an RL agent learns to be a cat.

    OBSERVATION SPACE  (12 floats, all normalised to [0, 1])
    ─────────────────────────────────────────────────────────
    0  cat_x                      — column / (GRID_W - 1)
    1  cat_y                      — row    / (GRID_H - 1)
    2  hunger_level               — 0 = full, 1 = starving
    3  energy_level               — 0 = exhausted, 1 = full energy
    4  time_of_day                — 0 = midnight, 1 = just-before-midnight
    5  nearest_food_distance      — Manhattan dist normalised by grid diagonal
    6  nearest_sunny_dist         — same
    7  nearest_toy_dist           — same
    8  human_distance             — same  (0 if no human on map)
    9  is_vet_visible             — 1 if vet carrier is in a 3-cell radius
    10 is_in_box                  — 1 if cat is currently on a BOX cell
    11 is_meow_cooldown           — 1 if meow is on cooldown (can't be used yet)

    ACTION SPACE  (14 discrete actions)
    ────────────────────────────────────
    0  move_up        move one cell up
    1  move_down      move one cell down
    2  move_left      move one cell left
    3  move_right     move one cell right
    4  zoom           move TWO cells in last direction, costs extra energy
    5  nap            rest; restores energy; bonus if on sunny spot
    6  eat            consume food; only rewarded if near food bowl
    7  hunt_toy       play with toy; only rewarded if near toy
    8  knock_object   knock thing off table; only rewarded if near TABLE_OBJ
    9  sit_in_box     settle into box; only rewarded if on BOX cell
    10 meow           call for food; summons a tiny hunger relief (cooldown 10 steps)
    11 seek_human     move toward nearest human cell
    12 ignore_human   stand still and pointedly ignore the human
    13 stare_at_wall  do nothing; pure vibe

    REWARD STRUCTURE  (see _compute_reward for full details)
    ──────────────────────────────────────────────────────────
    nap on sunny spot         +10
    knock object              +7
    hunt toy                  +5
    eat when hungry           +4  (scaled by hunger level)
    sit in box                +3  (bonus +2 per continued step in box)
    meow (hunger relief)      +2
    seek/ignore human         ±small, depends on energy
    every step (time penalty) -0.5
    high hunger untreated     -3
    being dragged (held)      -6
    entering vet carrier      -10  → episode ends immediately
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # ── grid dimensions ────────────────────────────────────────
    GRID_H = 12
    GRID_W = 12

    # ── episode length ─────────────────────────────────────────
    MAX_STEPS = 500

    # ── meow cooldown (steps) ──────────────────────────────────
    MEOW_COOLDOWN = 10

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── spaces ────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(14)

        # ── build the fixed grid map ───────────────────────────
        self._base_map = self._build_map()

        # ── pygame objects (created lazily on first render) ───
        self._screen = None
        self._clock  = None

        # ── state variables (set properly in reset()) ─────────
        self.cat_pos        = np.array([1, 1])   # [row, col]
        self.hunger         = 0.0
        self.energy         = 1.0
        self.time_of_day    = 0.0
        self.meow_cooldown  = 0
        self.steps          = 0
        self.last_direction = np.array([-1, 0])  # up by default
        self.in_box_streak  = 0                  # consecutive steps sitting in box

    # ──────────────────────────────────────────────────────────
    #  MAP BUILDER
    # ──────────────────────────────────────────────────────────
    def _build_map(self):
        """
        Returns a 12×12 numpy array.
        The coordinate system is (row, col) where (0,0) is top-left.

        Layout sketch:
          W = wall border
          . = empty floor
          S = sunny spot (window)
          F = food bowl
          T = toy
          O = table object (knockable)
          B = box
          V = vet carrier (danger)
          H = human starting cell
        """
        m = np.zeros((self.GRID_H, self.GRID_W), dtype=np.int32)

        # --- border walls ---
        m[0,  :]  = WALL
        m[-1, :]  = WALL
        m[:,  0]  = WALL
        m[:, -1]  = WALL

        # --- interior walls (room dividers) ---
        m[5, 1:6] = WALL     # horizontal divider top room / bottom room
        m[5, 6]   = EMPTY    # doorway

        # --- sunny spots (windows, top-right area) ---
        for r, c in [(1, 9), (1, 10), (2, 9), (2, 10)]:
            m[r, c] = SUNNY_SPOT

        # --- food bowls ---
        m[8, 2]  = FOOD_BOWL
        m[9, 9]  = FOOD_BOWL

        # --- toys ---
        m[2, 3]  = TOY
        m[7, 7]  = TOY

        # --- table objects (knockable) ---
        m[1, 5]  = TABLE_OBJ
        m[3, 8]  = TABLE_OBJ
        m[6, 3]  = TABLE_OBJ

        # --- boxes ---
        m[4, 2]  = BOX
        m[9, 5]  = BOX

        # --- vet carrier (top-left corner room) ---
        m[1, 1]  = VET_CARRIER

        # --- human position (fixed; could be made dynamic later) ---
        m[10, 8] = HUMAN

        return m

    # ──────────────────────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self._base_map.copy()

        # Place the cat at a safe starting position (row=3, col=5)
        self.cat_pos       = np.array([3, 5])
        self.hunger        = 0.0          # starts full
        self.energy        = 1.0          # starts rested
        self.time_of_day   = 0.0          # midnight
        self.meow_cooldown = 0
        self.steps         = 0
        self.last_direction = np.array([-1, 0])
        self.in_box_streak  = 0

        obs  = self._get_obs()
        info = {}
        return obs, info

    # ──────────────────────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────────────────────
    def step(self, action):
        self.steps      += 1
        reward           = -1.0           # time penalty every step
        terminated       = False
        truncated        = False

        # ── time of day ticks forward ─────────────────────────
        self.time_of_day = (self.time_of_day + 1 / self.MAX_STEPS) % 1.0

        # ── hunger slowly rises; energy slowly drains ─────────
        self.hunger  = min(1.0, self.hunger  + 0.002)
        self.energy  = max(0.0, self.energy  - 0.001)

        # ── decrement meow cooldown ───────────────────────────
        if self.meow_cooldown > 0:
            self.meow_cooldown -= 1

        # ─────────────────────────────────────────
        #  ACTION DISPATCH
        # ─────────────────────────────────────────
        if action in (0, 1, 2, 3):          # directional moves
            reward += self._act_move(action)

        elif action == 4:                   # zoom
            reward += self._act_zoom()

        elif action == 5:                   # nap
            reward += self._act_nap()

        elif action == 6:                   # eat
            reward += self._act_eat()

        elif action == 7:                   # hunt toy
            reward += self._act_hunt_toy()

        elif action == 8:                   # knock object
            reward += self._act_knock_object()

        elif action == 9:                   # sit in box
            reward += self._act_sit_in_box()

        elif action == 10:                  # meow
            reward += self._act_meow()

        elif action == 11:                  # seek human
            reward += self._act_seek_human()

        elif action == 12:                  # ignore human
            reward += self._act_ignore_human()

        elif action == 13:                  # stare at wall — classic cat
            reward += 0.0                   # nothing happens. intentionally.

        # ── check if cat walked into vet carrier ──────────────
        r, c = self.cat_pos
        if self.grid[r, c] == VET_CARRIER:
            reward     += -10.0
            terminated  = True

        # ── starvation penalty ────────────────────────────────
        if self.hunger >= 0.8:
            reward += -3.0

        # ── truncate after MAX_STEPS ──────────────────────────
        if self.steps >= self.MAX_STEPS:
            truncated = True

        obs  = self._get_obs()
        info = {"hunger": self.hunger, "energy": self.energy,
                "steps": self.steps, "cat_pos": self.cat_pos.copy()}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    #  INDIVIDUAL ACTION HANDLERS
    # ──────────────────────────────────────────────────────────
    def _try_move(self, delta):
        """Attempt to move by delta=[dr, dc]. Returns True if successful."""
        new_pos = self.cat_pos + delta
        r, c    = new_pos
        if 0 <= r < self.GRID_H and 0 <= c < self.GRID_W:
            if self.grid[r, c] != WALL:
                self.cat_pos       = new_pos
                self.last_direction = delta
                return True
        return False

    # Direction deltas: up, down, left, right
    _DELTAS = {0: np.array([-1, 0]),
               1: np.array([ 1, 0]),
               2: np.array([ 0,-1]),
               3: np.array([ 0, 1])}

    def _act_move(self, action):
        delta = self._DELTAS[action]
        moved=self._try_move(delta)
        self.in_box_streak = 0
        return 0.1 if moved else -0.5 # small reward for moving, penalty for bumping into wall

    def _act_zoom(self):
        """Move 2 cells in last direction. Costs extra energy (3am zoomies)."""
        if self.energy < 0.2:
            return -1.0                  # too tired to zoom
        self._try_move(self.last_direction)
        self._try_move(self.last_direction)
        self.energy        -= 0.1        # zoom costs extra energy
        self.in_box_streak  = 0
        return 1.0                       # minor reward for chaotic movement

    def _act_nap(self):
        self.energy = min(1.0, self.energy + 0.15)
        r, c = self.cat_pos
        if self.grid[r, c] == SUNNY_SPOT:
            self.energy = min(1.0, self.energy + 0.05)  # extra cosy
            return 10.0
        return 1.5                       # napping anywhere is still okay

    def _act_eat(self):
        dist = self._dist_to(FOOD_BOWL)
        if dist == 0:                    # cat is ON the food bowl
            reduction    = min(self.hunger, 0.4)
            self.hunger -= reduction
            return 4.0 + self.hunger * 2.0   # hungrier → bigger reward
        elif dist <= 2:
            return -0.5                  # near food but not on it — frustrating
        return -1.0                      # nowhere near food

    def _act_hunt_toy(self):
        dist = self._dist_to(TOY)
        if dist == 0:
            self.energy = max(0.0, self.energy - 0.05)
            return 5.0
        elif dist <= 2:
            return 1.0                   # approaching toy is fine
        return -0.5

    def _act_knock_object(self):
        dist = self._dist_to(TABLE_OBJ)
        if dist <= 1:                    # adjacent to or on table object
            return 7.0                   # maximum cat satisfaction
        return -0.5

    def _act_sit_in_box(self):
        r, c = self.cat_pos
        if self.grid[r, c] == BOX:
            self.in_box_streak += 1
            bonus = min(2.0, self.in_box_streak * 0.5)  # streak bonus, capped
            return 3.0 + bonus
        self.in_box_streak = 0
        return -1.0                      # trying to sit in non-existent box

    def _act_meow(self):
        if self.meow_cooldown > 0:
            return -1.0                  # spam penalty
        self.meow_cooldown = self.MEOW_COOLDOWN
        self.hunger        = max(0.0, self.hunger - 0.1)   # human brings snack
        return 2.0

    def _act_seek_human(self):
        human_pos = self._find_cell(HUMAN)
        if human_pos is None:
            return -0.5
        direction = np.sign(human_pos - self.cat_pos)
        self._try_move(direction)
        dist = self._dist_to(HUMAN)
        if dist == 0:
            return 1.5                   # reached human
        return 0.5                       # moving toward human

    def _act_ignore_human(self):
        """
        Reward based on energy level — tired cat wants to be left alone.
        High energy → ignoring is slightly bad. Low energy → ignoring is good.
        """
        if self.energy < 0.3:
            return 2.0                   # tired cat ignoring human = valid
        return -0.5                      # full-energy cat should be doing something

    # ──────────────────────────────────────────────────────────
    #  OBSERVATION BUILDER
    # ──────────────────────────────────────────────────────────
    def _get_obs(self):
        diag = float(self.GRID_H + self.GRID_W)   # normaliser for distances

        cat_r, cat_c = self.cat_pos
        food_dist    = self._dist_to(FOOD_BOWL)   / diag
        sunny_dist   = self._dist_to(SUNNY_SPOT)  / diag
        toy_dist     = self._dist_to(TOY)          / diag
        human_dist   = self._dist_to(HUMAN)        / diag

        # vet visible = vet carrier within Manhattan distance of 3
        vet_dist     = self._dist_to(VET_CARRIER)
        vet_visible  = 1.0 if vet_dist <= 3 else 0.0

        is_in_box    = 1.0 if self.grid[cat_r, cat_c] == BOX else 0.0
        meow_cd      = 1.0 if self.meow_cooldown > 0 else 0.0

        obs = np.array([
            cat_c / (self.GRID_W - 1),  # normalise x (col)
            cat_r / (self.GRID_H - 1),  # normalise y (row)
            self.hunger,
            self.energy,
            self.time_of_day,
            food_dist,
            sunny_dist,
            toy_dist,
            human_dist,
            vet_visible,
            is_in_box,
            meow_cd,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────
    #  UTILITY HELPERS
    # ──────────────────────────────────────────────────────────
    def _dist_to(self, cell_type):
        """Manhattan distance from cat to nearest cell of given type."""
        cat_r, cat_c = self.cat_pos
        positions    = np.argwhere(self.grid == cell_type)
        if len(positions) == 0:
            return self.GRID_H + self.GRID_W   # impossibly far
        dists = np.abs(positions[:, 0] - cat_r) + np.abs(positions[:, 1] - cat_c)
        return int(dists.min())

    def _find_cell(self, cell_type):
        """Returns [row, col] of nearest cell of given type, or None."""
        cat_r, cat_c = self.cat_pos
        positions    = np.argwhere(self.grid == cell_type)
        if len(positions) == 0:
            return None
        dists = np.abs(positions[:, 0] - cat_r) + np.abs(positions[:, 1] - cat_c)
        return positions[np.argmin(dists)]

    # ──────────────────────────────────────────────────────────
    #  RENDER  (pygame)
    # ──────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("CatRL")
            w = self.GRID_W * CELL_SIZE
            h = self.GRID_H * CELL_SIZE + 60   # extra strip for HUD
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w, h))
            else:
                self._screen = pygame.Surface((w, h))
            self._clock = pygame.time.Clock()

        self._screen.fill((30, 30, 30))

        # draw cells
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                cell  = self.grid[r, c]
                color = CELL_COLORS.get(cell, (245, 245, 220))
                rect  = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE,
                                    CELL_SIZE - 1, CELL_SIZE - 1)
                pygame.draw.rect(self._screen, color, rect, border_radius=4)

        # draw cat (filled circle)
        cr, cc = self.cat_pos
        cx     = cc * CELL_SIZE + CELL_SIZE // 2
        cy     = cr * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self._screen, CAT_COLOR, (cx, cy), CELL_SIZE // 3)

        # draw HUD  (hunger / energy bars)
        font  = pygame.font.SysFont("monospace", 14)
        hud_y = self.GRID_H * CELL_SIZE + 8

        # hunger bar (red)
        pygame.draw.rect(self._screen, (200, 60, 60),
                         pygame.Rect(10, hud_y, int(self.hunger * 140), 12))
        self._screen.blit(font.render(f"Hunger", True, (220, 220, 220)),
                          (160, hud_y - 1))

        # energy bar (green)
        pygame.draw.rect(self._screen, (60, 180, 60),
                         pygame.Rect(10, hud_y + 18, int(self.energy * 140), 12))
        self._screen.blit(font.render(f"Energy", True, (220, 220, 220)),
                          (160, hud_y + 17))

        step_surf = font.render(f"Step {self.steps}/{self.MAX_STEPS}  "
                                f"Time {self.time_of_day:.2f}", True, (200, 200, 200))
        self._screen.blit(step_surf, (300, hud_y + 6))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None