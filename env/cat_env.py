import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


# ─────────────────────────────────────────────
#  CELL TYPE CONSTANTS
# ─────────────────────────────────────────────
EMPTY       = 0
WALL        = 1
SUNNY_SPOT  = 2   # window — best nap spot
FOOD_BOWL   = 3   # eating station
TOY         = 4   # hunting / play
TABLE_OBJ   = 5   # knockable objects
BOX         = 6   # cardboard box
VET_CARRIER = 7   # danger — episode ends
HUMAN       = 8   # human cell

# ─────────────────────────────────────────────
#  COLOURS  (pygame)
# ─────────────────────────────────────────────
CELL_COLORS = {
    EMPTY:       (245, 245, 220),
    WALL:        (80,  80,  80),
    SUNNY_SPOT:  (255, 230, 100),
    FOOD_BOWL:   (100, 200, 100),
    TOY:         (255, 150, 50),
    TABLE_OBJ:   (180, 140, 100),
    BOX:         (210, 170, 90),
    VET_CARRIER: (200, 60,  60),
    HUMAN:       (150, 180, 255),
}
CAT_COLOR = (80, 60, 120)
CELL_SIZE = 60


class CatEnv(gym.Env):
    """
    StochasticCat — a 12x12 grid-world where a PPO agent learns
    to behave like a real cat through state-gated rewards.

    KEY DESIGN PRINCIPLE:
    Every action only gives full reward in the correct internal state.
    This forces a natural cycle:
        energetic -> play/knock -> tired -> nap -> rested -> hungry -> eat -> repeat

    OBSERVATION SPACE (12 floats, normalised to [0,1])
    0  cat_x                  col / (GRID_W-1)
    1  cat_y                  row / (GRID_H-1)
    2  hunger                 0=full, 1=starving
    3  energy                 0=exhausted, 1=full
    4  time_of_day            0-1 cycling
    5  nearest_food_dist      Manhattan, normalised
    6  nearest_sunny_dist     Manhattan, normalised
    7  nearest_toy_dist       Manhattan, normalised
    8  human_distance         Manhattan, normalised
    9  is_vet_visible         1 if vet within 3 cells
    10 is_in_box              1 if on BOX cell
    11 is_meow_cooldown       1 if meow unavailable

    ACTION SPACE (14 discrete)
    0  move_up
    1  move_down
    2  move_left
    3  move_right
    4  zoom          2 cells, costs energy
    5  nap           best when tired + on sunny spot
    6  eat           best when hungry + on food bowl
    7  hunt_toy      best when energetic + near toy
    8  knock_object  best when bored (high energy, low hunger) + consumable
    9  sit_in_box    cosy regardless of state, mild reward
    10 meow          summons food, 20-step cooldown
    11 seek_human    rewarded when bored + energetic
    12 ignore_human  rewarded when tired
    13 stare_at_wall pure cat vibe, tiny reward always
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    GRID_H        = 12
    GRID_W        = 12
    MAX_STEPS     = 500
    MEOW_COOLDOWN = 20

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(14)

        self._base_map = self._build_map()

        self._screen = None
        self._clock  = None

        # state (properly initialised in reset)
        self.cat_pos        = np.array([3, 5])
        self.hunger         = 0.0
        self.energy         = 1.0
        self.time_of_day    = 0.0
        self.meow_cooldown  = 0
        self.steps          = 0
        self.last_direction = np.array([-1, 0])
        self.in_box_streak  = 0
        self.boredom        = 0.0
        self.grid           = self._base_map.copy()

    # ──────────────────────────────────────────────────────────
    #  MAP
    # ──────────────────────────────────────────────────────────
    def _build_map(self):
        m = np.zeros((self.GRID_H, self.GRID_W), dtype=np.int32)

        # border walls
        m[0, :]  = WALL
        m[-1, :] = WALL
        m[:, 0]  = WALL
        m[:, -1] = WALL

        # interior divider with doorway
        m[5, 1:6] = WALL
        m[5, 6]   = EMPTY

        # sunny spots — top right cluster
        for r, c in [(1, 9), (1, 10), (2, 9), (2, 10)]:
            m[r, c] = SUNNY_SPOT

        # food bowls
        m[8, 2]  = FOOD_BOWL
        m[9, 9]  = FOOD_BOWL

        # toys
        m[2, 3]  = TOY
        m[7, 7]  = TOY

        # table objects (knockable, consumable per episode)
        m[1, 5]  = TABLE_OBJ
        m[3, 8]  = TABLE_OBJ
        m[6, 3]  = TABLE_OBJ

        # boxes
        m[4, 2]  = BOX
        m[9, 5]  = BOX

        # vet carrier
        m[1, 1]  = VET_CARRIER

        # human
        m[10, 8] = HUMAN

        return m

    # ──────────────────────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # fresh grid every episode — restores knocked objects
        self.grid = self._base_map.copy()

        # random start on any empty floor cell
        while True:
            r = self.np_random.integers(1, self.GRID_H - 1)
            c = self.np_random.integers(1, self.GRID_W - 1)
            if self._base_map[r, c] == EMPTY:
                self.cat_pos = np.array([r, c])
                break

        self.hunger         = 0.0
        self.energy         = 1.0
        self.time_of_day    = 0.0
        self.meow_cooldown  = 0
        self.steps          = 0
        self.last_direction = np.array([-1, 0])
        self.in_box_streak  = 0
        self.boredom        = 0.0

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────────────────────
    def step(self, action):
        self.steps += 1
        reward      = -0.3
        terminated  = False
        truncated   = False

        # passive state changes each step
        self.time_of_day = (self.time_of_day + 1 / self.MAX_STEPS) % 1.0
        self.hunger      = min(1.0, self.hunger + 0.003)   # hungry in ~330 steps
        self.energy      = max(0.0, self.energy - 0.004)   # tired in ~250 steps
        self.boredom     = min(1.0, self.boredom + 0.005)  # bored if inactive

        if self.meow_cooldown > 0:
            self.meow_cooldown -= 1

        # action dispatch
        if action in (0, 1, 2, 3):
            reward += self._act_move(action)
        elif action == 4:
            reward += self._act_zoom()
        elif action == 5:
            reward += self._act_nap()
        elif action == 6:
            reward += self._act_eat()
        elif action == 7:
            reward += self._act_hunt_toy()
        elif action == 8:
            reward += self._act_knock_object()
        elif action == 9:
            reward += self._act_sit_in_box()
        elif action == 10:
            reward += self._act_meow()
        elif action == 11:
            reward += self._act_seek_human()
        elif action == 12:
            reward += self._act_ignore_human()
        elif action == 13:
            reward += self._act_stare_at_wall()

        # vet carrier — instant episode end
        r, c = self.cat_pos
        if self.grid[r, c] == VET_CARRIER:
            reward    += -15.0
            terminated = True

        # starvation penalty
        if self.hunger >= 0.85:
            reward += -2.0

        # exhaustion penalty
        if self.energy <= 0.0:
            reward += -1.0

        if self.steps >= self.MAX_STEPS:
            truncated = True

        obs  = self._get_obs()
        info = {
            "hunger":  self.hunger,
            "energy":  self.energy,
            "boredom": self.boredom,
            "steps":   self.steps,
            "cat_pos": self.cat_pos.copy(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    #  ACTION HANDLERS
    # ──────────────────────────────────────────────────────────

    _DELTAS = {
        0: np.array([-1,  0]),
        1: np.array([ 1,  0]),
        2: np.array([ 0, -1]),
        3: np.array([ 0,  1]),
    }

    def _try_move(self, delta):
        new_pos = self.cat_pos + delta
        r, c    = new_pos
        if 0 <= r < self.GRID_H and 0 <= c < self.GRID_W:
            if self.grid[r, c] != WALL:
                self.cat_pos        = new_pos
                self.last_direction = delta.copy()
                return True
        return False

    def _act_move(self, action):
        moved = self._try_move(self._DELTAS[action])
        self.in_box_streak = 0
        self.boredom = max(0.0, self.boredom - 0.02)
        return 0.15 if moved else -0.3

    def _act_zoom(self):
        """3am zoomies — only feels right when energetic."""
        if self.energy < 0.3:
            return -1.5
        self._try_move(self.last_direction)
        self._try_move(self.last_direction)
        self.energy       -= 0.08
        self.boredom       = max(0.0, self.boredom - 0.3)
        self.in_box_streak = 0
        return 2.0 + self.energy * 1.5

    def _act_nap(self):
        """
        Rewarding ONLY when tired (energy < 0.55).
        Best on a sunny spot. Scales with tiredness.
        """
        self.in_box_streak = 0

        if self.energy > 0.55:
            return -1.5          # not tired — napping feels restless

        tiredness   = 1.0 - self.energy
        self.energy = min(1.0, self.energy + 0.18)

        r, c = self.cat_pos
        if self.grid[r, c] == SUNNY_SPOT:
            self.energy = min(1.0, self.energy + 0.05)
            self.boredom = max(0.0, self.boredom - 0.2)
            return 10.0 + tiredness * 5.0   # +10 to +15

        return 3.0 + tiredness * 3.0        # +3 to +6

    def _act_eat(self):
        """
        Rewarding ONLY when hungry (hunger > 0.25).
        Scales with hunger level.
        """
        self.boredom = max(0.0, self.boredom - 0.1)

        if self.hunger < 0.25:
            return -1.0          # not hungry

        dist = self._dist_to(FOOD_BOWL)
        if dist == 0:
            hunger_factor = self.hunger
            reduction     = min(self.hunger, 0.45)
            self.hunger  -= reduction
            return 5.0 + hunger_factor * 5.0   # +5 to +10

        elif dist <= 2:
            return 0.5
        return -0.5

    def _act_hunt_toy(self):
        """
        Best when energetic. Tired or starving cats don't play.
        """
        if self.energy < 0.25:
            return -1.0

        if self.hunger > 0.75:
            return -0.5

        dist = self._dist_to(TOY)
        if dist == 0:
            self.energy  = max(0.0, self.energy - 0.06)
            self.boredom = max(0.0, self.boredom - 0.4)
            return 6.0 + self.energy * 3.0   # +6 to +9

        elif dist <= 2:
            return 1.0
        return -0.3

    def _act_knock_object(self):
        """
        Boredom behaviour — needs high energy, low hunger.
        Objects are CONSUMABLE — knocked objects disappear.
        Only 3 objects per episode so cat must diversify.
        """
        if self.energy < 0.35:
            return -1.0

        if self.hunger > 0.7:
            return -0.5

        cat_r, cat_c = self.cat_pos
        for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            r, c = cat_r + dr, cat_c + dc
            if 0 <= r < self.GRID_H and 0 <= c < self.GRID_W:
                if self.grid[r, c] == TABLE_OBJ:
                    self.grid[r, c] = EMPTY       # consumed!
                    self.boredom    = max(0.0, self.boredom - 0.6)
                    return 8.0 + self.boredom * 4.0   # +8 to +12

        return -0.5   # nothing to knock

    def _act_sit_in_box(self):
        """Universally appealing. Plateau reward after 20 steps."""
        r, c = self.cat_pos
        if self.grid[r, c] == BOX:
            self.in_box_streak += 1
            self.boredom = max(0.0, self.boredom - 0.1)
            if self.in_box_streak > 20:
                return 0.5
            bonus = min(3.0, self.in_box_streak * 0.2)
            return 3.0 + bonus

        self.in_box_streak = 0
        return -0.3

    def _act_meow(self):
        """Cry for food. Cooldown prevents spam."""
        if self.meow_cooldown > 0:
            return -1.5

        self.meow_cooldown = self.MEOW_COOLDOWN
        self.hunger        = max(0.0, self.hunger - 0.15)
        self.boredom       = max(0.0, self.boredom - 0.1)
        return 3.0 + self.hunger * 3.0

    def _act_seek_human(self):
        """Social behaviour — best when bored and energetic."""
        if self.energy < 0.3 or self.hunger > 0.6:
            return -0.5

        human_pos = self._find_cell(HUMAN)
        if human_pos is None:
            return -0.5

        direction = np.sign(human_pos - self.cat_pos)
        self._try_move(direction)
        dist = self._dist_to(HUMAN)
        self.boredom = max(0.0, self.boredom - 0.15)

        if dist == 0:
            return 3.0
        return 0.8

    def _act_ignore_human(self):
        """Rewarding when tired or hungry."""
        if self.energy < 0.35 or self.hunger > 0.5:
            return 2.5
        return -0.5

    def _act_stare_at_wall(self):
        """Pure cat behaviour. Always tiny reward."""
        self.boredom = max(0.0, self.boredom - 0.05)
        return 0.3

    # ──────────────────────────────────────────────────────────
    #  OBSERVATION
    # ──────────────────────────────────────────────────────────
    def _get_obs(self):
        diag         = float(self.GRID_H + self.GRID_W)
        cat_r, cat_c = self.cat_pos

        obs = np.array([
            cat_c / (self.GRID_W - 1),
            cat_r / (self.GRID_H - 1),
            self.hunger,
            self.energy,
            self.time_of_day,
            self._dist_to(FOOD_BOWL)  / diag,
            self._dist_to(SUNNY_SPOT) / diag,
            self._dist_to(TOY)         / diag,
            self._dist_to(HUMAN)       / diag,
            1.0 if self._dist_to(VET_CARRIER) <= 3 else 0.0,
            1.0 if self.grid[cat_r, cat_c] == BOX else 0.0,
            1.0 if self.meow_cooldown > 0 else 0.0,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────
    def _dist_to(self, cell_type):
        cat_r, cat_c = self.cat_pos
        positions    = np.argwhere(self.grid == cell_type)
        if len(positions) == 0:
            return self.GRID_H + self.GRID_W
        dists = np.abs(positions[:, 0] - cat_r) + np.abs(positions[:, 1] - cat_c)
        return int(dists.min())

    def _find_cell(self, cell_type):
        cat_r, cat_c = self.cat_pos
        positions    = np.argwhere(self.grid == cell_type)
        if len(positions) == 0:
            return None
        dists = np.abs(positions[:, 0] - cat_r) + np.abs(positions[:, 1] - cat_c)
        return positions[np.argmin(dists)]

    # ──────────────────────────────────────────────────────────
    #  RENDER
    # ──────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("StochasticCat")
            w = self.GRID_W * CELL_SIZE
            h = self.GRID_H * CELL_SIZE + 80
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w, h))
            else:
                self._screen = pygame.Surface((w, h))
            self._clock = pygame.time.Clock()

        self._screen.fill((20, 20, 30))

        # draw grid
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                cell  = self.grid[r, c]
                color = CELL_COLORS.get(cell, (245, 245, 220))
                rect  = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE,
                                    CELL_SIZE - 1, CELL_SIZE - 1)
                pygame.draw.rect(self._screen, color, rect, border_radius=4)

        # draw cat body
        cr, cc = self.cat_pos
        cx = cc * CELL_SIZE + CELL_SIZE // 2
        cy = cr * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self._screen, CAT_COLOR, (cx, cy), CELL_SIZE // 3)

        # draw cat ears
        pygame.draw.polygon(self._screen, CAT_COLOR, [
            (cx - 14, cy - 16), (cx - 6, cy - 26), (cx - 2, cy - 16)
        ])
        pygame.draw.polygon(self._screen, CAT_COLOR, [
            (cx + 2,  cy - 16), (cx + 6, cy - 26), (cx + 14, cy - 16)
        ])

        # HUD bars
        font  = pygame.font.SysFont("monospace", 13)
        hud_y = self.GRID_H * CELL_SIZE + 6

        for i, (label, value, color) in enumerate([
            ("Hunger",  self.hunger,  (200, 60,  60)),
            ("Energy",  self.energy,  (60,  180, 60)),
            ("Boredom", self.boredom, (180, 100, 200)),
        ]):
            y = hud_y + i * 18
            pygame.draw.rect(self._screen, (60, 60, 60),
                             pygame.Rect(10, y, 150, 12))
            pygame.draw.rect(self._screen, color,
                             pygame.Rect(10, y, int(value * 150), 12))
            self._screen.blit(font.render(label, True, (200, 200, 200)), (170, y))

        step_surf = font.render(
            f"Step {self.steps}/{self.MAX_STEPS}  t={self.time_of_day:.2f}",
            True, (180, 180, 180)
        )
        self._screen.blit(step_surf, (320, hud_y + 10))

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