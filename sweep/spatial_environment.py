"""
Spatial Environment and Agent for place-field navigation experiments.

Standalone module — depends only on numpy.

SpatialEnvironment: 2D arena with clustered features and a goal location.
Agent: simple navigating agent whose heading is pulled toward features
       proportional to synaptic strengths.
"""

import numpy as np


class SpatialEnvironment:
    """2D square arena with clustered spatial features and a goal."""

    def __init__(self, size=10.0, n_features=40, goal_center=None,
                 goal_radius=1.0, n_clusters=7, cluster_spread=0.8,
                 feature_sigma=0.5, seed=None):
        self.size = size
        self.n_features = n_features
        self.goal_radius = goal_radius
        self.feature_sigma = feature_sigma
        self.rng = np.random.default_rng(seed)

        # --- Place features in clusters ---
        # Pick cluster centers (margin 1.5 from edges)
        margin = 1.5
        centers = self.rng.uniform(margin, size - margin, size=(n_clusters, 2))

        # Assign 3-5 features per cluster, remainder placed uniformly
        positions = []
        for c in centers:
            n_in_cluster = self.rng.integers(3, 6)  # 3, 4, or 5
            offsets = self.rng.normal(0.0, cluster_spread, size=(n_in_cluster, 2))
            positions.append(c + offsets)

        positions = np.concatenate(positions, axis=0)

        # If we have more than n_features from clusters, trim; if fewer, fill uniformly
        if len(positions) >= n_features:
            positions = positions[:n_features]
        else:
            n_remaining = n_features - len(positions)
            extra = self.rng.uniform(0.5, size - 0.5, size=(n_remaining, 2))
            positions = np.concatenate([positions, extra], axis=0)

        # Clip to bounds
        positions = np.clip(positions, 0.5, size - 0.5)

        # --- Goal location ---
        if goal_center is None:
            goal_margin = 2.0
            goal_center = self.rng.uniform(goal_margin, size - goal_margin, size=(2,))
        self.goal_center = np.asarray(goal_center, dtype=float)

        # Remove features too close to goal, replace them elsewhere
        exclusion = goal_radius + 1.0
        dists_to_goal = np.linalg.norm(positions - self.goal_center, axis=1)
        too_close = dists_to_goal < exclusion

        if np.any(too_close):
            n_replace = int(np.sum(too_close))
            for _ in range(n_replace * 50):  # rejection sampling
                candidates = self.rng.uniform(0.5, size - 0.5, size=(n_replace, 2))
                cand_dists = np.linalg.norm(candidates - self.goal_center, axis=1)
                good = cand_dists >= exclusion
                if np.any(good):
                    # Fill in replacements one batch at a time
                    idx_close = np.where(too_close)[0]
                    good_candidates = candidates[good]
                    n_fill = min(len(idx_close), len(good_candidates))
                    positions[idx_close[:n_fill]] = good_candidates[:n_fill]
                    # Recheck
                    dists_to_goal = np.linalg.norm(positions - self.goal_center, axis=1)
                    too_close = dists_to_goal < exclusion
                    if not np.any(too_close):
                        break

        self.feature_positions = positions

    def get_activations(self, agent_position):
        """Gaussian activation of each feature given agent position.

        Returns (n_features,) array.
        """
        agent_position = np.asarray(agent_position)
        diffs = self.feature_positions - agent_position
        dist_sq = np.sum(diffs ** 2, axis=1)
        return np.exp(-dist_sq / (2.0 * self.feature_sigma ** 2))

    def check_goal(self, agent_position):
        """True if agent is within goal_radius of goal center."""
        agent_position = np.asarray(agent_position)
        return float(np.linalg.norm(agent_position - self.goal_center)) < self.goal_radius

    def get_feature_goal_distances(self):
        """Distance from each feature to the goal center. (n_features,) array."""
        return np.linalg.norm(self.feature_positions - self.goal_center, axis=1)


class Agent:
    """Navigating agent whose heading is biased toward features
    proportional to learned synaptic strengths."""

    def __init__(self, speed=0.2, sensory_horizon=2.0, heading_noise_std=0.3,
                 pull_strength=0.1, persistence=0.9, rng=None):
        self.speed = speed
        self.sensory_horizon = sensory_horizon
        self.heading_noise_std = heading_noise_std
        self.pull_strength = pull_strength
        self.persistence = persistence
        self.rng = rng or np.random.default_rng()
        self.position = None
        self.heading = None

    def reset(self, env_size, rng):
        """Place agent on a random edge, heading roughly inward."""
        self.rng = rng

        # Pick a random edge: 0=bottom, 1=top, 2=left, 3=right
        edge = self.rng.integers(4)
        t = self.rng.uniform(0.5, env_size - 0.5)

        if edge == 0:    # bottom
            self.position = np.array([t, 0.1])
            base_heading = np.pi / 2    # up
        elif edge == 1:  # top
            self.position = np.array([t, env_size - 0.1])
            base_heading = -np.pi / 2   # down
        elif edge == 2:  # left
            self.position = np.array([0.1, t])
            base_heading = 0.0          # right
        else:            # right
            self.position = np.array([env_size - 0.1, t])
            base_heading = np.pi        # left

        # Add ±45° noise
        noise = self.rng.uniform(-np.pi / 4, np.pi / 4)
        self.heading = base_heading + noise

    def step(self, dt, env, synaptic_strengths):
        """One movement step. Returns updated position (copy)."""
        old_heading = self.heading

        # Noise
        noisy_heading = old_heading + self.rng.normal(0.0, self.heading_noise_std)

        # Feature pull
        diffs = env.feature_positions - self.position
        dists = np.linalg.norm(diffs, axis=1)
        within = dists < self.sensory_horizon

        if np.any(within):
            angles_to_features = np.arctan2(diffs[within, 1], diffs[within, 0])
            pulls = synaptic_strengths[within] * self.pull_strength / (dists[within] + 0.1)
            # Angular difference wrapped to [-pi, pi]
            ang_diff = angles_to_features - noisy_heading
            ang_diff = (ang_diff + np.pi) % (2 * np.pi) - np.pi
            noisy_heading += np.sum(pulls * ang_diff)

        # Blend with persistence (angle wrapping)
        diff = noisy_heading - old_heading
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        self.heading = old_heading + (1.0 - self.persistence) * diff

        # Candidate position
        dx = self.speed * dt * np.cos(self.heading)
        dy = self.speed * dt * np.sin(self.heading)
        candidate = self.position + np.array([dx, dy])

        # Wall reflection + clamp
        if candidate[0] < 0:
            candidate[0] = -candidate[0]
            self.heading = np.pi - self.heading
        elif candidate[0] > env.size:
            candidate[0] = 2 * env.size - candidate[0]
            self.heading = np.pi - self.heading

        if candidate[1] < 0:
            candidate[1] = -candidate[1]
            self.heading = -self.heading
        elif candidate[1] > env.size:
            candidate[1] = 2 * env.size - candidate[1]
            self.heading = -self.heading

        candidate = np.clip(candidate, 0.0, env.size)
        self.position = candidate
        return self.position.copy()


# =============================================================================
# Demo / smoke test
# =============================================================================

if __name__ == "__main__":
    env = SpatialEnvironment(seed=42)
    agent = Agent()
    agent.reset(env.size, np.random.default_rng(123))

    strengths = np.zeros(env.n_features)
    encountered = set()

    for _ in range(200):
        pos = agent.step(1.0, env, strengths)
        acts = env.get_activations(pos)
        active = np.where(acts > 0.05)[0]
        encountered.update(active.tolist())

    print(f"Environment: {env.n_features} features, "
          f"goal at ({env.goal_center[0]:.2f}, {env.goal_center[1]:.2f})")
    print(f"Feature positions: x=[{env.feature_positions[:,0].min():.2f}, "
          f"{env.feature_positions[:,0].max():.2f}], "
          f"y=[{env.feature_positions[:,1].min():.2f}, "
          f"{env.feature_positions[:,1].max():.2f}]")
    print(f"Unique features encountered (activation > 0.05): "
          f"{len(encountered)} / {env.n_features}")
