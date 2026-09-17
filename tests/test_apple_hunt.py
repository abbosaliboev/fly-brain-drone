import unittest
import numpy as np
from src.control.apple_hunt import AppleHunt, safe_apple_position
from src.environment.world import build_model


class RoundTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.game = AppleHunt(clock=lambda: self.now)

    def test_pause_resume_and_new_round(self):
        self.game.start()
        self.now = 4
        self.game.pause()
        self.now = 20
        self.assertEqual(self.game.elapsed, 4)
        self.game.start()
        self.now = 23
        self.assertEqual(self.game.elapsed, 7)
        self.game.new_round()
        self.assertEqual(self.game.elapsed, 0)
        self.assertEqual(self.game.round, 2)

    def test_success_requires_three_fresh_visible_observations(self):
        self.game.start()
        self.assertIsNone(self.game.observe(.3, .8, True))
        self.assertIsNone(self.game.observe(.3, .8, False))
        self.assertIsNone(self.game.observe(.3, .8, True))
        self.assertIsNone(self.game.observe(.3, .8, True))
        self.now = 2
        self.assertIsNotNone(self.game.observe(.3, .8, True))
        self.now = 12
        self.assertEqual(self.game.elapsed, 2)
        self.assertIsNone(self.game.observe(.3, .8, True))
        self.assertEqual(len(self.game.history), 1)

    def test_score_contacts_and_separate_records(self):
        self.game.start()
        for _ in range(3):
            self.game.physics(2/3, 1, True, True)
        self.assertEqual((self.game.escapes, self.game.collisions), (1, 1))
        self.now = 2
        for _ in range(3):
            self.game.observe(.3, .8, True)
        self.assertEqual(self.game.score, 3390)
        self.assertIn(("HYBRID", "NORMAL"), self.game.best)
        self.game.new_round("ENGINEERED", "HARD")
        self.assertNotIn(("ENGINEERED", "HARD"), self.game.best)

    def test_history_is_bounded(self):
        for _ in range(12):
            self.game.new_round()
            self.game.start()
            for _ in range(3):
                self.game.observe(.3, .8, True)
        self.assertEqual(len(self.game.history), 10)

    def test_wall_and_pillar_placement_rejected(self):
        model = build_model(challenge_arena=True)
        self.assertFalse(safe_apple_position(model, (3.7, 0, 1)))
        self.assertFalse(safe_apple_position(model, (-2.25, -.7, 1)))
        self.assertTrue(safe_apple_position(model, (3, 3, 1)))
        self.assertFalse(safe_apple_position(model, (np.nan, 0, 1)))

    def test_game_time_is_physics_time_not_wall_time(self):
        self.game.start()
        self.now = 10
        for _ in range(15):
            self.game.physics(.005, 0, False, False)
        self.assertAlmostEqual(self.game.sim_time, .075)
        self.assertEqual(self.game.elapsed, 10)


class ArrivalRenderTests(unittest.TestCase):
    def test_apple_visible_on_approach_and_hidden_when_facing_away(self):
        import mujoco
        from src.visualization.arrival import visible_apple
        model = build_model(food_position=(1, 0, 1))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=120, width=160)
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("food", "food_lobe", "food_stem", "food_leaf")]
        try:
            game = AppleHunt()
            game.start()
            for distance in (.79, .77, .75):
                data.qpos[:3] = (1-distance, 0, 1)
                mujoco.mj_forward(model, data)
                visible, confidence = visible_apple(renderer, data, ids)
                self.assertTrue(visible)
                result = game.observe(distance, confidence, visible)
            self.assertIsNotNone(result)
            data.qpos[3:7] = (0, 0, 0, 1)
            mujoco.mj_forward(model, data)
            self.assertFalse(visible_apple(renderer, data, ids)[0])
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
