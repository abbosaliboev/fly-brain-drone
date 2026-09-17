import unittest

import mujoco
import numpy as np
import pandas as pd
import torch

from src.brain.neurons import LIFParams
from src.brain.simulator import ConnectomeNetwork, LIFSimulator
from src.drone.drone import Drone
from src.drone.physics import mix_to_rotors
from src.control.motor_decoder import MotorCommand
from src.control.flight_behavior import FlySearchBehavior, fly_search_intent
from src.cli import build_parser
from src.environment.world import build_model
from src.vision.encoder import detect_food_in_frame, encode_left_right_from_bearing
from src.vision.flyvis_backend import decode_t4_t5_motion
from src.visualization.flyvis_dashboard import world_from_arena_pixel


class BearingEncoderTests(unittest.TestCase):
    def setUp(self):
        self.position = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.tonic = 15.0

    def encode(self, target):
        return encode_left_right_from_bearing(
            self.position, self.quaternion, np.asarray(target, dtype=float), self.tonic
        )

    def test_target_ahead_is_visible_bilaterally(self):
        ahead = self.encode([3.0, 0.0, 0.0])
        behind = self.encode([-3.0, 0.0, 0.0])
        self.assertLess(ahead[0], behind[0])
        self.assertLess(ahead[1], behind[1])
        self.assertAlmostEqual(ahead[0], ahead[1])

    def test_target_side_controls_correct_sensor(self):
        left = self.encode([3.0, 3.0, 0.0])
        right = self.encode([3.0, -3.0, 0.0])
        self.assertLess(left[0], left[1])
        self.assertLess(right[1], right[0])

    def test_food_detector_uses_pixels_and_reports_left_positive(self):
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[10:25, 8:20] = [255, 90, 2]
        error, confidence = detect_food_in_frame(frame)
        self.assertGreater(error, 0.0)
        self.assertGreater(confidence, 0.0)

    def test_food_detector_reports_hidden(self):
        error, confidence = detect_food_in_frame(np.zeros((40, 80, 3), dtype=np.uint8))
        self.assertEqual((error, confidence), (0.0, 0.0))


class CliTests(unittest.TestCase):
    def test_all_documented_commands_parse(self):
        for command in ("search", "demo", "dashboard", "hues", "robustness", "verify"):
            self.assertEqual(build_parser().parse_args([command]).command, command)

    def test_arena_click_mapping(self):
        self.assertEqual(world_from_arena_pixel(200, 200, 400, 400), (0.0, 0.0))
        x, y = world_from_arena_pixel(18, 18, 400, 400)
        self.assertEqual((x, y), (-3.55, 3.55))


class FlyVisMotionDecoderTests(unittest.TestCase):
    def test_opponent_populations_cancel_common_baseline(self):
        names = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
        indices = {name: [i] for i, name in enumerate(names)}
        activity = np.full(8, 5.0)
        activity[1] += 2.0  # T4b
        activity[5] += 2.0  # T5b

        motion = decode_t4_t5_motion(activity, indices)

        self.assertAlmostEqual(motion.horizontal, 2.0)
        self.assertAlmostEqual(motion.vertical, 0.0)
        self.assertAlmostEqual(motion.confidence, 2.0)

class DroneResetTests(unittest.TestCase):
    def test_food_position_is_configurable(self):
        expected = np.array([-2.0, 1.5, 0.8])
        model = build_model(food_position=expected)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        np.testing.assert_allclose(model.geom_pos[geom_id], expected)

    def test_food_colour_is_configurable(self):
        expected = np.array([0.7, 0.2, 0.05, 1.0])
        model = build_model(food_rgba=expected)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        np.testing.assert_allclose(model.geom_rgba[geom_id], expected)

    def test_light_level_is_configurable(self):
        model = build_model(light_level=0.5)
        np.testing.assert_allclose(model.vis.headlight.ambient, [0.2, 0.2, 0.2])
        np.testing.assert_allclose(model.light_diffuse[0], [0.4, 0.4, 0.4])

    def test_negative_light_level_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model(light_level=-0.1)

    def test_achromatic_world_keeps_food_chromatic(self):
        model = build_model(achromatic_background=True)
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_east")
        trim_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "room_trim_east")
        food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "food")
        self.assertAlmostEqual(float(np.ptp(model.geom_rgba[wall_id, :3])), 0.0)
        self.assertAlmostEqual(float(np.ptp(model.geom_rgba[trim_id, :3])), 0.0)
        self.assertGreater(float(np.ptp(model.geom_rgba[food_id, :3])), 0.5)

    def test_visible_vehicle_has_fly_shell(self):
        model = build_model()
        for name in ("fly_head", "fly_eye_left", "fly_eye_right", "fly_wing_left", "fly_wing_right"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(geom_id, 0, name)

    def test_forward_obstacle_ray_sees_physical_wall_not_apple(self):
        model = build_model(drone_start_pos=(0.0, 0.0, 1.0))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        drone = Drone(model, data)
        angles, distances = drone.obstacle_rays()
        center = int(np.argmin(np.abs(angles)))
        self.assertAlmostEqual(float(distances[center]), 3.8, delta=0.08)

    def test_food_is_a_composite_apple(self):
        model = build_model()
        for name in ("food", "food_lobe", "food_stem", "food_leaf"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(geom_id, 0, name)

    def test_challenge_arena_is_dashboard_optional(self):
        plain = build_model()
        challenge = build_model(challenge_arena=True)
        plain_id = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_GEOM, "island_center")
        challenge_id = mujoco.mj_name2id(challenge, mujoco.mjtObj.mjOBJ_GEOM, "island_center")
        self.assertEqual(plain_id, -1)
        self.assertGreaterEqual(challenge_id, 0)

    def test_world_can_add_coloured_distractors(self):
        model = build_model(distractors=(((1.0, -1.0, 1.0), (0.0, 1.0, 0.0, 1.0)),))
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "distractor_0")
        self.assertGreaterEqual(geom_id, 0)
        np.testing.assert_allclose(model.geom_pos[geom_id], [1.0, -1.0, 1.0])


class FlySearchBehaviorTests(unittest.TestCase):
    ANGLES = np.radians(np.array([-60.0, -30.0, 0.0, 30.0, 60.0]))

    def test_search_cast_moves_forward_instead_of_spinning_in_place(self):
        intent = fly_search_intent(100, 0.0, 0.0, 0.0, self.ANGLES, np.full(5, 3.0))
        self.assertEqual(intent.mode, "CASTING SEARCH")
        self.assertGreater(intent.forward_force, 0.0)

    def test_looming_escape_overrides_visible_apple(self):
        distances = np.array([2.0, 1.8, 0.3, 0.7, 1.2])
        intent = fly_search_intent(100, 0.8, -0.5, 0.0, self.ANGLES, distances)
        self.assertEqual(intent.mode, "LOOMING ESCAPE")
        self.assertLess(intent.forward_force, 0.0)
        self.assertLess(intent.yaw, 0.0)

    def test_escape_hysteresis_does_not_flip_around_pillar(self):
        behavior = FlySearchBehavior(minimum_escape_steps=20)
        first = behavior.update(
            0, 0.0, 0.0, 0.0, self.ANGLES,
            np.array([2.0, 1.8, 0.3, 0.7, 1.2]),
        )
        # Ray jitter now makes the left side look clearer, but the committed
        # escape saccade must retain its original direction.
        jittered = behavior.update(
            1, 0.0, 0.0, 0.0, self.ANGLES,
            np.array([0.8, 0.7, 0.32, 1.8, 2.0]),
        )
        self.assertEqual(first.mode, "LOOMING ESCAPE")
        self.assertEqual(jittered.mode, "LOOMING ESCAPE")
        self.assertEqual(np.sign(first.yaw), np.sign(jittered.yaw))

    def test_casting_flight_stays_inside_physical_arena(self):
        model = build_model(
            drone_start_pos=(0.0, 3.0, 1.0), drone_start_yaw_deg=180.0,
            challenge_arena=True,
        )
        data = mujoco.MjData(model)
        drone = Drone(model, data)
        mujoco.mj_forward(model, data)
        max_abs_xy = 0.0
        modes = set()
        for step in range(3000):
            state = drone.get_state()
            angles, distances = drone.obstacle_rays()
            intent = fly_search_intent(step, 0.0, 0.0, 0.0, angles, distances)
            modes.add(intent.mode)
            drone.apply_rotor_thrusts(mix_to_rotors(MotorCommand(0.0, 0.0, 0.0)))
            drone.apply_planar_velocity_control(
                intent.forward_force, state.quaternion, state.linear_velocity,
                drag_gain=2.1,
            )
            drone.apply_yaw_rate_control(intent.yaw, state.angular_velocity)
            drone.step()
            max_abs_xy = max(max_abs_xy, float(np.max(np.abs(drone.get_state().position[:2]))))
        self.assertIn("LOOMING ESCAPE", modes)
        self.assertLess(max_abs_xy, 3.9)

    def test_forward_camera_sees_food_straight_ahead(self):
        model = build_model(drone_start_pos=(0.0, 3.0, 1.0), drone_start_yaw_deg=0.0)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=120, width=160)
        renderer.update_scene(data, camera="drone_eye")
        error, confidence = detect_food_in_frame(renderer.render())
        renderer.close()
        self.assertGreater(confidence, 0.0)
        self.assertAlmostEqual(error, 0.0, delta=0.05)

    def test_reset_preserves_configured_yaw(self):
        model = build_model(drone_start_yaw_deg=137.0)
        data = mujoco.MjData(model)
        drone = Drone(model, data)
        data.qpos[2] = 100.0
        mujoco.mj_forward(model, data)
        self.assertTrue(drone.reset_if_unstable(max_height=3.0))
        np.testing.assert_allclose(data.qpos[3:7], model.qpos0[3:7])

    def test_positive_yaw_turns_left(self):
        model = build_model()
        data = mujoco.MjData(model)
        drone = Drone(model, data)
        mujoco.mj_forward(model, data)
        command = MotorCommand(yaw=1.0, left_turn=1.0, right_turn=0.0)
        for _ in range(200):
            drone.apply_rotor_thrusts(mix_to_rotors(command))
            drone.apply_attitude_damping(drone.get_state().angular_velocity)
            drone.step()
        # Quaternion z is positive for a left/CCW yaw from identity.
        self.assertGreater(drone.get_state().quaternion[3], 0.0)


class TemporalSimulatorTests(unittest.TestCase):
    @staticmethod
    def network():
        neurons = pd.DataFrame({"bodyId": [1, 2], "type": ["pre", "post"]})
        connections = pd.DataFrame({
            "bodyId_pre": [1], "bodyId_post": [2], "weight": [200.0]
        })
        nt = pd.DataFrame({"bodyId": [1, 2], "nt": ["acetylcholine", "acetylcholine"]})
        return ConnectomeNetwork(neurons, connections, nt, device="cpu")

    def first_postsynaptic_spike(self, delay):
        net = self.network()
        sim = LIFSimulator(
            net, LIFParams(), gain=1.0,
            delay_steps_per_connection=torch.tensor([delay]),
        )
        first = None
        for step in range(8):
            external = torch.tensor([200.0 if step == 0 else 0.0, 0.0])
            spikes = sim.step(external)
            if spikes[1] and first is None:
                first = step
        return first

    def test_connection_delay_changes_arrival_time(self):
        self.assertEqual(self.first_postsynaptic_spike(1), 1)
        self.assertEqual(self.first_postsynaptic_spike(3), 3)

    def test_synaptic_filter_retains_presynaptic_trace(self):
        net = self.network()
        sim = LIFSimulator(
            net, LIFParams(), gain=0.0,
            synaptic_tau_ms_per_neuron=torch.tensor([10.0, 10.0]),
        )
        sim.step(torch.tensor([200.0, 0.0]))
        sim.step(torch.zeros(2))
        after_spike = float(sim.synaptic_trace[0])
        sim.step(torch.zeros(2))
        self.assertGreater(after_spike, 0.0)
        self.assertGreater(float(sim.synaptic_trace[0]), 0.0)
        self.assertLess(float(sim.synaptic_trace[0]), after_spike)

    def test_graded_cell_transmits_subthreshold_voltage(self):
        net = self.network()
        plain = LIFSimulator(net, LIFParams(), gain=1.0)
        graded = LIFSimulator(
            net, LIFParams(), gain=1.0,
            graded_mask=torch.tensor([True, False]),
        )
        # Four current units cannot make the presynaptic LIF neuron spike,
        # but must produce a continuous output in the graded model.
        for _ in range(4):
            drive = torch.tensor([4.0, 0.0])
            plain.step(drive)
            graded.step(drive)
        self.assertEqual(float(plain.output_prev[0]), 0.0)
        self.assertGreater(float(graded.output_prev[0]), 0.0)
        self.assertEqual(float(plain.v[1]), plain.p.v_rest)
        self.assertGreater(float(graded.v[1]), graded.p.v_rest)


if __name__ == "__main__":
    unittest.main()
