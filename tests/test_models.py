import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from algorithm.sac import SAC
from algorithm.td3 import TD3
from environment.fchev_env import FCHEVEnv
from environment.scenario.build_scenario import generate_episode_randomization
from train import _plot_episode_rewards, map_action
from util.aging import FCSDegradation
from util.battery_model import BatteryModel
from util.data import ProjectData
from util.edu_model import EDUModel
from util.fcs_model import FCSModel
from util.observation import normalize_observation
from util.reward import compute_reward, following_reward, safety_distance
from util.vehicle_dynamics import longitudinal_demand


ROOT = Path(__file__).resolve().parents[1]


class DataAndPlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config" / "default.json").open(encoding="utf-8") as f:
            cls.cfg = json.load(f)
        cls.data = ProjectData(ROOT)

    def test_data_grids_load_in_documented_orientation(self):
        self.assertEqual(self.data.motor["electric_power_w"].shape, (16, 19))
        self.assertEqual(self.data.motor["operating_mask"].shape, (16, 19))
        self.assertEqual(self.data.fcs.shape, (501, 4))
        table = self.data.battery.tables[("rs", "discharge")]
        self.assertEqual(table.values.shape, (9, 8))
        self.assertAlmostEqual(float(table.soc_axis[0]), 0.105, places=6)

    def test_fast_battery_grid_query_matches_scipy_bilinear(self):
        rng = np.random.default_rng(41)
        for table in self.data.battery.tables.values():
            reference = RegularGridInterpolator(
                (table.soc_axis, table.temp_axis), table.values, method="linear"
            )
            queries = [
                (table.soc_axis[0], table.temp_axis[0]),
                (table.soc_axis[-1], table.temp_axis[-1]),
            ]
            queries.extend(
                (rng.uniform(table.soc_axis[0], table.soc_axis[-1]),
                 rng.uniform(table.temp_axis[0], table.temp_axis[-1]))
                for _ in range(20)
            )
            for soc, temp in queries:
                self.assertAlmostEqual(
                    table.query(soc, temp).value,
                    float(reference([[soc, temp]])[0]),
                    delta=1e-9,
                )

    def test_battery_zero_current_stability(self):
        b = BatteryModel(self.data.battery, self.cfg["battery"])
        before = b.state
        result = b.step(0.0, 1.0)
        self.assertAlmostEqual(b.state.soc, before.soc, places=12)
        self.assertAlmostEqual(b.state.vp1_v, 0.0, places=12)
        self.assertAlmostEqual(b.state.vp2_v, 0.0, places=12)
        self.assertAlmostEqual(b.state.core_temperature_c, 25.0, places=10)
        self.assertAlmostEqual(b.state.surface_temperature_c, 25.0, places=10)
        self.assertAlmostEqual(result.current_pack_a, 0.0, places=12)

    def test_battery_1c_discharge_and_charge_soc(self):
        discharge = BatteryModel(self.data.battery, self.cfg["battery"])
        p, _ = discharge._electrical_parameters("discharge")
        veq = p["voc"] - discharge.state.vp1_v - discharge.state.vp2_v
        i_cell = discharge.cfg["nominal_cell_capacity_ah"]
        p_pack = (veq * i_cell - p["rs"] * i_cell**2) * discharge.ns * discharge.np
        result = discharge.step(p_pack, 1.0)
        self.assertAlmostEqual(result.current_cell_a, i_cell, places=8)
        self.assertAlmostEqual(result.state.soc, 0.5 - 1.0 / 3600.0, places=9)
        self.assertAlmostEqual(
            result.terminal_voltage_pack_v * result.current_pack_a, p_pack, delta=1e-5
        )

        charge = BatteryModel(self.data.battery, self.cfg["battery"])
        p, _ = charge._electrical_parameters("charge")
        veq = p["voc"] - charge.state.vp1_v - charge.state.vp2_v
        i_cell = -charge.cfg["nominal_cell_capacity_ah"]
        p_pack = (veq * i_cell - p["rs"] * i_cell**2) * charge.ns * charge.np
        result = charge.step(p_pack, 1.0)
        self.assertAlmostEqual(result.current_cell_a, i_cell, places=8)
        self.assertAlmostEqual(result.state.soc, 0.5 + 1.0 / 3600.0, places=9)
        self.assertLess(result.terminal_voltage_cell_v, charge.cfg["maximum_cell_voltage_v"])

    def test_battery_pack_scaling_and_aging_units(self):
        b = BatteryModel(self.data.battery, self.cfg["battery"])
        c_rate, ah20, n, clipped = b._aging_parameters(1.15, 25.0)
        self.assertAlmostEqual(c_rate, 0.5, places=12)
        self.assertFalse(clipped)
        self.assertGreater(ah20, 0)
        self.assertAlmostEqual(n, 3600.0 * ah20 / b.q_nom_c, places=10)
        life_throughput_c = 2.0 * n * b.q_nom_c
        normalized_damage_at_eol = life_throughput_c / (2.0 * n * b.q_nom_c)
        self.assertAlmostEqual(normalized_damage_at_eol, 1.0, places=12)
        self.assertEqual(b.ns, 106)
        self.assertEqual(b.np, 134)

    def test_edu_reference_and_regeneration(self):
        edu = EDUModel(self.data.motor)
        point = edu.reference_point(75000.0, 156.0, 4591.0)
        self.assertTrue(point.feasible)
        self.assertLess(abs(point.mechanical_power_w - 75000.0), 100.0)
        self.assertGreater(point.loss_power_w, 0.0)
        regen = edu.lookup(3000.0, -100.0)
        self.assertTrue(regen.feasible)
        self.assertLess(regen.electrical_power_w, 0.0)
        self.assertGreaterEqual(regen.loss_power_w, 0.0)
        self.assertFalse(edu.lookup(3000.0, -400.0).feasible)

    def test_fcs_off_and_original_50kw_data(self):
        fcs = FCSModel(self.data.fcs, 50.0)
        self.assertEqual(fcs.lookup(0.0).hydrogen_rate_g_s, 0.0)
        full = fcs.lookup(50.0)
        self.assertAlmostEqual(full.hydrogen_rate_g_s, 0.8616625, places=10)
        self.assertAlmostEqual(full.efficiency_paper, 0.5077393991266882, places=10)

    def test_fcs_degradation_event_units(self):
        cfg = dict(self.cfg["fcs"])
        d = FCSDegradation(cfg)
        first = d.step(2.0, 1.0)
        self.assertAlmostEqual(first.startup_percent, 1.96e-3)
        self.assertAlmostEqual(first.low_power_percent, 1.26e-3 / 3600.0)
        self.assertGreater(first.load_change_percent, 0.0)
        second = d.step(2.0, 1.0)
        self.assertEqual(second.startup_percent, 0.0)

    def test_reward_uses_signed_following_reward_and_percent_degradation_units(self):
        components = {
            "hydrogen_g": 0.2,
            "fcs_degradation_cumulative": 2e-5,
            "fcs_degradation_increment": 2e-7,
            "battery_soh_loss": 1e-8,
            "edu_clr_cumulative": 1e-5,
            "edu_clr_increment": 1e-9,
            "comfort": 0.1,
            "following": 1.0,
            "safety_violation": False,
            "soc_penalty": 0.01,
            "collision": False,
        }
        reward = compute_reward(components, self.cfg["reward"])
        expected = {
            "following": 1.0,
            "hydrogen": -0.04,
            "fcs": -0.1,
            "battery": -0.1,
            "edu": -0.01,
            "comfort": -0.1,
            "soc": -0.1,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(components[f"reward_contribution_{name}"], value)
        self.assertAlmostEqual(components["cost_fcs"], 0.00002)
        self.assertAlmostEqual(components["cost_battery"], 1e-6)
        self.assertAlmostEqual(components["cost_edu"], 1e-7)
        self.assertAlmostEqual(reward, sum(expected.values()), places=12)
        for removed_field in (
            "normalized_hydrogen", "scaled_hydrogen", "safety_reward",
            "beta_safety_cost", "normalized_safety",
        ):
            self.assertNotIn(removed_field, components)

    def test_following_reward_and_distance_envelope(self):
        dsafe, dmax = safety_distance(10.0)
        self.assertAlmostEqual(dsafe, 2.0 + 0.5 * 10.0 + 0.0625 * 10.0**2)
        self.assertAlmostEqual(dmax, 10.0 + 10.0 + 0.0825 * 10.0**2)
        self.assertGreater(dmax, dsafe)
        c0, c1, c2, c3 = 5.0, 10.0, 3.0, 0.05
        close_gap = dsafe - 0.1
        self.assertAlmostEqual(
            following_reward(10.0, close_gap, 200.0, close_gap, c0, c1, c2, c3),
            c0 - c1 * (1.0 - close_gap / dsafe) ** 2,
        )
        self.assertEqual(following_reward(10.0, 0.0, 200.0, 0.0, c0, c1, c2, c3), c0 - c1)
        self.assertEqual(following_reward(10.0, dsafe, 200.0, dsafe, c0, c1, c2, c3), c0)
        self.assertEqual(following_reward(10.0, dmax, 200.0, dmax, c0, c1, c2, c3), c0)
        d_mid = 0.5 * (dmax + 200.0)
        self.assertAlmostEqual(
            following_reward(10.0, d_mid, 200.0, d_mid, c0, c1, c2, c3),
            c0 - c2 * (d_mid - dmax) / (200.0 - dmax),
        )
        self.assertAlmostEqual(
            following_reward(10.0, 200.0, 200.0, 205.0, c0, c1, c2, c3),
            c3 * 5.0,
        )
        self.assertAlmostEqual(
            following_reward(10.0, 200.0, 200.0, 195.0, c0, c1, c2, c3),
            -c3 * 5.0,
        )
        self.assertAlmostEqual(
            following_reward(10.0, 995.0, 200.0, 1000.0, c0, c1, c2, c3),
            c3 * 5.0,
        )
        self.assertAlmostEqual(
            following_reward(10.0, 1005.0, 200.0, 1000.0, c0, c1, c2, c3),
            -c3 * 5.0,
        )
        self.assertEqual(self.cfg["reward"]["reward_coefficients"]["beta_0"], 1.0)
        self.assertEqual(self.cfg["reward"]["following_c0"], 5.0)
        self.assertEqual(self.cfg["reward"]["following_c1"], 10.0)
        self.assertEqual(self.cfg["reward"]["following_c2"], 3.0)
        self.assertEqual(self.cfg["reward"]["following_c3"], 0.05)

    def test_make_plan_keeps_only_physical_speed_and_powertrain_limits(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            accel, candidate, physical_clip, no_reverse, speed_limit = env._make_plan(
                13.0, 0.0, 0.0
            )
            self.assertAlmostEqual(accel, 0.0)
            self.assertIsNotNone(candidate)
            self.assertFalse(physical_clip)
            self.assertFalse(no_reverse)
            self.assertFalse(speed_limit)
            accel, _, _, no_reverse, _ = env._make_plan(0.0, -0.5, 0.0)
            self.assertEqual(accel, 0.0)
            self.assertTrue(no_reverse)
        finally:
            env.close()

    def test_vehicle_dynamics_and_action_limits(self):
        demand = longitudinal_demand(10.0, 0.7, 1.0, self.cfg["vehicle"])
        self.assertAlmostEqual(demand.next_speed_m_s, 10.7)
        self.assertGreater(demand.traction_force_n, 0)
        self.assertGreater(demand.motor_speed_rpm, 0)
        self.assertGreater(demand.motor_torque_nm, 0)
        at_limit = longitudinal_demand(
            self.cfg["vehicle"]["max_speed_mps"] - 0.1, 0.7, 1.0, self.cfg["vehicle"]
        )
        self.assertAlmostEqual(
            at_limit.next_speed_m_s,
            self.cfg["vehicle"]["max_speed_mps"] + 0.6,
        )
        self.assertAlmostEqual(at_limit.executed_acceleration_m_s2, 0.7, places=8)
        point = EDUModel(self.data.motor).lookup(at_limit.motor_speed_rpm, at_limit.motor_torque_nm)
        self.assertTrue(point.feasible)

    def test_continuous_action_mapping_uses_physical_bounds(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            low_action = map_action(np.asarray([-1.0, -1.0]), env.action_space)
            high_action = map_action(np.asarray([1.0, 1.0]), env.action_space)
            np.testing.assert_allclose(low_action, [-1.5, 0.0], atol=1e-6)
            np.testing.assert_allclose(high_action, [0.7, 50.0], atol=1e-6)
            zero_action = map_action(np.asarray([0.0, 0.0]), env.action_space)
            np.testing.assert_allclose(zero_action, [0.0, 25.0], atol=1e-6)
            mixed_action = map_action(np.asarray([-0.5, 0.5]), env.action_space)
            np.testing.assert_allclose(mixed_action, [-0.75, 37.5], atol=1e-6)
            self.assertEqual(float(env.observation_space.low[8]), 0.0)
            self.assertEqual(float(env.observation_space.high[8]), 30000.0)
            observation = np.zeros(15, dtype=np.float32)
            observation[4] = -1.5
            observation[5] = 13.0
            observation[6] = 0.7
            observation[7] = 19.166666
            observation[8] = 30000.0
            observation[9] = 2.0
            observation[10] = 30000.0
            observation[11] = 100.0
            observation[12] = 19.166666
            observation[13] = 19.166666
            observation[14] = 30000.0
            normalized = normalize_observation(observation, env.vehicle_cfg)
            np.testing.assert_allclose(
                normalized, [0, 0, 0, 0, -1.5, 13.0, 0.7, 19.166666,
                             30000.0, 1.0, 30000.0, 100.0,
                             19.166666, 19.166666, 30000.0],
                rtol=0.0, atol=1e-6,
            )
        finally:
            env.close()

    def test_reward_plot_uses_zero_vertical_baseline(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes_csv = root / "episodes.csv"
            episodes_csv.write_text(
                "episode,return\n1,-75\n2,25\n", encoding="utf-8"
            )
            output_png = root / "reward.png"
            with patch("matplotlib.axes.Axes.set_ylim") as set_ylim:
                _plot_episode_rewards(episodes_csv, output_png)
            self.assertTrue(output_png.exists())
            self.assertTrue(any(
                call.kwargs.get("bottom") == 0.0
                for call in set_ylim.call_args_list
            ))
        self.assertEqual(self.cfg["reward"]["abnormal_termination_penalty"], -500.0)

    def test_sac_and_td3_update(self):
        cfg = dict(self.cfg["training"])
        cfg["replay_capacity"] = 32
        sac = SAC(15, 2, cfg, seed=1)
        td3 = TD3(15, 2, cfg, seed=1)
        rng = np.random.default_rng(3)
        for agent in (sac, td3):
            for _ in range(16):
                s = rng.normal(size=15).astype(np.float32)
                a = rng.uniform(-1, 1, size=2).astype(np.float32)
                agent.replay.add(s, a, -1.0, s * 0.99, False)
            loss = agent.update(8)
            self.assertTrue(np.isfinite(loss["critic_loss"]))
            self.assertTrue(np.all(np.isfinite(agent.select_action(np.zeros(15), deterministic=True))))


class SumoIntegrationTests(unittest.TestCase):
    def test_randomized_scenario_is_seed_reproducible_and_signal_cycle_is_fixed(self):
        with (ROOT / "config" / "default.json").open(encoding="utf-8") as f:
            sumo_cfg = json.load(f)["sumo"]
        first = generate_episode_randomization(sumo_cfg, 731)
        repeated = generate_episode_randomization(sumo_cfg, 731)
        changed = generate_episode_randomization(sumo_cfg, 732)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        offsets = [item["cycle_position_s"] for item in first["signal_initial_conditions"]]
        self.assertEqual(len(offsets), len(set(offsets)))
        remaining = [item["remaining_s"] for item in first["signal_initial_conditions"]]
        self.assertEqual(len(remaining), len(set(remaining)))
        self.assertEqual(len(first["lead_stop_plan"]), sumo_cfg["signal_count"] - 1)
        self.assertTrue(all(50.0 <= stop["duration_s"] <= 100.0 for stop in first["lead_stop_plan"]))
        self.assertTrue(all(100.0 <= stop["position_m"] <= 900.0 for stop in first["lead_stop_plan"]))

        signals = ET.parse(
            ROOT / "environment" / "scenario" / "generated" / "signals.tll.xml"
        ).getroot()
        phase_durations = {
            tuple(int(float(phase.attrib["duration"])) for phase in logic.findall("phase"))
            for logic in signals.findall("tlLogic")
        }
        self.assertEqual(phase_durations, {(40, 3, 47)})

    def test_episode_ends_when_lead_reaches_terminal_stop(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            obs, _ = env.reset(seed=7)
            # Isolate the lead route completion by holding the ego stationary.
            for tls_id in env.sumo.traci.trafficlight.getIDList():
                env.sumo.traci.trafficlight.setPhase(tls_id, 0)
                env.sumo.traci.trafficlight.setPhaseDuration(tls_id, 5000.0)
            for step in range(1, int(env.sumo_cfg["episode_limit_s"]) + 1):
                obs, reward, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminated_reason"], "lead_terminal_stop")
            self.assertTrue(info["leader_terminal_stop_reached"])
            self.assertLess(info["lead_speed_m_s"], 0.1)
            self.assertEqual(info["terminal_penalty"], 0.0)
        finally:
            env.close()

    def test_rest_start_and_step_power_balance(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            obs, reset_info = env.reset(seed=11)
            self.assertEqual(obs.shape, (15,))
            self.assertEqual(float(env.observation_space.low[8]), 0.0)
            self.assertEqual(float(env.observation_space.high[8]), 30000.0)
            self.assertEqual(reset_info["traffic_light_count"], 25)
            self.assertEqual(len(reset_info["lead_stop_plan"]), 24)
            self.assertEqual(len(env.sumo.traci.vehicle.getStops("lead")), 25)
            self.assertEqual(reset_info["scenario_seed"], 11)
            self.assertAlmostEqual(float(obs[8]), 10.0, places=5)
            self.assertAlmostEqual(float(obs[5]), 0.0, places=6)
            self.assertAlmostEqual(float(obs[7]), 0.0, places=6)
            self.assertIn(float(obs[9]), (0.0, 1.0, 2.0))
            self.assertGreater(float(obs[10]), 900.0)
            self.assertGreaterEqual(float(obs[11]), 0.0)
            self.assertLessEqual(float(obs[11]), 100.0)
            self.assertAlmostEqual(
                float(obs[12]), env.vehicle_cfg["max_speed_mps"], places=5
            )
            if float(obs[9]) == 0.0:
                self.assertEqual(float(obs[13]), 0.0)
                self.assertAlmostEqual(float(obs[14]), float(obs[10]))
            else:
                self.assertAlmostEqual(
                    float(obs[13]),
                    env.sumo_cfg["intersection_speed_limit_kph"] / 3.6,
                    places=5,
                )
                self.assertAlmostEqual(
                    float(obs[14]),
                    max(0.0, float(obs[10]) - env.sumo_cfg["intersection_speed_zone_m"]),
                    places=3,
                )
            self.assertAlmostEqual(reset_info["initial_gap_m"], 10.0, delta=0.02)
            self.assertAlmostEqual(reset_info["gap_diagnostic_threshold_m"], 300.0)
            self.assertAlmostEqual(reset_info["gap_monitor_threshold_m"], 1000.0)
            route_root = env.project_root / "environment" / "scenario" / "generated" / "vehicles.rou.xml"
            route_text = route_root.read_text(encoding="utf-8")
            self.assertIn('id="lead"', route_text)
            self.assertEqual(route_text.count('departSpeed="0.000"'), 2)
            self.assertIn('id="ego" type="egoType" route="egoRoute"', route_text)
            self.assertRegex(route_text, r'id="egoRoute" edges="[^"]* e26"')
            self.assertIn('arrivalSpeed="0.0"', route_text)
            self.assertIn('lane="e26_0"', route_text)
            self.assertIn('endPos="200.000"', route_text)
            self.assertIn(
                f'duration="{float(env.sumo_cfg["episode_limit_s"] + 1):.3f}"',
                route_text,
            )
            self.assertIn('triggered="false"', route_text)
            tls_states = {
                env.sumo.traci.trafficlight.getRedYellowGreenState(tls_id)
                for tls_id in env.sumo.traci.trafficlight.getIDList()
            }
            self.assertGreater(len(tls_states), 1)
            # Keep the reset assertion at rest above, accelerate both vehicles
            # together, then exercise braking/regen at a representative speed.
            for _ in range(18):
                obs, _, terminated, truncated, _ = env.step(np.asarray([0.7, 0.0]))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            self.assertGreater(float(obs[5]), 10.0)
            t0 = env.sumo.traci.simulation.getTime()
            _, _, terminated, truncated, info = env.step(np.asarray([-1.332, 39.0]))
            t1 = env.sumo.traci.simulation.getTime()
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(t1 - t0, 1.0)
            self.assertAlmostEqual(info["power_balance_residual_w"], 0.0, delta=1e-6)
            self.assertAlmostEqual(info["requested_acceleration_m_s2"], -1.332, places=6)
            self.assertAlmostEqual(info["powertrain_limited_acceleration_m_s2"], -1.332, places=6)
            self.assertAlmostEqual(info["executed_acceleration_m_s2"], -1.332, places=6)
            self.assertFalse(info["driver_supervisor_override"])
            self.assertAlmostEqual(info["requested_fcs_power_raw_kw"], 39.0, places=6)
            self.assertGreaterEqual(info["fcs_power_executed_kw"], 0.0)
            self.assertLessEqual(info["fcs_power_executed_kw"], 50.0)
            self.assertAlmostEqual(info["fcs_power_executed_kw"], 39.0, places=7)
            self.assertGreaterEqual(info["battery_power_w"], env.battery.last_limits["minimum_power_w"] - 1e-5)
            self.assertLessEqual(info["battery_power_w"], env.battery.last_limits["maximum_power_w"] + 1e-5)
            self.assertGreater(info["dc_bus_dump_power_w"], 0.0)
            self.assertTrue(info["dc_bus_dump_load_active"])
            env.sumo_cfg["gap_diagnostic_threshold_m"] = 20.0
            original_states = env.sumo.states

            def report_gap_above_diagnostic_threshold():
                states = original_states()
                states["gap_m"] = 21.0
                return states

            env.sumo.states = report_gap_above_diagnostic_threshold
            try:
                _, _, terminated, truncated, gap_info = env.step(np.asarray([0.0, 15.0]))
            finally:
                env.sumo.states = original_states
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIsNone(gap_info["terminated_reason"])
            self.assertTrue(gap_info["gap_over_diagnostic_threshold"])
            self.assertGreaterEqual(info["edu_loss_power_w"], 0.0)
            self.assertGreaterEqual(info["battery_soc"], 0.15)
            self.assertLessEqual(info["battery_soc"], 0.90)
            self.assertFalse(info["sumo_acceleration_intervention"])
            self.assertFalse(info["physical_powertrain_clip"])
            self.assertFalse(info["fcs_power_clipped_for_battery_limits"])
        finally:
            env.close()

    def test_negative_acceleration_from_rest_is_clipped_with_minus_ten_penalty(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            obs, _ = env.reset(seed=23)
            self.assertAlmostEqual(float(obs[5]), 0.0, places=6)
            self.assertAlmostEqual(float(obs[7]), 0.0, places=6)
            _, neutral_reward, terminated, truncated, neutral_info = env.step(
                np.asarray([0.0, 0.0])
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                neutral_info["reward_components"]["zero_speed_negative_accel_penalty"],
                0.0,
            )
            env.reset(seed=23)
            _, reward, terminated, truncated, info = env.step(
                np.asarray([-0.5, 0.0])
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(info["requested_acceleration_m_s2"], -0.5)
            self.assertTrue(info["no_reverse_clip"])
            self.assertAlmostEqual(info["executed_acceleration_m_s2"], 0.0, places=6)
            self.assertEqual(
                info["reward_components"]["zero_speed_negative_accel_penalty"],
                -10.0,
            )
            self.assertEqual(
                info["reward_components"][
                    "reward_contribution_zero_speed_negative_accel_penalty"
                ],
                -10.0,
            )
            self.assertAlmostEqual(reward, neutral_reward - 10.0, places=6)
            self.assertAlmostEqual(
                reward,
                sum(v for k, v in info["reward_components"].items()
                    if k.startswith("reward_contribution_")),
            )

            _, _, terminated, truncated, moving_info = env.step(
                np.asarray([0.5, 0.0])
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertGreater(moving_info["ego_speed_m_s"], 0.0)
            _, _, terminated, truncated, moving_info = env.step(
                np.asarray([-0.5, 0.0])
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                moving_info["reward_components"]["zero_speed_negative_accel_penalty"],
                0.0,
            )
        finally:
            env.close()

    def test_positive_acceleration_beyond_global_speed_envelope_terminates_without_clipping(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            env.reset(seed=29)
            # Put the ego at the configured speed limit without advancing
            # SUMO, then verify positive acceleration terminates immediately.
            env.vehicle_cfg["max_speed_mps"] = 0.2
            original_states = env.sumo.states

            def report_at_max_speed(force_fresh=False):
                states = original_states(force_fresh=force_fresh)
                states["ego"] = dict(states["ego"])
                states["ego"]["speed_m_s"] = 0.2
                return states

            env.sumo.states = report_at_max_speed
            time_before = env.sumo.traci.simulation.getTime()
            step_calls_before = env.sumo.step_calls
            _, reward, terminated, truncated, info = env.step(np.asarray([0.7, 0.0]))
            env.sumo.states = original_states
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["max_speed_action_violation"])
            self.assertEqual(info["terminated_reason"], "max_speed_action")
            self.assertFalse(info["speed_limit_clip"])
            self.assertAlmostEqual(info["requested_acceleration_m_s2"], 0.7)
            self.assertIsNone(info["executed_acceleration_m_s2"])
            self.assertEqual(info["terminal_penalty"], -500.0)
            self.assertEqual(
                info["reward_components"]["reward_contribution_terminal_penalty"],
                -500.0,
            )
            self.assertEqual(reward, -500.0)
            self.assertEqual(env.sumo.step_calls, step_calls_before)
            self.assertEqual(env.sumo.traci.simulation.getTime(), time_before)
        finally:
            env.close()

    def test_gap_diagnostic_does_not_terminate_and_overlap_does(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            env.reset(seed=12)
            original_states = env.sumo.states
            states_calls = 0

            def report_far_gap_once():
                nonlocal states_calls
                states = original_states()
                states_calls += 1
                if states_calls == 2:
                    states["gap_m"] = 1001.0
                return states

            env.sumo.states = report_far_gap_once
            try:
                _, reward, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
            finally:
                env.sumo.states = original_states
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertNotEqual(reward, -35.0)
            self.assertIsNone(info["terminated_reason"])
            self.assertEqual(info["terminal_penalty"], 0.0)
            self.assertTrue(info["gap_over_monitor_threshold"])

            env.reset(seed=13)
            original_collision_ids = env.sumo.collision_ids
            env.sumo.collision_ids = lambda: ["ego", "lead"]
            try:
                _, reward, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
            finally:
                env.sumo.collision_ids = original_collision_ids
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminated_reason"], "collision")
            self.assertTrue(info["collision_event"])
            self.assertEqual(info["terminal_penalty"], -500.0)
            self.assertEqual(info["collision_penalty"], 0.0)
            self.assertEqual(
                info["reward_components"]["reward_contribution_terminal_penalty"],
                -500.0,
            )
            self.assertAlmostEqual(
                reward,
                sum(v for k, v in info["reward_components"].items()
                    if k.startswith("reward_contribution_")),
            )

            env.reset(seed=14)
            original_states = env.sumo.states
            states_calls = 0

            def report_overlap_once():
                nonlocal states_calls
                states = original_states()
                states_calls += 1
                if states_calls == 2:
                    states["gap_m"] = -1.0
                return states

            env.sumo.states = report_overlap_once
            try:
                _, reward, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
            finally:
                env.sumo.states = original_states
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminated_reason"], "collision")
            self.assertTrue(info["gap_overlap"])
            self.assertTrue(info["collision_event"])
            self.assertEqual(info["terminal_penalty"], -500.0)
            self.assertEqual(info["collision_penalty"], 0.0)
            self.assertEqual(
                info["reward_components"]["reward_contribution_terminal_penalty"],
                -500.0,
            )
            self.assertAlmostEqual(
                reward,
                sum(v for k, v in info["reward_components"].items()
                    if k.startswith("reward_contribution_")),
            )

        finally:
            env.close()

    def test_no_movement_does_not_terminate_after_600_simulation_seconds(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            env.reset(seed=15)
            env.sumo.set_ego_speed(0.0)
            env.sumo.simulation_step()
            states = env.sumo.states()
            env._last_x = states["ego"]["x_m"]
            env._stuck_steps = 599
            _, _, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIsNone(info["terminated_reason"])
            self.assertEqual(info["stuck_steps"], 600)
            self.assertAlmostEqual(info["stuck_duration_s"], 600.0)
            self.assertEqual(info["terminal_penalty"], 0.0)
        finally:
            env.close()

    def test_red_light_crossing_detection(self):
        red = {"id": "tls0", "state": "r", "distance_m": 4.0}
        same_light = {"id": "tls0", "state": "r", "distance_m": -8.0}
        before = {"x_m": 990.0, "lane_id": "e0_0"}
        crossed = {"x_m": 1002.0, "lane_id": ":n1_0"}
        not_crossed = {"x_m": 993.0, "lane_id": "e0_0"}
        self.assertTrue(FCHEVEnv._crossed_red_light(red, same_light, before, crossed))
        self.assertFalse(FCHEVEnv._crossed_red_light(red, same_light, before, not_crossed))
        self.assertFalse(FCHEVEnv._crossed_red_light(
            {"id": "tls0", "state": "G", "distance_m": 4.0},
            same_light, before, crossed,
        ))
        self.assertTrue(FCHEVEnv._crossed_traffic_light(
            {"id": "tls0", "state": "G", "distance_m": 4.0},
            same_light, before, crossed,
        ))

    def test_speed_region_observation_uses_entry_distance_and_red_stop_line(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            tls = {"id": "n1", "link_index": 0, "distance_m": 150.0,
                   "state": "G", "remaining_s": 20.0}
            current, upcoming, distance = env._speed_region_observation(
                {"next_tls": tls, "upcoming_tls": [tls]},
                {"lane_id": "e0_0"},
                tls,
            )
            self.assertAlmostEqual(current, env.vehicle_cfg["max_speed_mps"])
            self.assertAlmostEqual(
                upcoming, env.sumo_cfg["intersection_speed_limit_kph"] / 3.6
            )
            self.assertAlmostEqual(distance, 50.0)

            # After entering the current zone, report the next region entry.
            tls = {"id": "n1", "link_index": 0, "distance_m": 50.0,
                   "state": "G", "remaining_s": 20.0}
            future_tls = {"id": "n2", "link_index": 0, "distance_m": 1050.0,
                          "state": "G", "remaining_s": 10.0}
            current, upcoming, distance = env._speed_region_observation(
                {"next_tls": tls, "upcoming_tls": [tls, future_tls]},
                {"lane_id": "e1_0"},
                tls,
            )
            self.assertAlmostEqual(
                current, env.sumo_cfg["intersection_speed_limit_kph"] / 3.6
            )
            self.assertAlmostEqual(
                upcoming, env.sumo_cfg["intersection_speed_limit_kph"] / 3.6
            )
            self.assertAlmostEqual(distance, 950.0)

            # Red encodes a zero-speed target at its stop line.
            tls["state"] = "r"
            current, upcoming, distance = env._speed_region_observation(
                {"next_tls": tls, "upcoming_tls": [tls, future_tls]},
                {"lane_id": "e1_0"},
                tls,
            )
            self.assertEqual(upcoming, 0.0)
            self.assertAlmostEqual(distance, 50.0)
        finally:
            env.close()

    def test_intersection_speeding_terminates_with_minus_500_penalty(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            env.reset(seed=143)
            original_states = env.sumo.states

            def report_speeding_state(force_fresh=False):
                states = original_states(force_fresh=force_fresh)
                states["ego"]["speed_m_s"] = 10.0
                states["next_tls"] = {
                    "id": "n1", "link_index": 0, "distance_m": 80.0,
                    "state": "G", "remaining_s": 20.0,
                }
                return states

            env.sumo.states = report_speeding_state

            _, reward, terminated, truncated, info = env.step(np.asarray([0.0, 0.0]))
            env.sumo.states = original_states
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminated_reason"], "speeding")
            self.assertTrue(info["speeding_violation"])
            self.assertIn("ego", info["speeding_vehicles"])
            self.assertEqual(info["terminal_penalty"], -500.0)
            self.assertEqual(info["collision_penalty"], 0.0)
            self.assertEqual(info["speed_limit_positive_accel_penalty"], 0.0)
        finally:
            env.close()

    def test_red_light_crossing_terminates_with_minus_500_penalty(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            env.reset(seed=56)
            traci = env.sumo.traci
            lane_id = traci.vehicle.getLaneID("ego")
            lane_length = traci.lane.getLength(lane_id)
            traci.vehicle.moveTo("ego", lane_id, lane_length - 25.0)
            traci.vehicle.setSpeedMode("ego", 0)
            traci.vehicle.setSpeed("ego", 13.0)
            traci.simulationStep()
            tls = env.sumo.next_tls()
            self.assertIsNotNone(tls)
            traci.trafficlight.setPhase(tls["id"], 2)
            traci.trafficlight.setPhaseDuration(tls["id"], 30.0)

            _, reward, terminated, truncated, info = env.step([0.7, 0.0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["terminated_reason"], "red_light_violation")
            self.assertTrue(info["red_light_violation"])
            self.assertEqual(info["terminal_penalty"], -500.0)
            self.assertEqual(info["collision_penalty"], 0.0)
            self.assertEqual(
                info["reward_components"]["reward_contribution_terminal_penalty"],
                -500.0,
            )
            self.assertAlmostEqual(
                reward,
                sum(v for k, v in info["reward_components"].items()
                    if k.startswith("reward_contribution_")),
            )
            self.assertAlmostEqual(info["requested_acceleration_m_s2"], 0.7, places=6)
            self.assertAlmostEqual(info["executed_acceleration_m_s2"], 0.7, places=6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
