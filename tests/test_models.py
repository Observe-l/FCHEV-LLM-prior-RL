import json
from pathlib import Path
import unittest

import numpy as np

from algorithm.sac import SAC
from algorithm.td3 import TD3
from environment.fchev_env import FCHEVEnv
from util.aging import FCSDegradation
from util.battery_model import BatteryModel
from util.data import ProjectData
from util.edu_model import EDUModel
from util.fcs_model import FCSModel
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

    def test_vehicle_dynamics_and_action_limits(self):
        demand = longitudinal_demand(10.0, 0.7, 1.0, self.cfg["vehicle"])
        self.assertAlmostEqual(demand.next_speed_m_s, 10.7)
        self.assertGreater(demand.traction_force_n, 0)
        self.assertGreater(demand.motor_speed_rpm, 0)
        self.assertGreater(demand.motor_torque_nm, 0)
        at_limit = longitudinal_demand(
            self.cfg["vehicle"]["max_speed_mps"] - 0.1, 0.7, 1.0, self.cfg["vehicle"]
        )
        self.assertAlmostEqual(at_limit.next_speed_m_s, self.cfg["vehicle"]["max_speed_mps"])
        self.assertAlmostEqual(at_limit.executed_acceleration_m_s2, 0.1, places=8)
        point = EDUModel(self.data.motor).lookup(at_limit.motor_speed_rpm, at_limit.motor_torque_nm)
        self.assertTrue(point.feasible)

    def test_sac_and_td3_update(self):
        cfg = dict(self.cfg["training"])
        cfg["replay_capacity"] = 32
        sac = SAC(9, 2, cfg, seed=1)
        td3 = TD3(9, 2, cfg, seed=1)
        rng = np.random.default_rng(3)
        for agent in (sac, td3):
            for _ in range(16):
                s = rng.normal(size=9).astype(np.float32)
                a = rng.uniform(-1, 1, size=2).astype(np.float32)
                agent.replay.add(s, a, -1.0, s * 0.99, False)
            loss = agent.update(8)
            self.assertTrue(np.isfinite(loss["critic_loss"]))
            self.assertTrue(np.all(np.isfinite(agent.select_action(np.zeros(9), deterministic=True))))


class SumoIntegrationTests(unittest.TestCase):
    def test_one_step_has_physical_power_balance_and_one_sumo_tick(self):
        env = FCHEVEnv(project_root=ROOT)
        try:
            obs, reset_info = env.reset(seed=11)
            self.assertEqual(obs.shape, (9,))
            self.assertEqual(reset_info["traffic_light_count"], 25)
            self.assertAlmostEqual(float(obs[8]), 72.0, places=5)
            tls_states = {
                env.sumo.traci.trafficlight.getRedYellowGreenState(tls_id)
                for tls_id in env.sumo.traci.trafficlight.getIDList()
            }
            self.assertGreater(len(tls_states), 1)
            t0 = env.sumo.traci.simulation.getTime()
            _, _, terminated, truncated, info = env.step(np.asarray([0.2, 15.0]))
            t1 = env.sumo.traci.simulation.getTime()
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(t1 - t0, 1.0)
            self.assertAlmostEqual(info["power_balance_residual_w"], 0.0, delta=1e-6)
            self.assertGreaterEqual(info["edu_loss_power_w"], 0.0)
            self.assertGreaterEqual(info["battery_soc"], 0.15)
            self.assertLessEqual(info["battery_soc"], 0.90)
            self.assertFalse(info["sumo_acceleration_intervention"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
