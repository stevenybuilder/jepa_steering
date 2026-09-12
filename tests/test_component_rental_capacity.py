"""Read-only scheduling-policy tests; no provider calls or GPU launches."""
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/vast'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('rental_capacity', SCRIPTS / 'component_rental_fleet.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def offer(**changes):
    return dict(dict(id=1, gpu_name='RTX 5090', num_gpus=1, geolocation='Ohio, US',
        reliability2=.99, driver_version='595.71', cpu_cores_effective=8,
        cpu_ram=64000, dph_base=1., storage_cost=.2), **changes)


class CapacityTests(unittest.TestCase):
    def test_price_above_one_dollar_is_not_an_artificial_blocker(self):
        self.assertEqual(module.choose_offer([offer()], None, 1.5)['id'], 1)

    def test_accepts_compatible_alternative_without_precision_change(self):
        alternate = offer(gpu_name='RTX PRO 6000 WS', dph_base=1.5)
        self.assertEqual(module.choose_offer([alternate], None, 2)['id'], 1)

    def test_rejects_budget_geography_hardware_or_capacity_mismatch(self):
        for item in (offer(dph_base=1.6), offer(geolocation='Shanghai, CN'),
                     offer(driver_version='535.10'), offer(reliability2=.9),
                     offer(cpu_cores_effective=2), offer(gpu_name='Tesla V100'),
                     offer(num_gpus=3)):
            with self.assertRaises(RuntimeError):
                module.choose_offer([item], None, 1.5)

    def test_explicit_offer_does_not_silently_rent_something_else(self):
        with self.assertRaises(RuntimeError):
            module.choose_offer([offer()], 2, 2)

    def test_multigpu_host_can_fill_noncontiguous_free_slots(self):
        multi = offer(id=2, num_gpus=2, dph_base=1.8)
        self.assertEqual(module.choose_offer([offer(), multi], None, 2.1, [0, 3])['id'], 2)
        with self.assertRaises(RuntimeError):
            module.choose_offer([multi], None, 2.1, [0])
        with self.assertRaises(RuntimeError):
            module.choose_offer([multi], None, 1.8, [0, 3])

    def test_multigpu_requires_cpu_and_host_ram_per_device(self):
        for changes in (dict(cpu_ram=47000), dict(cpu_cores_effective=7)):
            with self.assertRaises(RuntimeError):
                module.choose_offer([offer(num_gpus=2, **changes)], None, 3)

    def test_account_cost_includes_stopped_storage_and_pending_compute(self):
        rows = [dict(id=1, cur_state='running', intended_status='running', dph_total=1.,
                     instance={'totalHour': 1.}),
                dict(id=2, cur_state='stopped', intended_status='stopped', dph_total=5.,
                     instance={'totalHour': .7}, storage_total_cost=.7),
                dict(id=3, cur_state='stopped', intended_status='running', dph_total=2.,
                     instance={'totalHour': .1})]
        self.assertAlmostEqual(module.fleet.account_hourly(rows), 3.7)

    def test_ownership_releases_only_unlaunched_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            def save(attempt, slots, stopped=False, launched=False, rejected=False):
                root = base / f'rentals-v{attempt}' / f'slot-{slots[0]:02d}'
                root.mkdir(parents=True)
                module.fleet.write(root / 'RESERVATION.json', dict(gpu_offset=slots[0],
                    gpus=len(slots), logical_slots=slots))
                module.fleet.write(root / 'CREATE.json', {'success': not rejected})
                if stopped:
                    module.fleet.write(root / 'STOP.json', {})
                if launched:
                    module.fleet.write(root / 'LAUNCH.json', {})
            save(1, [1], launched=True)
            save(2, [0], stopped=True)
            save(3, [3], rejected=True)
            save(4, [4, 5], stopped=True, launched=True)
            self.assertEqual(module.claimed_slots(base), {1, 4, 5})
            save(5, [0, 3])
            self.assertEqual(module.claimed_slots(base), {0, 1, 3, 4, 5})
            save(6, [5])
            with self.assertRaises(ValueError):
                module.claimed_slots(base)

    def test_reservations_include_unlisted_acquired_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / 'rentals-v5/slot-00/LEASE.json'
            module.fleet.write(path, dict(instance=11, hourly_usd=2.))
            rows = [dict(id=10, cur_state='stopped', intended_status='stopped',
                storage_total_cost=.7)]
            self.assertAlmostEqual(module.reserved_hourly(rows, base), 2.7)


if __name__ == '__main__':
    unittest.main()
