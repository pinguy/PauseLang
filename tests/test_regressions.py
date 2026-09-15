import contextlib
import io
import math
import unittest
from unittest.mock import patch

from PauseLang_v0_7_13 import (INSTRUCTIONS, SPEC, IoTDemos, PauseLangCompiler,
                              PauseLangVM, TimeQuantizer)


def run(source, **kwargs):
    pauses, data, _, labels = PauseLangCompiler.compile(source)
    vm = PauseLangVM(**kwargs)
    return vm, vm.execute(data, pauses, labels=labels)


class TimingTests(unittest.TestCase):
    def test_defaults_resolved_at_construction(self):
        with patch.dict(SPEC, guard_band=.004, time_quantum=.010):
            self.assertEqual(PauseLangVM().quantizer.guard_band, .004)
            self.assertEqual(TimeQuantizer().quantum, .010)
            self.assertEqual(PauseLangVM(guard_band=.001).quantizer.guard_band, .001)

    def test_overlap_rejected_instead_of_table_order(self):
        q = TimeQuantizer(guard_band=.004)
        for pause in [.046, .047, .0475, .048, .049]:
            with self.subTest(pause=pause):
                self.assertIsNone(q.decode(pause))
        self.assertEqual(q.decode(.045).opcode, 'PUSH')
        result = PauseLangVM(guard_band=.004).execute([42], [.048], sync=False)
        self.assertEqual(result['traps'], ['INVALID_INSTRUCTION'])
        self.assertEqual(result['final_state']['stack'], [])

    def test_all_opcode_boundaries(self):
        for guard in [.0015, .002]:
            q = TimeQuantizer(guard_band=guard)
            for instr in INSTRUCTIONS.values():
                for delta in [-guard, 0, guard]:
                    with self.subTest(opcode=instr.opcode, delta=delta, guard=guard):
                        self.assertIs(q.decode(instr.pause + delta), instr)
                self.assertIsNone(q.decode(instr.pause + guard + .000002))
                self.assertIsNone(q.decode(instr.pause - guard - .000002))

    def test_bad_pauses_trap_without_python_crash(self):
        for pause in [math.nan, math.inf, -math.inf, 0, -.045]:
            with self.subTest(pause=pause):
                result = PauseLangVM().execute([42], [pause], sync=False)
                self.assertEqual(result['traps'], ['INVALID_INSTRUCTION'])

    def test_constant_offset_sync_acquisition(self):
        pauses, data, _, _ = PauseLangCompiler.compile('CONST 42\nHALT')
        vm = PauseLangVM()
        result = vm.execute(data, [p + .010 for p in pauses])
        self.assertEqual(result['traps'], ['HALT'])
        self.assertEqual(result['final_state']['stack'], [42])
        self.assertAlmostEqual(vm.quantizer.drift_estimate, .010)

    def test_affine_offset_and_skew(self):
        for scale in [.95, 1.0, 1.05]:
            for offset in [-.001, 0, .010]:
                q = TimeQuantizer(calibration='affine')
                self.assertTrue(q.calibrate([scale*p + offset for p in SPEC['sync_phrase']]))
                for instr in INSTRUCTIONS.values():
                    self.assertIs(q.decode(scale*instr.pause + offset), instr)
                pauses, data, _, _ = PauseLangCompiler.compile('CONST 42\nHALT')
                result = PauseLangVM(calibration='affine').execute(data, [scale*p + offset for p in pauses])
                self.assertEqual(result['traps'], ['HALT'])
                self.assertEqual(result['final_state']['stack'], [42])

    def test_invalid_calibration_is_atomic(self):
        for mode in ['offset', 'affine']:
            q = TimeQuantizer(calibration=mode)
            q.calibrate([p+.001 for p in SPEC['sync_phrase']])
            before = (q.drift_estimate, q.scale_estimate, list(q.calibration_history))
            for phrase in [[], [.29], [.30, .29], [.045, .05], [math.nan, .3], [.5, .6]]:
                self.assertFalse(q.calibrate(phrase))
                self.assertEqual((q.drift_estimate, q.scale_estimate, list(q.calibration_history)), before)

    def test_reset_clears_calibration_but_preserves_configuration(self):
        vm = PauseLangVM(guard_band=.002, calibration='affine')
        vm.quantizer.calibrate([p*1.01+.001 for p in SPEC['sync_phrase']])
        vm.reset()
        self.assertEqual(vm.quantizer.drift_estimate, 0)
        self.assertEqual(vm.quantizer.scale_estimate, 1)
        self.assertEqual(list(vm.quantizer.calibration_history), [])
        self.assertEqual(vm.quantizer.guard_band, .002)
        self.assertEqual(vm.quantizer.calibration, 'affine')

    def test_invalid_configuration(self):
        for args in [dict(guard_band=-1), dict(guard_band=math.nan),
                     dict(quantum=0), dict(quantum=math.inf), dict(calibration='guess')]:
            with self.assertRaises(ValueError):
                TimeQuantizer(**args)

    def test_length_validation_precedes_calibration(self):
        vm = PauseLangVM()
        result = vm.execute([0], [.291, .301])
        self.assertIn('error', result)
        self.assertEqual(vm.quantizer.drift_estimate, 0)


class CompilerTests(unittest.TestCase):
    def test_extra_tokens_rejected_with_source_line(self):
        for source in ['\nCONST 70 STOREI INCIX', '\nADD2 0 HALT', '\nINC 1 2']:
            with self.assertRaisesRegex(ValueError, 'line 2'):
                PauseLangCompiler.compile(source)

    def test_malformed_labels(self):
        for source in [':', 'bad label:', '123:', 'same:\nsame:']:
            with self.assertRaises(ValueError):
                PauseLangCompiler.compile(source)

    def test_operands_and_comments(self):
        vm, result = run('CONST +42 # a comment with tokens\nHALT')
        self.assertEqual(result['final_state']['stack'], [42])
        for operand in ['--2', '²', 'missing']:
            with self.assertRaisesRegex(ValueError, 'line 1'):
                PauseLangCompiler.compile('CONST ' + operand)

    def test_storei_pop_consumes_exactly_one_value(self):
        _, result = run('CONST 99\nCONST 42\nSTOREI_POP\nHALT')
        self.assertEqual(result['final_state']['stack'], [99])
        self.assertEqual(result['final_state']['memory'], {0:42})
        self.assertEqual(result['traps'], ['HALT'])
        _, result = run('CONST 42\nSTOREI_POP\nHALT')
        self.assertEqual(result['traps'], ['HALT'])
        self.assertEqual(result['final_state']['stack'], [])

    def test_labels_after_changed_macro_expansion(self):
        _, result = run('CONST 42\nSTOREI_POP\nJUMP end\nCONST 999\nend:\nCONST 7\nHALT')
        self.assertEqual(result['traps'], ['HALT'])
        self.assertEqual(result['final_state']['stack'], [7])


class VMTests(unittest.TestCase):
    def test_meta_memory_bounds_and_store_atomicity(self):
        for slot in [-1, 256]:
            for opcode in ['STORE', 'LOAD']:
                _, result = run(f'CONST 7\nSET_META\n{opcode} {slot}\nHALT')
                self.assertEqual(result['traps'], ['INVALID_MEMORY', 'HALT'])
                self.assertEqual(result['final_state']['stack'], [7])
                self.assertEqual(result['final_state']['memory'], {})

    def test_max_depth_loop_can_reenter(self):
        with patch.dict(SPEC, max_loop_depth=1):
            _, result = run('CONST 2\nLOOP_START\nDEC\nDUP\nLOOP_END\nHALT')
        self.assertEqual(result['traps'], ['HALT'])
        self.assertEqual(result['final_state']['stack'], [0])

    def test_actual_stack_limit_all_push_paths(self):
        with patch.dict(SPEC, max_stack_size=1):
            for opcode in ['CONST 2', 'DUP', 'LOAD 0', 'LOADI', 'GETIX']:
                _, result = run('CONST 1\n' + opcode + '\nHALT')
                self.assertEqual(result['traps'], ['STACK_OVERFLOW', 'HALT'])
                self.assertEqual(result['final_state']['stack'], [1])

    def test_gas_never_exceeds_budget(self):
        for budget in [0, 1, 20]:
            _, result = run('again:\nJUMP again', gas_limit=budget)
            self.assertEqual(result['gas_used'], budget)
            self.assertEqual(result['stats']['instructions_executed'], budget)
            self.assertTrue(result['halted'])
            self.assertEqual(result['traps'], ['GAS_EXHAUSTED'])

    def test_trap_storm_reported(self):
        with patch.dict(SPEC, max_traps=2):
            result = PauseLangVM().execute([0]*10, [.05]*10, sync=False)
        self.assertTrue(result['halted'])
        self.assertEqual(result['traps'], ['STACK_UNDERFLOW']*2 + ['TRAP_STORM'])

    def test_halt_and_raise_policies(self):
        _, result = run('POP\nCONST 7', trap_policy='halt')
        self.assertTrue(result['halted'])
        self.assertEqual(result['final_state']['stack'], [])
        with self.assertRaisesRegex(RuntimeError, 'STACK_UNDERFLOW'):
            run('POP', trap_policy='raise')

    def test_integer_division_with_large_operands(self):
        for a, b in [(10**400+3, 1), (2**80+123, 7), (-2**80-123, 7)]:
            _, result = run(f'CONST {a}\nCONST {b}\nDIV2\nHALT')
            quotient = abs(a)//abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            expected = (quotient+2**31) % 2**32 - 2**31
            self.assertEqual(result['final_state']['stack'], [expected])

    def test_demos_have_no_hidden_traps(self):
        original = PauseLangVM.execute
        results = []
        def capture(vm, *args, **kwargs):
            result = original(vm, *args, **kwargs)
            results.append(result)
            return result
        with patch.object(PauseLangVM, 'execute', capture), contextlib.redirect_stdout(io.StringIO()):
            IoTDemos.run_all()
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result['traps'], ['HALT'])
