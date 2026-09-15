"""Original torture suite, retained as regression coverage."""
import random
from PauseLang_v0_7_13 import INSTRUCTIONS, SPEC, PauseLangCompiler, PauseLangVM, TimeQuantizer

class TortureTests:
    @staticmethod
    def test_labels():
        source = """
        start:
            PUSH 5
            SETF 0
            JUMP_IF_ODD skip_even
            PUSH 10
        skip_even:
            PUSH 20
            JZ end
            PUSH 30
        end:
            HALT
        """
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)
        stack = result['final_state']['stack']
        assert 10 not in stack, f"Failed to skip: {stack}"
        assert stack == [5, 20, 30], f"Unexpected stack: {stack}"
        return "✓ Label compilation passed"

    @staticmethod
    def test_aliases():
        source = """
            CONST 42
            PEEK
            DROP
            CONST 0
            SETF 0
            JZ done
            CONST 99
        done:
            HALT
        """
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses)
        stack = result['final_state']['stack']
        assert stack == [42, 0], f"Aliases failed: {stack}"
        assert 99 not in stack, f"Should have jumped over CONST 99"
        return "✓ Instruction aliases passed"

    @staticmethod
    def test_division_semantics():
        vm = PauseLangVM(debug=False)
        tests = [(7,2,3), (-7,2,-3), (7,-2,-3), (-7,-2,3)]
        for a,b,expected in tests:
            vm.reset()
            pauses = [0.045, 0.045, 0.115]
            data = [a,b,0]
            result = vm.execute(data, pauses, sync=False)
            actual = result['final_state']['stack'][0]
            assert actual == expected, f"DIV2({a},{b}) = {actual}, expected {expected}"
        mod_tests = [(7,3,1), (-7,3,2), (7,-3,1), (-7,-3,2)]
        for a,b,expected in mod_tests:
            vm.reset()
            pauses = [0.045, 0.045, 0.120]
            data = [a,b,0]
            result = vm.execute(data, pauses, sync=False)
            actual = result['final_state']['stack'][0]
            assert actual == expected, f"MOD2({a},{b}) = {actual}, expected {expected}"
        return "✓ Division/modulo semantics passed"

    @staticmethod
    def test_jitter_gauntlet():
        vm = PauseLangVM(debug=False)
        pauses = [0.045, 0.045, 0.100]
        data = [5, 3, 0]
        for _ in range(100):
            jittered = [p + random.uniform(-0.0007, 0.0007) for p in pauses]
            result = vm.execute(data, jittered, sync=False)
            vm.reset()
            opcodes = [r[1] for r in result['results']]
            assert opcodes == ['PUSH', 'PUSH', 'ADD2'], f"Jitter broke decoding: {opcodes}"
        return "✓ Jitter gauntlet passed"

    @staticmethod
    def test_flag_race():
        vm = PauseLangVM(debug=False)
        pauses = [0.045, 0.040, 0.045, 0.100, 0.045, 0.120]
        data = [7, 7, 3, 0, 2, 0]
        result = vm.execute(data, pauses, sync=False)
        final_flags = result['final_state']['flags']
        assert final_flags['ZERO'] == True, f"Expected ZERO flag, got {final_flags}"
        return "✓ Flag race passed"

    @staticmethod
    def test_stack_underflow_protection():
        vm = PauseLangVM(debug=False)
        ops_to_test = [(0.050, 'POP'), (0.055, 'DUP'), (0.200, 'SETIX')]
        for pause, opcode in ops_to_test:
            vm.reset()
            result = vm.execute([0], [pause], sync=False)
            assert 'STACK_UNDERFLOW' in result['traps'], f"{opcode} should trap on empty stack"
        return "✓ Stack underflow protection passed"

    @staticmethod
    def test_loop_memory():
        source = """
        main:
            CONST 3
        loop_label:
            LOOP_START
            DEC
            PEEK
            LOOP_END
            HALT
        """
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(gas_limit=1000, debug=False)
        result = vm.execute(data, pauses, labels=labels)
        assert len(vm.state.loop_stack) == 0, "LOOP_START/END memory leak detected"
        final_stack = result['final_state']['stack']
        assert final_stack == [0], f"Loop stack leak detected. Expected [0], got {final_stack}"
        return "✓ LOOP memory management passed"

    @staticmethod
    def test_unconditional_jump():
        source = """
        main:
            CONST 100
            JMP skip
            CONST 200
            CONST 300
        skip:
            CONST 400
            HALT
        """
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)
        stack = result['final_state']['stack']
        assert stack == [100, 400], f"JUMP failed: {stack}"
        assert 200 not in stack and 300 not in stack, f"Failed to skip: {stack}"
        return "✓ Unconditional JUMP passed"

    @staticmethod
    def test_div_overflow():
        vm = PauseLangVM(debug=False)
        INT32_MIN = -2**31
        vm.reset()
        pauses = [0.045, 0.045, 0.115]
        data = [INT32_MIN, -1, 0]
        result = vm.execute(data, pauses, sync=False)
        stack = result['final_state']['stack']
        flags = result['final_state']['flags']
        assert stack == [INT32_MIN], f"DIV2 overflow failed: expected [{INT32_MIN}], got {stack}"
        assert flags['OVERFLOW'] == True, "DIV2 overflow did not set OVERFLOW flag"
        return "✓ DIV2 overflow (MIN / -1) passed"

    @staticmethod
    def test_sticky_overflow_flag():
        vm = PauseLangVM(debug=False)
        INT32_MAX = 2**31 - 1
        vm.reset()
        pauses = [0.045, 0.045, 0.100]
        data = [INT32_MAX, INT32_MAX, 0]
        result = vm.execute(data, pauses, sync=False)
        flags = result['final_state']['flags']
        assert flags['OVERFLOW'] == True, "First ADD2 should set OVERFLOW"
        pauses.extend([0.045, 0.045, 0.100])
        data.extend([1, 2, 0])
        result = vm.execute(data, pauses, sync=False)
        flags = result['final_state']['flags']
        assert flags['OVERFLOW'] == False, "Second ADD2 should reset OVERFLOW flag"
        return "✓ Sticky overflow flag fix passed"

    @staticmethod
    def test_stack_growth_protection():
        vm = PauseLangVM(debug=False)
        source = "CONST 1\n"
        for _ in range(20):
            source += "    DUP\n"
        source += "    HALT"
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        result = vm.execute(data, pauses, labels=labels)
        assert result['stats']['stack_high_water'] > 0, "Stack high water not tracked"
        assert result['final_state']['stack_high_water'] > 0, "Stack high water not in state"
        return "✓ Stack growth protection passed"

    @staticmethod
    def test_loop_depth_protection():
        vm = PauseLangVM(debug=False, gas_limit=50000)
        source = "CONST 2\n"
        for i in range(300):
            source += f"loop{i}:\n    LOOP_START\n"
        source += "    CONST 1\n"
        for i in range(300):
            source += "    LOOP_END\n"
        source += "HALT\n"
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        result = vm.execute(data, pauses, labels=labels)
        assert 'LOOP_DEPTH_EXCEEDED' in result['traps'], "Loop depth limit not enforced"
        return "✓ Loop depth protection passed"

    @staticmethod
    def test_macros_not():
        source_not = "CONST 0\nNOT\nHALT"
        pauses, data, _, _ = PauseLangCompiler.compile(source_not)
        vm = PauseLangVM(debug=False)
        res = vm.execute(data, pauses, sync=False)
        assert res['final_state']['stack'] == [-1], f"NOT(0) failed: {res['final_state']['stack']}"
        source_not2 = "CONST 1\nNOT\nHALT"
        pauses, data, _, _ = PauseLangCompiler.compile(source_not2)
        vm = PauseLangVM(debug=False)
        res = vm.execute(data, pauses, sync=False)
        assert res['final_state']['stack'] == [-2], f"NOT(1) failed: {res['final_state']['stack']}"
        source_lnot = "CONST 0\nLNOT\nHALT"
        pauses, data, _, _ = PauseLangCompiler.compile(source_lnot)
        vm = PauseLangVM(debug=False)
        res = vm.execute(data, pauses, sync=False)
        assert res['final_state']['stack'] == [1], f"LNOT(0) failed: {res['final_state']['stack']}"
        source_neg = "CONST 5\nNEG\nHALT"
        pauses, data, _, _ = PauseLangCompiler.compile(source_neg)
        vm = PauseLangVM(debug=False)
        res = vm.execute(data, pauses, sync=False)
        assert res['final_state']['stack'] == [-5], f"NEG(5) failed: {res['final_state']['stack']}"
        return "✓ NOT/LNOT/NEG macros passed"

    @staticmethod
    def test_ret_without_call():
        vm = PauseLangVM(debug=False)
        pauses = [INSTRUCTIONS[140].pause]  # RET
        data = [0]
        result = vm.execute(data, pauses, sync=False)
        assert 'RETURN_WITHOUT_CALL' in result['traps'], "RET without CALL should trap"
        return "✓ RET trap works"

    @staticmethod
    def test_loadi_uninit():
        """LOADI must return 0 for uninitialised slots, not trap."""
        vm = PauseLangVM(debug=False)
        # PUSH 42, SETIX, LOADI, HALT
        pauses = [INSTRUCTIONS[k].pause for k in [45, 200, 205, 150]]
        data   = [42, 0, 0, 0]
        result = vm.execute(data, pauses, sync=False)
        stack = result['final_state']['stack']
        assert vm.state.ix == 42
        assert result["traps"] == ["HALT"], result["traps"]
        assert stack == [0], f"LOADI on uninit should push 0, got {stack}"
        assert 'INVALID_MEMORY' not in result['traps'], "LOADI should not trap on uninit"
        return "✓ LOADI uninitialised returns 0"

    @staticmethod
    def test_rot():
        """ROT: ( a b c -- b c a )"""
        vm = PauseLangVM(debug=False)
        P = INSTRUCTIONS[45].pause   # PUSH
        R = INSTRUCTIONS[165].pause  # ROT
        H = INSTRUCTIONS[150].pause  # HALT
        pauses = [P, P, P, R, H]
        data   = [10, 20, 30, 0, 0]
        result = vm.execute(data, pauses, sync=False)
        stack = result['final_state']['stack']
        assert stack == [20, 30, 10], f"ROT failed: expected [20, 30, 10], got {stack}"
        return "✓ ROT instruction passed"

    @staticmethod
    def test_rot_underflow():
        """ROT with < 3 items should trap."""
        vm = PauseLangVM(debug=False)
        pauses = [INSTRUCTIONS[45].pause, INSTRUCTIONS[165].pause]  # PUSH + ROT (only 1 item)
        data   = [99, 0]
        result = vm.execute(data, pauses, sync=False)
        assert 'STACK_UNDERFLOW' in result['traps'], "ROT with <3 items should trap"
        return "✓ ROT underflow protection passed"

    @staticmethod
    def test_strict_sync():
        """strict_sync=True must NOT auto-strip the sync phrase."""
        P = INSTRUCTIONS[45].pause   # PUSH
        H = INSTRUCTIONS[150].pause  # HALT
        pauses = SPEC['sync_phrase'] + [P, H]
        data   = [0, 0, 42, 0]

        # Default behavior: auto-strip → only PUSH + HALT run
        vm = PauseLangVM(debug=False)
        res_normal = vm.execute(data, pauses, sync=False, strict_sync=False)
        assert res_normal['final_state']['stack'] == [42], \
            f"Default (strict_sync=False) should auto-strip and push 42, got {res_normal['final_state']['stack']}"

        # strict_sync=True: sync phrase treated as normal instructions → should produce INVALID_INSTRUCTION
        vm = PauseLangVM(debug=False)
        res_strict = vm.execute(data, pauses, sync=False, strict_sync=True)
        assert 'INVALID_INSTRUCTION' in res_strict['traps'], \
            "strict_sync=True should treat sync phrase as invalid opcodes"
        return "✓ strict_sync parameter passed"

    @staticmethod
    def test_short_sync_phrase():
        """Verify short 2-symbol sync works with calibration."""
        vm = PauseLangVM(debug=False)
        P = INSTRUCTIONS[45].pause
        H = INSTRUCTIONS[150].pause
        pauses = SPEC['sync_phrase'] + [P, H]
        data   = [0, 0, 42, 0]
        result = vm.execute(data, pauses, sync=True, strict_sync=False)
        assert 'error' not in result
        assert result['final_state']['stack'] == [42]
        return "✓ Short 2-symbol sync phrase passed"

    @staticmethod
    def test_sync_jitter_tolerance():
        """Sync phrase should tolerate small jitter within guard band."""
        vm = PauseLangVM(debug=False)
        P = INSTRUCTIONS[45].pause
        H = INSTRUCTIONS[150].pause
        for _ in range(30):
            jittered = [p + random.uniform(-0.0008, 0.0008) for p in SPEC['sync_phrase']]
            pauses = jittered + [P, H]
            data   = [0, 0, 99, 0]
            result = vm.execute(data, pauses, sync=True, strict_sync=False)
            assert 'error' not in result
            vm.reset()
        return "✓ Sync jitter tolerance passed"

    @staticmethod
    def test_ix_wrapping():
        """IX must wrap around at max_memory_slots (256)."""
        vm = PauseLangVM(debug=False)
        # PUSH 255, SETIX, INCIX, GETIX, HALT
        pauses = [INSTRUCTIONS[k].pause for k in [45, 200, 215, 220, 150]]
        data   = [255, 0, 0, 0, 0]
        result = vm.execute(data, pauses, sync=False)
        assert result['final_state']['stack'] == [0]
        assert result['final_state']['ix'] == 0
        return "✓ IX register wrapping passed"

    @staticmethod
    def test_store_worked_example():
        """Verify documented STORE example: PUSH 99 / STORE 42 → mem[42] = 99, then LOAD 42 → pushes 99."""
        vm = PauseLangVM(debug=False)
        # PUSH 99, STORE 42, LOAD 42, HALT
        pauses = [INSTRUCTIONS[k].pause for k in [45, 80, 85, 150]]
        data   = [99, 42, 42, 0]
        result = vm.execute(data, pauses, sync=False)
        mem = result['final_state']['memory']
        stack = result['final_state']['stack']
        assert mem.get(42) == 99, f"STORE failed: mem[42] = {mem.get(42)}"
        assert stack == [99], f"LOAD should have pushed 99, got {stack}"
        return "✓ STORE worked example verified"

    @staticmethod
    def test_fuzz_v077():
        """Fuzz with full v0.7.13 instruction set (including ROT).
        Pause stream uses canonical float pauses (seconds) from instr.pause."""
        vm = PauseLangVM(debug=False)
        all_pauses = [instr.pause for instr in INSTRUCTIONS.values()]
        for _ in range(150):
            length = random.randint(4, 20)
            pauses = [random.choice(all_pauses) for _ in range(length)]
            data = [random.randint(-200, 200) for _ in range(length)]
            try:
                vm.execute(data, pauses, sync=False)
            except Exception as e:
                raise AssertionError(f"Fuzz crash: {e}")
            vm.reset()
        return "✓ Fuzz test passed (v0.7.13 ISA)"

    @staticmethod
    def test_gas_exhaustion_halted():
        """GAS_EXHAUSTED should set halted=True."""
        vm = PauseLangVM(gas_limit=2, debug=False)
        P = INSTRUCTIONS[45].pause  # PUSH
        pauses = [P, P, P]  # three PUSHes, gas limit 2
        data   = [1, 2, 3]
        result = vm.execute(data, pauses, sync=False)
        assert result['halted'] is True, "GAS_EXHAUSTED did not set halted"
        assert 'GAS_EXHAUSTED' in result['traps']
        return "✓ GAS_EXHAUSTED sets halted flag"

    @staticmethod
    def test_jitter_no_snap():
        """Large jitter should produce INVALID_INSTRUCTION, not a silent wrong opcode."""
        vm = PauseLangVM(debug=False)
        # Expect PUSH (0.045) but add +3ms jitter → 0.048, beyond 1.5ms guard band
        pauses = [0.048, INSTRUCTIONS[150].pause]
        data   = [42, 0]
        result = vm.execute(data, pauses, sync=False)
        assert 'INVALID_INSTRUCTION' in result['traps'], "Large jitter should trap, not snap to MEAN"
        opcodes = [r[1] for r in result['results']]
        assert opcodes[0] == 'PASS', "Fallback PASS should be used on invalid decode"
        return "✓ Jitter no longer snaps to wrong opcode"

    @staticmethod
    def test_loop_mismatch_trap():
        """LOOP_END without LOOP_START must use the dedicated trap code."""
        source = "CONST 9\nLOOP_END\nHALT"
        pauses, data, _, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)
        assert 'LOOP_MISMATCH' in result['traps'], result['traps']
        assert 'INVALID_INSTRUCTION' not in result['traps'], result['traps']
        return "✓ Dedicated LOOP_MISMATCH trap passed"

    @staticmethod
    def test_trace_pc_accuracy():
        """Control-flow trace must record the instruction that executed, not its target."""
        source = """
        start:
            CONST 1
            JUMP target
            CONST 999
        target:
            CONST 2
            HALT
        """
        pauses, data, _, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        vm.execute(data, pauses, labels=labels)
        pcs = [step['absolute_pc'] for step in vm.execution_trace]
        assert pcs == [2, 3, 5, 6], f"Wrong trace PCs: {pcs}"
        return "✓ Control-flow trace PC accuracy passed"

    @staticmethod
    def test_guard_boundary_stability():
        """Exactly ±guard_band must decode inclusively despite float representation."""
        target = INSTRUCTIONS[45].pause
        q = TimeQuantizer()
        assert q.in_guard_band(target + q.guard_band, target)
        assert q.in_guard_band(target - q.guard_band, target)
        return "✓ Guard boundary stability passed"

    @staticmethod
    def run_all():
        tests = [
            TortureTests.test_labels,
            TortureTests.test_aliases,
            TortureTests.test_unconditional_jump,
            TortureTests.test_division_semantics,
            TortureTests.test_div_overflow,
            TortureTests.test_sticky_overflow_flag,
            TortureTests.test_jitter_gauntlet,
            TortureTests.test_flag_race,
            TortureTests.test_stack_underflow_protection,
            TortureTests.test_loop_memory,
            TortureTests.test_stack_growth_protection,
            TortureTests.test_loop_depth_protection,
            TortureTests.test_macros_not,
            TortureTests.test_ret_without_call,
            TortureTests.test_loadi_uninit,
            TortureTests.test_rot,
            TortureTests.test_rot_underflow,
            TortureTests.test_strict_sync,
            TortureTests.test_short_sync_phrase,
            TortureTests.test_sync_jitter_tolerance,
            TortureTests.test_ix_wrapping,
            TortureTests.test_store_worked_example,
            TortureTests.test_fuzz_v077,
            TortureTests.test_gas_exhaustion_halted,
            TortureTests.test_jitter_no_snap,
            TortureTests.test_loop_mismatch_trap,
            TortureTests.test_trace_pc_accuracy,
            TortureTests.test_guard_boundary_stability,
        ]
        print("\n🔥 TORTURE TEST SUITE v0.7.13 🔥")
        print("=" * 50)
        passed = 0
        failed = 0
        for test in tests:
            try:
                result = test()
                print(result)
                passed += 1
            except AssertionError as e:
                print(f"✗ {test.__name__} FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"✗ {test.__name__} ERROR: {e}")
                failed += 1
        print("=" * 50)
        print(f"Results: {passed} passed, {failed} failed")
        return passed, failed

