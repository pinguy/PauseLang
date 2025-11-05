"""
PauseLang v0.7.3 - Bugfix
======================================
Changes in v0.7.3:
- FIXED: `DIV2` now correctly handles the overflow case of INT32_MIN / -1 by wrapping the result, as per spec.
- Retained all v0.7.2 features.
- FIXED: The OVERFLOW flag is now correctly reset before each arithmetic
  operation, preventing a "sticky" flag from a previous overflow.
- Retained all v0.7.3 features.
- FIXED: `LOOP_END` now pops the counter value from the stack instead of
  just peeking. This fixes a stack leak where intermediate loop
  values were retained after the loop terminated.
- UPDATED: `test_loop_memory` rewritten to correctly test for this
  stack leak, as the previous test was flawed and hid the bug.
- Retained all v0.7.4 features.

Changes in v0.7.2:
- FIXED: `LOAD` instruction now correctly pushes the loaded value onto the stack.
- FIXED: `LOAD` instruction definition updated with `stack_delta=1` for proper pre-execution overflow checks.
- Retained all v0.7.1 features.

Original features from v0.6.7 retained.
"""

import time
import random
# ... existing code ...
from math import exp

# === FORMAL SPECIFICATION ===

SPEC = {
    'version': '0.7.3',
    'word_size': 32,
    'overflow': 'wrap',
# ... existing code ...
    0.13: Instruction('SKIP_NEXT', 0.13, '[CONTROL] Skip next instruction', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.14: Instruction('LOOP_START', 0.14, '[CONTROL] Mark loop start', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.15: Instruction('LOOP_END', 0.15, '[CONTROL] Pop TOS; loop if > 0', OpCategory.CONTROL, updates_flags=False, modifies_flow=True, requires_stack=1, stack_delta=-1),
    
    # Memory - HYBRID OPS (use both operand and stack)
    0.16: Instruction('STORE', 0.16, '[HYBRID] Store TOS at mem[operand]', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
# ... existing code ...
                if not (0 <= target < len(data_stream)):
                    self.push_trap(TrapCode.INVALID_JUMP)
                    result = f"INVALID JUMP TARGET {value}"
                else:
                    self.state.pc = target - 1
                    result = f"JUMPED to {value}"
            elif instr.opcode == 'SKIP_NEXT':
                self.state.pc += 1
                result = f"SKIPPING PC {self.state.pc + 1}"
            elif instr.opcode == 'LOOP_START':
                if not self.state.loop_stack or self.state.loop_stack[-1] != self.state.pc:
                    self.state.loop_stack.append(self.state.pc)
                result = "LOOP_START"
            elif instr.opcode == 'LOOP_END':
                # Pre-exec check already confirmed stack has >= 1 item
                if not self.state.loop_stack:
                    result = "LOOP_EXIT (no loop)"
                    # We still pop the value, as the instruction promises
                    self.state.stack.pop() 
                else:
                    counter_value = self.state.stack.pop() # Pop the counter
                    if counter_value > 0:
                        # Loop back: pc is set to LOOP_START
                        self.state.pc = self.state.loop_stack[-1] - 1 
                        result = "LOOP_CONTINUE"
                    else:
                        # Exit loop: pop the loop stack
                        self.state.loop_stack.pop()
                        result = "LOOP_EXIT"
            elif instr.opcode == 'CALL':
                if len(self.state.call_stack) >= SPEC['max_call_depth']:
                    self.push_trap(TrapCode.CALL_DEPTH_EXCEEDED)
# ... existing code ...
            assert 'STACK_UNDERFLOW' in result['traps'], f"{opcode} should trap on empty stack"
        
        return "✓ Stack underflow protection passed"
    
    @staticmethod
    def test_loop_memory():
        """Test loop stack hygiene (LOOP_END must pop)."""
        source = """
        main:
            CONST 3     # Push counter
        loop_label:
            LOOP_START
            DEC         # counter -> counter - 1
            PEEK        # Duplicate for the loop check
            LOOP_END    # Pops the copy, loops if > 0
        # After loop, HALT is implied by end of stream
        """
        # Note: The test will halt naturally. We add HALT for clarity
        # and to ensure no fall-through if test is modified.
        source += "\n HALT"
        
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(gas_limit=1000, debug=False)
        result = vm.execute(data, pauses, labels=labels)
        
        assert len(vm.state.loop_stack) == 0, "LOOP_START/END memory leak detected"
        final_stack = result['final_state']['stack']
        
        # Final stack should be [0] (the final result of the countdown)
        # not [3, 2, 1, 0] which would be a stack leak.
        assert final_stack == [0], f"Loop stack leak detected. Expected [0], got {final_stack}"
        return "✓ LOOP memory management passed"
    
    @staticmethod
    def test_unconditional_jump():
# ... existing code ...
            TortureTests.test_aliases,
            TortureTests.test_unconditional_jump,
            TortureTests.test_division_semantics,
            TortureTests.test_div_overflow, # Added test for the patch
            Tort_ureTests.test_sticky_overflow_flag, # Added test for this patch
            TortureTests.test_jitter_gauntlet,
            Tortagains.test_flag_race,
            TortureTests.test_stack_underflow_protection,
            TortureTests.test_loop_memory, # This test is now updated
            TortureTests.test_fuzz,
        ]
        print("\n🔥 TORTURE TEST SUITE v0.7.3 🔥")
        print("=" * 50)
        passed = 0
        failed = 0
# ... existing code ...
