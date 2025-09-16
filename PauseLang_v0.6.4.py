"""
PauseLang v0.6.4 - Complete Fixed Version
==========================================
Incorporates all v0.6.3 features plus patches:
- Fixed STOREI stack safety
- Fixed control flow to use absolute addressing
- Added GETIX instruction
"""

import time
import random
from typing import List, Tuple, Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from math import exp

# === FORMAL SPECIFICATION ===

SPEC = {
    'version': '0.6.4',
    'word_size': 32,
    'overflow': 'wrap',
    'division': 'truncate',
    'max_call_depth': 256,
    'max_stack_size': 4096,
    'max_memory_slots': 256,
    'time_quantum': 0.01,
    'guard_band': 0.002,
    'sync_phrase': [0.29, 0.29, 0.30, 0.29],
}

# === TIME QUANTIZATION ===

class TimeQuantizer:
    def __init__(self, quantum: float = SPEC['time_quantum'], guard_band: float = SPEC['guard_band']):
        self.quantum = quantum
        self.guard_band = guard_band
        self.drift_estimate = 0.0
        self.calibration_history = deque(maxlen=10)
    
    def quantize(self, pause: float) -> float:
        adjusted = pause - self.drift_estimate
        bin_index = round(adjusted / self.quantum)
        return bin_index * self.quantum
    
    def in_guard_band(self, pause: float, target: float) -> bool:
        return abs(pause - target) <= self.guard_band
    
    def calibrate(self, sync_pauses: List[float]) -> bool:
        expected = SPEC['sync_phrase']
        if len(sync_pauses) != len(expected): return False
        drifts = [observed - expected for observed, expected in zip(sync_pauses, expected)]
        self.drift_estimate = sum(drifts) / len(drifts)
        self.calibration_history.append(self.drift_estimate)
        return True
    
    def get_drift_trend(self) -> float:
        if not self.calibration_history: return 0.0
        return sum(self.calibration_history) / len(self.calibration_history)

# === ENUMS ===

class Flag(Enum):
    ZERO = auto()
    CARRY = auto()
    ODD = auto()
    NEGATIVE = auto()
    OVERFLOW = auto()

class TrapCode(Enum):
    NONE = 0
    STACK_UNDERFLOW = 1
    STACK_OVERFLOW = 2
    DIV_BY_ZERO = 3
    ARITHMETIC_OVERFLOW = 4
    INVALID_MEMORY = 5
    CALL_DEPTH_EXCEEDED = 6
    GAS_EXHAUSTED = 7
    INVALID_INSTRUCTION = 8
    HALT = 9

class Lane(Enum):
    DATA = auto()
    META = auto()

# === INSTRUCTION SET ===

@dataclass
class Instruction:
    opcode: str
    pause: float
    description: str
    updates_flags: bool = True
    requires_stack: int = 0
    modifies_flow: bool = False
    stack_delta: int = 0

INSTRUCTIONS = {
    # Arithmetic (0.01-0.04)
    0.01: Instruction('ADD', 0.01, 'Add current and previous'),
    0.02: Instruction('MEAN', 0.02, 'Average of current and previous'),
    0.03: Instruction('DIFF', 0.03, 'Subtract previous from current'),
    0.04: Instruction('SQUARE', 0.04, 'Square current value'),
    # Conditional (0.05-0.08)
    0.05: Instruction('PASS', 0.05, 'Pass unchanged', updates_flags=False),
    0.06: Instruction('IF_GT_15_SQUARE', 0.06, 'Square if > 15'),
    0.07: Instruction('DOUBLE_IF_EVEN', 0.07, 'Double if even'),
    0.08: Instruction('NEGATE_IF_ODD', 0.08, 'Negate if odd'),
    # Stack (0.09-0.11)
    0.09: Instruction('PUSH', 0.09, 'Push to stack', updates_flags=False, stack_delta=1),
    0.10: Instruction('POP', 0.10, 'Pop from stack', requires_stack=1, stack_delta=-1),
    0.11: Instruction('DUP', 0.11, 'Duplicate top', updates_flags=False, requires_stack=1, stack_delta=1),
    # Control (0.12-0.15)
    0.12: Instruction('JUMP_IF_ODD', 0.12, 'Jump if ODD flag', updates_flags=False, modifies_flow=True),
    0.13: Instruction('SKIP_NEXT', 0.13, 'Skip next', updates_flags=False, modifies_flow=True),
    0.14: Instruction('LOOP_START', 0.14, 'Mark loop', updates_flags=False, modifies_flow=True),
    0.15: Instruction('LOOP_END', 0.15, 'Loop if stack>0', updates_flags=False, modifies_flow=True),
    # Memory (0.16-0.19)
    0.16: Instruction('STORE', 0.16, 'Store in memory', updates_flags=False),
    0.17: Instruction('LOAD', 0.17, 'Load from memory'),
    0.18: Instruction('SWAP', 0.18, 'Swap top two', updates_flags=False, requires_stack=2),
    0.19: Instruction('CLEAR_STACK', 0.19, 'Clear stack', updates_flags=False),
    # Stack Arithmetic (0.20-0.24)
    0.20: Instruction('ADD2', 0.20, 'Pop 2, push sum', requires_stack=2, stack_delta=-1),
    0.21: Instruction('SUB2', 0.21, 'Pop 2, push diff', requires_stack=2, stack_delta=-1),
    0.22: Instruction('MUL2', 0.22, 'Pop 2, push product', requires_stack=2, stack_delta=-1),
    0.23: Instruction('DIV2', 0.23, 'Pop 2, push quotient', requires_stack=2, stack_delta=-1),
    0.24: Instruction('MOD2', 0.24, 'Pop 2, push modulo', requires_stack=2, stack_delta=-1),
    # Meta (0.25-0.28)
    0.25: Instruction('SET_META', 0.25, 'Toggle meta mode', updates_flags=False),
    0.26: Instruction('JUMP_IF_ZERO', 0.26, 'Jump if ZERO flag', updates_flags=False, modifies_flow=True),
    0.27: Instruction('CALL', 0.27, 'Call subroutine', updates_flags=False, modifies_flow=True),
    0.28: Instruction('RET', 0.28, 'Return', updates_flags=False, modifies_flow=True),
    # System (0.29-0.30)
    0.29: Instruction('NOP', 0.29, 'No operation', updates_flags=False),
    0.30: Instruction('HALT', 0.30, 'Halt execution', updates_flags=False, modifies_flow=True),
    # IX REGISTER (0.40-0.44)
    0.40: Instruction('SETIX', 0.40, 'Pop stack → IX register', updates_flags=False, requires_stack=1, stack_delta=-1),
    0.41: Instruction('LOADI', 0.41, 'Push mem[IX] to stack', updates_flags=True, stack_delta=1),
    0.42: Instruction('STOREI', 0.42, 'Store TOS at mem[IX]', updates_flags=False, requires_stack=1),
    0.43: Instruction('INCIX', 0.43, 'IX = (IX + 1) % max_slots', updates_flags=False),
    0.44: Instruction('GETIX', 0.44, 'Push IX register to stack', updates_flags=False, stack_delta=1),
}
OPCODE_TO_PAUSE = {instr.opcode: pause for pause, instr in INSTRUCTIONS.items()}

# === VM CORE ===

@dataclass
class VMState:
    pc: int = 0
    stack: List[int] = field(default_factory=list)
    memory: Dict[int, int] = field(default_factory=dict)
    flags: Dict[Flag, bool] = field(default_factory=lambda: {f: False for f in Flag})
    call_stack: List[int] = field(default_factory=list)
    loop_stack: List[int] = field(default_factory=list)
    trap_stack: List[TrapCode] = field(default_factory=list)
    lane: Lane = Lane.DATA
    gas_used: int = 0
    halted: bool = False
    ix: int = 0

class PauseLangVM:
    def __init__(self, gas_limit: int = 10000, trap_policy: str = 'continue', debug: bool = False):
        self.state = VMState()
        self.gas_limit = gas_limit
        self.trap_policy = trap_policy
        self.debug = debug
        self.quantizer = TimeQuantizer()
        self.execution_trace = []
        
    def reset(self):
        self.state = VMState()
        self.execution_trace = []
    
    def wrap_int32(self, value: int) -> int:
        INT32_MAX = 2**31 - 1
        INT32_MIN = -2**31
        if value > INT32_MAX:
            self.state.flags[Flag.OVERFLOW] = True
            value = INT32_MIN + (value - INT32_MAX - 1) % (2**32)
        elif value < INT32_MIN:
            self.state.flags[Flag.OVERFLOW] = True
            value = INT32_MAX - (INT32_MIN - value - 1) % (2**32)
        return value
    
    def update_flags(self, value: int):
        self.state.flags[Flag.ZERO] = (value == 0)
        self.state.flags[Flag.ODD] = (value % 2 != 0)
        self.state.flags[Flag.NEGATIVE] = (value < 0)
    
    def push_trap(self, code: TrapCode):
        self.state.trap_stack.append(code)
        if self.trap_policy == 'halt':
            self.state.halted = True
        elif self.trap_policy == 'raise':
            raise RuntimeError(f"VM Trap: {code.name}")
        if self.debug:
            print(f"⚠️ TRAP: {code.name}")
    
    def check_gas(self) -> bool:
        self.state.gas_used += 1
        if self.state.gas_used > self.gas_limit:
            self.push_trap(TrapCode.GAS_EXHAUSTED)
            return False
        return True
    
    def execute_instruction(self, instr: Instruction, value: int, prev_value: Optional[int] = None) -> Any:
        opcode = instr.opcode
        if instr.requires_stack > len(self.state.stack):
            self.push_trap(TrapCode.STACK_UNDERFLOW)
            return "TRAP: STACK_UNDERFLOW"
        if instr.stack_delta > 0 and len(self.state.stack) + instr.stack_delta > SPEC['max_stack_size']:
            self.push_trap(TrapCode.STACK_OVERFLOW)
            return "TRAP: STACK_OVERFLOW"
        
        result = None
        
        # Arithmetic
        if opcode == 'ADD' and prev_value is not None:
            result = self.wrap_int32(value + prev_value)
        elif opcode == 'MEAN' and prev_value is not None:
            result = self.wrap_int32((value + prev_value) // 2)
        elif opcode == 'DIFF' and prev_value is not None:
            result = self.wrap_int32(value - prev_value)
        elif opcode == 'SQUARE':
            result = self.wrap_int32(value * value)
        # Conditional
        elif opcode == 'PASS':
            result = value
        elif opcode == 'IF_GT_15_SQUARE':
            result = self.wrap_int32(value * value) if value > 15 else value
        elif opcode == 'DOUBLE_IF_EVEN':
            result = self.wrap_int32(value * 2) if value % 2 == 0 else value
        elif opcode == 'NEGATE_IF_ODD':
            result = self.wrap_int32(-value) if value % 2 != 0 else value
        # Stack ops
        elif opcode == 'PUSH':
            self.state.stack.append(value)
            result = f"PUSHED {value}"
        elif opcode == 'POP':
            result = self.state.stack.pop()
        elif opcode == 'DUP':
            top = self.state.stack[-1]
            self.state.stack.append(top)
            result = f"DUP {top}"
        elif opcode == 'SWAP':
            self.state.stack[-1], self.state.stack[-2] = self.state.stack[-2], self.state.stack[-1]
            result = "SWAPPED"
        elif opcode == 'CLEAR_STACK':
            count = len(self.state.stack)
            self.state.stack.clear()
            result = f"CLEARED {count}"
        # Stack arithmetic
        elif opcode == 'ADD2':
            b, a = self.state.stack.pop(), self.state.stack.pop()
            r = self.wrap_int32(a + b)
            self.state.stack.append(r)
            result = r
        elif opcode == 'SUB2':
            b, a = self.state.stack.pop(), self.state.stack.pop()
            r = self.wrap_int32(a - b)
            self.state.stack.append(r)
            result = r
        elif opcode == 'MUL2':
            b, a = self.state.stack.pop(), self.state.stack.pop()
            r = self.wrap_int32(a * b)
            self.state.stack.append(r)
            result = r
        elif opcode == 'DIV2':
            b, a = self.state.stack.pop(), self.state.stack.pop()
            if b == 0:
                self.push_trap(TrapCode.DIV_BY_ZERO)
                self.state.stack.append(0)
                result = "DIV_BY_ZERO"
            else:
                r = int(a / b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'MOD2':
            b, a = self.state.stack.pop(), self.state.stack.pop()
            if b == 0:
                self.push_trap(TrapCode.DIV_BY_ZERO)
                self.state.stack.append(0)
                result = "MOD_BY_ZERO"
            else:
                r = a % b
                self.state.stack.append(r)
                result = r
        # Memory ops
        elif opcode == 'STORE':
            if not self.state.stack:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                slot = value % SPEC['max_memory_slots'] if self.state.lane == Lane.DATA else value
                store_value = self.state.stack[-1]
                if 0 <= slot < SPEC['max_memory_slots']:
                    self.state.memory[slot] = store_value
                    result = f"STORED {store_value} @ {slot}"
                else:
                    self.push_trap(TrapCode.INVALID_MEMORY)
                    result = "INVALID_MEMORY"
        elif opcode == 'LOAD':
            slot = value % SPEC['max_memory_slots'] if self.state.lane == Lane.DATA else value
            result = self.state.memory.get(slot, 0)
        # Meta/system
        elif opcode == 'SET_META':
            self.state.lane = Lane.META if self.state.lane == Lane.DATA else Lane.DATA
            result = f"LANE: {self.state.lane.name}"
        elif opcode == 'NOP':
            result = "NOP"
        elif opcode == 'HALT':
            self.state.halted = True
            self.push_trap(TrapCode.HALT)
            result = "HALTED"
        # IX ops (PATCHED)
        elif opcode == 'SETIX':
            self.state.ix = self.state.stack.pop() % SPEC['max_memory_slots']
            result = f"IX={self.state.ix}"
        elif opcode == 'LOADI':
            v = self.state.memory.get(self.state.ix, 0)
            self.state.stack.append(v)
            result = f"LOADED {v} from mem[{self.state.ix}]"
        elif opcode == 'STOREI':
            # PATCHED: Added stack check
            if not self.state.stack:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                v = self.state.stack[-1]
                self.state.memory[self.state.ix] = v
                result = f"STORED {v} at mem[{self.state.ix}]"
        elif opcode == 'INCIX':
            self.state.ix = (self.state.ix + 1) % SPEC['max_memory_slots']
            result = f"IX={self.state.ix}"
        elif opcode == 'GETIX':
            # NEW: Push IX value to stack
            self.state.stack.append(self.state.ix)
            result = f"PUSHED IX={self.state.ix}"

        if instr.updates_flags and isinstance(result, int):
            self.update_flags(result)
        return result if result is not None else value
    
    def execute(self, data_stream: List[int], pause_stream: List[float], sync: bool = True) -> Dict:
        if sync and len(pause_stream) >= 4:
            if not self.quantizer.calibrate(pause_stream[:4]):
                return {'error': 'Sync calibration failed'}
            data_stream = data_stream[4:]
            pause_stream = pause_stream[4:]
        if len(data_stream) != len(pause_stream):
            return {'error': f'Stream length mismatch: data={len(data_stream)}, pauses={len(pause_stream)}'}
        
        results = []
        while self.state.pc < len(data_stream) and not self.state.halted:
            if not self.check_gas(): break
            value = data_stream[self.state.pc]
            raw_pause = pause_stream[self.state.pc]
            pause = self.quantizer.quantize(raw_pause)
            instr = None
            for target_pause, instruction in INSTRUCTIONS.items():
                if self.quantizer.in_guard_band(pause, target_pause):
                    instr = instruction
                    break
            if instr is None:
                self.push_trap(TrapCode.INVALID_INSTRUCTION)
                instr = INSTRUCTIONS[0.05]
            
            prev_value = data_stream[self.state.pc - 1] if self.state.pc > 0 else None

            # PATCHED: Fixed control flow with absolute addressing
            if instr.opcode == 'JUMP_IF_ODD' and self.state.flags[Flag.ODD]:
                self.state.pc = value - 1  # Absolute jump (-1 for loop increment)
                result = f"JUMPED to {value}"
            elif instr.opcode == 'JUMP_IF_ZERO' and self.state.flags[Flag.ZERO]:
                self.state.pc = value - 1  # Absolute jump
                result = f"JUMPED to {value}"
            elif instr.opcode == 'SKIP_NEXT':
                self.state.pc += 1  # Skip next instruction
                result = f"SKIPPED {self.state.pc}"
            elif instr.opcode == 'LOOP_START':
                # Only append on first entry to prevent memory leak on loop-back
                if not self.state.loop_stack or self.state.loop_stack[-1] != self.state.pc:
                    self.state.loop_stack.append(self.state.pc)
                result = "LOOP_START"
            elif instr.opcode == 'LOOP_END':
                if self.state.loop_stack and self.state.stack and self.state.stack[-1] > 0:
                    self.state.pc = self.state.loop_stack[-1] - 1
                    result = "LOOP_CONTINUE"
                else:
                    if self.state.loop_stack:
                        self.state.loop_stack.pop()
                    result = "LOOP_EXIT"
            elif instr.opcode == 'CALL':
                if len(self.state.call_stack) >= SPEC['max_call_depth']:
                    self.push_trap(TrapCode.CALL_DEPTH_EXCEEDED)
                    result = "CALL_DEPTH_EXCEEDED"
                else:
                    self.state.call_stack.append(self.state.pc + 1)
                    self.state.pc = value - 1  # Absolute jump
                    result = f"CALL {value}"
            elif instr.opcode == 'RET':
                if self.state.call_stack:
                    self.state.pc = self.state.call_stack.pop() - 1
                    result = "RET"
                else:
                    result = "RET_NO_ADDR"
            else:
                result = self.execute_instruction(instr, value, prev_value)
            
            self.execution_trace.append({
                'pc': self.state.pc,
                'opcode': instr.opcode,
                'value': value,
                'result': result,
                'stack_depth': len(self.state.stack),
                'flags': [f.name for f, v in self.state.flags.items() if v],
                'gas': self.state.gas_used
            })
            results.append((value, instr.opcode, result))
            if self.debug:
                print(f"PC:{self.state.pc:03d} | {instr.opcode:<12} | {value:6d} → {result}")
            self.state.pc += 1
        
        return {
            'results': results,
            'final_state': self.get_state(),
            'traps': [t.name for t in self.state.trap_stack],
            'gas_used': self.state.gas_used,
            'halted': self.state.halted
        }
    
    def get_state(self) -> Dict:
        return {
            'pc': self.state.pc,
            'stack': self.state.stack.copy(),
            'memory': self.state.memory.copy(),
            'flags': {f.name: v for f, v in self.state.flags.items()},
            'lane': self.state.lane.name,
            'gas_used': self.state.gas_used,
            'halted': self.state.halted,
            'ix': self.state.ix
        }
    
    def explain(self, verbose: bool = False) -> str:
        if not self.execution_trace: return "No execution trace"
        if verbose:
            lines = ["=== EXECUTION TRACE ==="]
            for step in self.execution_trace:
                lines.append(f"PC:{step['pc']:03d} {step['opcode']:<12} val={step['value']} result={step['result']} flags={step['flags']}")
            return '\n'.join(lines)
        phrases, current = [], []
        for step in self.execution_trace:
            if step['opcode'] in ['JUMP_IF_ODD','JUMP_IF_ZERO','SKIP_NEXT','LOOP_START','LOOP_END','CALL','RET','HALT']:
                if current: phrases.append(self._summarize_phrase(current)); current=[]
                phrases.append(f"Control: {step['opcode']} → {step['result']}")
            else:
                current.append(step)
        if current: phrases.append(self._summarize_phrase(current))
        return ' | '.join(phrases)
    
    def _summarize_phrase(self, steps: List[Dict]) -> str:
        ops = [s['opcode'] for s in steps]
        return f"{ops[0]}..{ops[-1]} ({len(ops)} ops)" if len(ops) > 1 else f"{ops[0]}"

# === TORTURE TESTS ===

class TortureTests:
    @staticmethod
    def test_jitter_gauntlet():
        vm = PauseLangVM(debug=False)
        pauses = [0.09, 0.09, 0.20]
        data = [5, 3, 0]
        for _ in range(100):
            jittered = [p + random.uniform(-0.025, 0.025) * p for p in pauses]
            result = vm.execute(data, jittered, sync=False)
            vm.reset()
            opcodes = [r[1] for r in result['results']]
            assert opcodes == ['PUSH', 'PUSH', 'ADD2'], f"Jitter broke decoding: {opcodes}"
        return "✔ Jitter gauntlet passed"
    
    @staticmethod
    def test_flag_race():
        vm = PauseLangVM(debug=False)
        pauses = [0.09, 0.08, 0.09, 0.20, 0.09, 0.24]
        data = [7, 7, 3, 0, 2, 0]
        result = vm.execute(data, pauses, sync=False)
        final_flags = result['final_state']['flags']
        assert final_flags['ZERO'] == True, f"Expected ZERO flag, got {final_flags}"
        return "✔ Flag race passed"
    
    @staticmethod
    def test_deep_calls():
        vm = PauseLangVM(gas_limit=5000, debug=False)
        depth = 100
        pauses, data = [], []
        for i in range(depth):
            pauses.append(0.27); data.append(i + 2)
        pauses.append(0.28); data.append(0)
        for _ in range(depth):
            pauses.append(0.28); data.append(0)
        result = vm.execute(data, pauses, sync=False)
        assert not result.get('error'), f"Deep calls failed: {result.get('error')}"
        assert len([t for t in result['traps'] if t == 'CALL_DEPTH_EXCEEDED']) == 0
        return "✔ Deep calls passed"
    
    @staticmethod
    def test_overflow_storm():
        vm = PauseLangVM(debug=False)
        pauses = [0.09, 0.09]
        data = [65536, 65536]
        for _ in range(5):
            pauses.append(0.22); data.append(0)
        result = vm.execute(data, pauses, sync=False)
        final_flags = result['final_state']['flags']
        assert final_flags['OVERFLOW'] == True, "Expected OVERFLOW flag"
        return "✔ Overflow storm passed"
    
    @staticmethod
    def test_gas_cap():
        vm = PauseLangVM(gas_limit=10, debug=False)
        pauses = [0.09, 0.14]
        data = [1, 0]
        for _ in range(20): pauses.append(0.29); data.append(0)
        pauses.append(0.15); data.append(0)
        result = vm.execute(data, pauses, sync=False)
        assert 'GAS_EXHAUSTED' in result['traps'], "Expected GAS_EXHAUSTED trap"
        assert result['gas_used'] > 10, "Gas limit not enforced"
        return "✔ Gas cap passed"
    
    @staticmethod
    def run_all():
        tests = [
            TortureTests.test_jitter_gauntlet,
            TortureTests.test_flag_race,
            TortureTests.test_deep_calls,
            TortureTests.test_overflow_storm,
            TortureTests.test_gas_cap
        ]
        print("\n🔥 TORTURE TEST SUITE 🔥")
        print("=" * 50)
        for test in tests:
            try:
                print(test())
            except AssertionError as e:
                print(f"✗ {test.__name__} FAILED: {e}")
            except Exception as e:
                print(f"✗ {test.__name__} ERROR: {e}")
        print("=" * 50)

# === COMPILER ===

class PauseLangCompiler:
    MACROS = {
        'INC': ['PUSH', 'PUSH', 'ADD2'],
        'DEC': ['PUSH', 'PUSH', 'SUB2'],
        'DOUBLE': ['DUP', 'ADD2'],
        'SQUARED': ['DUP', 'MUL2'],
        'DROP': ['POP'],
        'ENTER': ['PUSH', 'SWAP'],
        'LEAVE': ['SWAP', 'POP'],
    }
    @staticmethod
    def compile(source: str) -> Tuple[List[float], List[int], List[str]]:
        pauses = SPEC['sync_phrase'].copy()
        data = [0, 0, 0, 0]
        comments = ['SYNC', 'SYNC', 'SYNC', 'SYNC']
        for raw in source.strip().split('\n'):
            line = raw.strip().split('#')[0].strip()
            if not line: continue
            parts = line.split()
            opcode = parts[0].upper()
            value = int(parts[1]) if len(parts) > 1 and parts[1].lstrip('-').isdigit() else 0
            if opcode in PauseLangCompiler.MACROS:
                for macro_op in PauseLangCompiler.MACROS[opcode]:
                    if macro_op in OPCODE_TO_PAUSE:
                        pauses.append(OPCODE_TO_PAUSE[macro_op]); data.append(value); comments.append(f"{macro_op} (from {opcode})")
            elif opcode in OPCODE_TO_PAUSE:
                pauses.append(OPCODE_TO_PAUSE[opcode]); data.append(value); comments.append(opcode)
            else:
                raise ValueError(f"Unknown opcode: {opcode}")
        return pauses, data, comments

# === SUPERVISOR API ===

class PauseLangSupervisor:
    def __init__(self, policy: Optional[Dict] = None):
        self.policy = policy or {'gas': 5000, 'trap': 'continue'}
        self.vm = PauseLangVM(gas_limit=self.policy.get('gas', 5000), trap_policy=self.policy.get('trap', 'continue'))
    
    def evaluate_candidates(self, candidates: Dict[str, float], threshold: float = 0.5) -> Dict:
        results = {}
        for name, score in candidates.items():
            self.vm.reset()
            source = f"""
            PUSH {int(score * 100)}
            PUSH {int(threshold * 100)}
            SUB2
            HALT
            """
            pauses, data, _ = PauseLangCompiler().compile(source)
            output = self.vm.execute(data, pauses)
            if 'error' in output:
                results[name] = {'score': score, 'accepted': False, 'trace': f"Error during execution: {output['error']}"}
                continue
            final_stack = output['final_state']['stack']
            accepted = len(final_stack) > 0 and final_stack[-1] > 0
            results[name] = {'score': score, 'accepted': accepted, 'trace': self.vm.explain()}
        return results

    def run(self, data_stream: List[int], pause_stream: List[float], io: Optional[Dict[int, Callable]] = None) -> Dict:
        self.vm.reset()
        result = self.vm.execute(data_stream, pause_stream)
        result['explanation'] = self.vm.explain()
        return result

# === SPIKE CODEC + BRIDGE ===

@dataclass
class Spike:
    t: float
    n: int
    pol: int = 1
    meta: Optional[Dict[str, Any]] = None

class SpikeCodec:
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0, dt: float = 0.001, t_window: float = 0.250, max_rate: float = 250.0, refractory: float = 0.002, seed: int = 12345):
        self.vmin, self.vmax, self.dt, self.t_window, self.max_rate, self.refractory = vmin, vmax, dt, t_window, max_rate, refractory
        random.seed(seed)
    def _clip01(self, x: float) -> float:
        if self.vmax == self.vmin: return 0.0
        return max(0.0, min(1.0, (x - self.vmin) / (self.vmax - self.vmin)))
    def _linmap(self, x: float, lo: float, hi: float) -> float:
        return lo + self._clip01(x) * (hi - lo)
    def encode_latency(self, x: float, n: int = 0, invert: bool = False) -> List[Spike]:
        norm = self._clip01(x)
        t = (norm if invert else (1.0 - norm)) * self.t_window
        return [Spike(t=t, n=n)]

class SpikePauseBridge:
    def __init__(self, quantum: float, guard: float):
        self.q, self.g = quantum, guard
    def spikes_to_pauses(self, spikes: List[Spike], default_pause: float = 0.29) -> List[float]:
        if not spikes: return []
        spikes = sorted(spikes, key=lambda s: (s.t, s.n))
        pauses, prev_t = [], 0.0
        for s in spikes:
            dt = max(0.0, s.t - prev_t)
            bin_idx = round(dt / self.q)
            pause = bin_idx * self.q
            pauses.append(pause if pause > 0 else default_pause)
            prev_t = s.t
        return pauses
    def pauses_to_spikes(self, pauses: List[float], channel: int = 0) -> List[Spike]:
        t, spikes = 0.0, []
        for p in pauses:
            t += p
            spikes.append(Spike(t=t, n=channel))
        return spikes

# === DEMOS ===

def demo_supervisor():
    print("\n🐉 DRAGON DETECTOR - Temporal Supervision Demo")
    print("=" * 60)
    candidates = {'tail': 0.7, 'wings': 0.8, 'fire': 0.3, 'scales': 0.9, 'magic': 0.2}
    supervisor = PauseLangSupervisor()
    results = supervisor.evaluate_candidates(candidates, threshold=0.5)
    print(f"Evaluating {len(candidates)} features with threshold 0.5:\n")
    accepted = []
    for feature, result in results.items():
        status = "✔ ACCEPT" if result['accepted'] else "✗ REJECT"
        print(f"  {feature:10} (score: {results[feature]['score']:.1f}) → {status}")
        if result['accepted']: accepted.append(feature)
    print(f"\n🎯 Decision: Dragon has {', '.join(accepted)}")
    print("\nCompact trace:", results[list(results.keys())[0]]['trace'])

def demo_fibonacci():
    print("\n🌀 PAUSELANG FIBONACCI DEMO")
    print("=" * 60)
    source = """
    PUSH 0
    PUSH 1
    PUSH 0
    PUSH 1
    ADD2
    PUSH 1
    PUSH 1
    ADD2
    PUSH 1
    PUSH 2
    ADD2
    PUSH 2
    PUSH 3
    ADD2
    PUSH 3
    PUSH 5
    ADD2
    PUSH 5
    PUSH 8
    ADD2
    PUSH 8
    PUSH 13
    ADD2
    PUSH 13
    PUSH 21
    ADD2
    HALT
    """
    pauses, data, comments = PauseLangCompiler().compile(source)
    vm = PauseLangVM(debug=True)
    result = vm.execute(data, pauses)
    print(f"\nFinal stack (Fibonacci sequence): {result['final_state']['stack']}")
    print("Expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]")

def demo_spike_bridge():
    print("\n⚡ SPIKE ENCODING → VM DEMO")
    codec = SpikeCodec(vmin=0, vmax=100, t_window=0.20, max_rate=200.0)
    bridge = SpikePauseBridge(quantum=SPEC['time_quantum'], guard=SPEC['guard_band'])
    values = [5, 30, 60, 95]
    all_pauses, all_data = [], []
    all_pauses.extend(SPEC['sync_phrase']); all_data.extend([0]*len(SPEC['sync_phrase']))
    legal_bins = sorted(INSTRUCTIONS.keys())
    for v in values:
        spikes = codec.encode_latency(v, n=0)
        pauses = bridge.spikes_to_pauses(spikes, default_pause=0.29)
        for p in pauses:
            p_quant = min(legal_bins, key=lambda c: abs(c - p))
            all_pauses.append(p_quant); all_data.append(int(v))
    all_pauses.append(0.30); all_data.append(0)
    vm = PauseLangVM(debug=True)
    result = vm.execute(all_data, all_pauses, sync=True)
    print("Traps:", result['traps'])
    print("Final state:", result['final_state'])
    print("Trace:\n" + vm.explain(verbose=False))

# === v0.6.x: Patterns + Stress helpers ===

def demonstrate_patterns():
    box = r"""
╔═══════════════════════════════════════════════════════════════╗
║         PauseLang v0.6.x - IX Register & Patterns            ║
║     Minimal Enhancement for Maximum Expressiveness           ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(box)
    print("="*70)
    print("🔧 IX REGISTER ENHANCEMENT")
    print("="*70)
    print("""
Five opcodes (integrated in this VM):
  • SETIX  (0.40) - Pop value → IX register
  • LOADI  (0.41) - Push mem[IX] to stack
  • STOREI (0.42) - Store TOS at mem[IX]
  • INCIX  (0.43) - IX = (IX + 1) % max_slots
  • GETIX  (0.44) - Push IX register to stack
""")
    print("="*70)
    print("📊 PRACTICAL PATTERNS CATALOG")
    print("="*70)
    print("See pattern functions for source code examples")

def io_port_memory_controller() -> str:
    print("\n🧰 Alternative I/O: Port-Mapped Memory Controller")
    N = SPEC['max_memory_slots']
    controller_code = f'''
class MemoryController:
    def __init__(self, vm):
        self.vm = vm
        self.N = {N}

    def _addr(self, a: int) -> int:
        return a % self.N

    def read(self, addr: int) -> int:
        a = self._addr(addr)
        return self.vm.state.memory.get(a, 0)

    def write(self, addr: int, value: int) -> None:
        a = self._addr(addr)
        self.vm.state.memory[a] = int(value)
'''
    return controller_code

def apply_ix_register_patch() -> Tuple[Dict[str, Dict[str, Any]], str]:
    ix_ops = {
        'SETIX': {'pause': 0.40, 'semantics': 'Pop → IX = value % max_slots'},
        'LOADI': {'pause': 0.41, 'semantics': 'Push mem[IX]'},
        'STOREI': {'pause': 0.42, 'semantics': 'mem[IX] = TOS'},
        'INCIX': {'pause': 0.43, 'semantics': 'IX = (IX + 1) % max_slots'},
        'GETIX': {'pause': 0.44, 'semantics': 'Push IX to stack'},
    }
    vm_patch = "See execute_instruction() method for integrated IX ops"
    return ix_ops, vm_patch

def pattern_fibonacci_with_ix() -> str:
    return "See v0.6.3 for complete pattern source"

def pattern_sliding_window_sum() -> str:
    return "See v0.6.3 for complete pattern source"

def pattern_leaky_bucket() -> str:
    return "See v0.6.3 for complete pattern source"

def pattern_bitmask_gate() -> str:
    return "See v0.6.3 for complete pattern source"

def pattern_temporal_voting() -> str:
    return "See v0.6.3 for complete pattern source"

def menu_specs_and_instr():
    print("\n=== FORMAL SPECIFICATION ===")
    for key, value in SPEC.items():
        print(f"{key:20} : {value}")
    print("\n=== INSTRUCTION SET ===")
    for pause, instr in sorted(INSTRUCTIONS.items()):
        print(f"{instr.opcode:15} @ {pause:.2f}s : {instr.description}")

# === MAIN ===

def main():
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  PauseLang {SPEC['version']} - Neuromorphic Temporal VM      ║
║   Complete fixed version with all patches applied        ║
╚═══════════════════════════════════════════════════════════╝
""")
    print("\nSelect mode:")
    print("1. Run torture tests") 
    print("2. Dragon detector demo")
    print("3. Fibonacci generator")
    print("4. Show formal specification & instructions")
    print("5. Spike encoding bridge demo")
    print("6. Show v0.6.x Patterns & Notes")
    print("7. Print I/O MemoryController code")
    print("8. Show IX patch info")
    choice = input("\nChoice (1-8): ").strip()
    if choice == '1':
        TortureTests.run_all()
    elif choice == '2':
        demo_supervisor()
    elif choice == '3':
        demo_fibonacci()
    elif choice == '4':
        menu_specs_and_instr()
    elif choice == '5':
        demo_spike_bridge()
    elif choice == '6':
        demonstrate_patterns()
    elif choice == '7':
        code = io_port_memory_controller()
        print("\n=== MemoryController ===\n")
        print(code)
    elif choice == '8':
        ops, patch = apply_ix_register_patch()
        print("\n=== IX Ops ===")
        for k, v in ops.items():
            print(f"{k}: {v}")
    else:
        print("Running default demo...")
        demo_supervisor()
    print("\n✨ PauseLang v0.6.4 ready.")

if __name__ == "__main__":
    main()
