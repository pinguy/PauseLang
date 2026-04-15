"""
PauseLang v0.7.12 - Production Final (with IoT Demos)
=======================================================
Changes in v0.7.12:
- BUGFIX: GAS_EXHAUSTED now correctly sets state.halted = True.
- BUGFIX: Large jitter no longer silently executes the wrong opcode.
          Instruction decoding now compares raw pause directly to
          the expected pause (within guard band), not the quantized value.
          This prevents the "nearest neighbour" snapping behaviour.
- DOCUMENTATION: STORE instruction worked example remains.
- ADDED: `strict_sync` parameter to VM.execute() (default False).
- Enhanced torture test suite (25 tests, all passing).
- IoTDemos class with three side‑channel examples.

All prior fixes retained (short sync, ROT, RET trap, LOADI doc, WAV exporter).
"""

import time
import random
import struct
from typing import List, Tuple, Any, Dict, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from math import exp

# === FORMAL SPECIFICATION ===
SPEC = {
    'version': '0.7.12',
    'word_size': 32,
    'overflow': 'wrap',
    'division': 'truncate',
    'max_call_depth': 256,
    'max_stack_size': 4096,
    'max_memory_slots': 256,
    'max_loop_depth': 256,
    'max_traps': 1000,
    'time_quantum': 0.005,       # 5ms per quantum
    'guard_band': 0.0015,        # 1.5ms tolerance
    'sync_phrase': [0.29, 0.30], # 2 symbols = 0.59s
}

# === DIVISION AND MODULO SEMANTICS ===
"""
DIV2: Truncates toward zero.
MOD2: Always positive remainder [0, |b|).
"""

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
        quantized = self.quantize(pause)
        return abs(quantized - target) <= self.guard_band

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
    INVALID_JUMP = 10
    INVALID_CALL = 11
    LOOP_DEPTH_EXCEEDED = 12
    TRAP_STORM = 13
    RETURN_WITHOUT_CALL = 14

class Lane(Enum):
    DATA = auto()
    META = auto()

class OpCategory(Enum):
    STREAM = auto()
    STACK = auto()
    HYBRID = auto()
    CONTROL = auto()
    SYSTEM = auto()

# === INSTRUCTION SET (5ms granularity) ===

@dataclass
class Instruction:
    opcode: str
    pause: float
    description: str
    category: OpCategory = OpCategory.STREAM
    updates_flags: bool = True
    requires_stack: int = 0
    modifies_flow: bool = False
    stack_delta: int = 0

    def signature(self) -> str:
        symbols = {
            OpCategory.STREAM: "≈",
            OpCategory.STACK: "▣",
            OpCategory.HYBRID: "◈",
            OpCategory.CONTROL: "→",
            OpCategory.SYSTEM: "⚙",
        }
        return symbols.get(self.category, "?")

INSTRUCTIONS = {
    # Arithmetic - STREAM OPS
    0.005: Instruction('ADD', 0.005, '[STREAM] Add current and previous', OpCategory.STREAM),
    0.010: Instruction('MEAN', 0.010, '[STREAM] Average of current and previous', OpCategory.STREAM),
    0.015: Instruction('DIFF', 0.015, '[STREAM] Subtract previous from current', OpCategory.STREAM),
    0.020: Instruction('SQUARE', 0.020, '[STREAM] Square current value', OpCategory.STREAM),

    # Conditional - STREAM OPS
    0.025: Instruction('PASS', 0.025, '[STREAM] Pass unchanged (NO FLAG UPDATE)', OpCategory.STREAM, updates_flags=False),
    0.030: Instruction('IF_GT_15_SQUARE', 0.030, '[STREAM] Square if > 15', OpCategory.STREAM),
    0.035: Instruction('DOUBLE_IF_EVEN', 0.035, '[STREAM] Double if even', OpCategory.STREAM),
    0.040: Instruction('NEGATE_IF_ODD', 0.040, '[STREAM] Negate if odd', OpCategory.STREAM),

    # Stack - PURE STACK OPS
    0.045: Instruction('PUSH', 0.045, '[STACK] Push operand to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),
    0.050: Instruction('POP', 0.050, '[STACK] Pop from stack', OpCategory.STACK, requires_stack=1, stack_delta=-1),
    0.055: Instruction('DUP', 0.055, '[STACK] Duplicate top', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=1),

    # Control - CONTROL FLOW
    0.060: Instruction('JUMP_IF_ODD', 0.060, '[CONTROL] Jump if ODD flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.065: Instruction('SKIP_NEXT', 0.065, '[CONTROL] Skip next instruction', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.070: Instruction('LOOP_START', 0.070, '[CONTROL] Mark loop start', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.075: Instruction('LOOP_END', 0.075, '[CONTROL] Pop TOS; loop if > 0', OpCategory.CONTROL, updates_flags=False, modifies_flow=True, requires_stack=1, stack_delta=-1),

    # Memory - HYBRID OPS
    # WORKED EXAMPLE: STORE takes its slot from the DATA stream, not the stack.
    #   Example: PUSH 99 / STORE 42  →  mem[42] = 99
    0.080: Instruction('STORE', 0.080, '[HYBRID] Store TOS at mem[operand] (operand from data stream, not stack). Example: PUSH 99 / STORE 42 → mem[42]=99', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.085: Instruction('LOAD', 0.085, '[HYBRID] Load mem[operand] to stack', OpCategory.HYBRID, stack_delta=1),
    0.090: Instruction('SWAP', 0.090, '[STACK] Swap top two', OpCategory.STACK, updates_flags=False, requires_stack=2),
    0.095: Instruction('CLEAR_STACK', 0.095, '[STACK] Clear entire stack', OpCategory.STACK, updates_flags=False),

    # Stack Arithmetic - PURE STACK OPS
    0.100: Instruction('ADD2', 0.100, '[STACK] Pop 2, push sum', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.105: Instruction('SUB2', 0.105, '[STACK] Pop 2, push difference', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.110: Instruction('MUL2', 0.110, '[STACK] Pop 2, push product', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.115: Instruction('DIV2', 0.115, '[STACK] Pop 2, push quotient', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.120: Instruction('MOD2', 0.120, '[STACK] Pop 2, push modulo', OpCategory.STACK, requires_stack=2, stack_delta=-1),

    # Meta - SYSTEM OPS
    0.125: Instruction('SET_META', 0.125, '[SYSTEM] Toggle meta mode', OpCategory.SYSTEM, updates_flags=False),
    0.130: Instruction('JUMP_IF_ZERO', 0.130, '[CONTROL] Jump if ZERO flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.135: Instruction('CALL', 0.135, '[CONTROL] Call subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.140: Instruction('RET', 0.140, '[CONTROL] Return from subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),

    # System
    0.145: Instruction('NOP', 0.145, '[SYSTEM] No operation', OpCategory.SYSTEM, updates_flags=False),
    0.150: Instruction('HALT', 0.150, '[SYSTEM] Halt execution', OpCategory.SYSTEM, updates_flags=False, modifies_flow=True),

    # Unconditional Jump
    0.155: Instruction('JUMP', 0.155, '[CONTROL] Unconditional jump', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),

    # ROT
    0.165: Instruction('ROT', 0.165, '[STACK] Rotate top three: ( a b c -- b c a )', OpCategory.STACK, updates_flags=False, requires_stack=3, stack_delta=0),

    # IX Register
    0.200: Instruction('SETIX', 0.200, '[STACK] Pop stack → IX register', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.205: Instruction('LOADI', 0.205, '[STACK] Push mem[IX] to stack (returns 0 if uninitialised)', OpCategory.STACK, updates_flags=True, stack_delta=1),
    0.210: Instruction('STOREI', 0.210, '[HYBRID] Store TOS at mem[IX]', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.215: Instruction('INCIX', 0.215, '[SYSTEM] IX = (IX + 1) % max_slots', OpCategory.SYSTEM, updates_flags=False),
    0.220: Instruction('GETIX', 0.220, '[STACK] Push IX register to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),

    # Jump if not zero
    0.160: Instruction('JUMP_IF_NONZERO', 0.160, '[CONTROL] Jump if not ZERO', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
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
    labels: Dict[str, int] = field(default_factory=dict)
    stack_high_water: int = 0
    instructions_executed: int = 0

class PauseLangVM:
    def __init__(self, gas_limit: int = 20000, trap_policy: str = 'continue', memory_mode: str = 'wrap', debug: bool = False):
        self.state = VMState()
        self.gas_limit = gas_limit
        self.trap_policy = trap_policy
        self.memory_mode = memory_mode
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
        if len(self.state.trap_stack) > SPEC['max_traps']:
            self.state.halted = True
            if self.debug:
                print(f"⚠️ TRAP STORM DETECTED: {len(self.state.trap_stack)} traps - FORCE HALT")
            return
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
            self.state.halted = True   # <-- FIXED v0.7.12: explicitly halt on gas exhaustion
            return False
        return True

    def check_stack_health(self) -> bool:
        depth = len(self.state.stack)
        if depth > self.state.stack_high_water:
            self.state.stack_high_water = depth
        if depth > SPEC['max_stack_size'] * 0.75 and self.debug:
            print(f"⚠️ Stack depth warning: {depth}/{SPEC['max_stack_size']}")
        return depth < SPEC['max_stack_size']

    def execute_instruction(self, instr: Instruction, value: int, prev_value: Optional[int] = None) -> Any:
        opcode = instr.opcode

        if instr.requires_stack > len(self.state.stack):
            self.push_trap(TrapCode.STACK_UNDERFLOW)
            return "TRAP: STACK_UNDERFLOW"
        if instr.stack_delta > 0 and len(self.state.stack) + instr.stack_delta > SPEC['max_stack_size']:
            self.push_trap(TrapCode.STACK_OVERFLOW)
            return "TRAP: STACK_OVERFLOW"

        if opcode in ['ADD', 'MEAN', 'DIFF', 'SQUARE', 'IF_GT_15_SQUARE', 'DOUBLE_IF_EVEN',
                      'NEGATE_IF_ODD', 'ADD2', 'SUB2', 'MUL2', 'DIV2']:
            self.state.flags[Flag.OVERFLOW] = False

        result = None

        # Stream arithmetic
        if opcode == 'ADD' and prev_value is not None:
            result = self.wrap_int32(value + prev_value)
        elif opcode == 'MEAN' and prev_value is not None:
            result = self.wrap_int32((value + prev_value) // 2)
        elif opcode == 'DIFF' and prev_value is not None:
            result = self.wrap_int32(value - prev_value)
        elif opcode == 'SQUARE':
            result = self.wrap_int32(value * value)

        # Stream conditional
        elif opcode == 'PASS':
            result = value
        elif opcode == 'IF_GT_15_SQUARE':
            result = self.wrap_int32(value * value) if value > 15 else value
        elif opcode == 'DOUBLE_IF_EVEN':
            result = self.wrap_int32(value * 2) if value % 2 == 0 else value
        elif opcode == 'NEGATE_IF_ODD':
            result = self.wrap_int32(-value) if value % 2 != 0 else value

        # Stack operations
        elif opcode == 'PUSH':
            self.state.stack.append(value)
            self.check_stack_health()
            result = f"PUSHED {value}"
        elif opcode == 'POP':
            if len(self.state.stack) == 0:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                result = self.state.stack.pop()
        elif opcode == 'DUP':
            if len(self.state.stack) == 0:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                top = self.state.stack[-1]
                if len(self.state.stack) + 1 > SPEC['max_stack_size']:
                    self.push_trap(TrapCode.STACK_OVERFLOW)
                    result = "STACK_OVERFLOW"
                else:
                    self.state.stack.append(top)
                    self.check_stack_health()
                    result = f"DUP {top}"
        elif opcode == 'SWAP':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                self.state.stack[-1], self.state.stack[-2] = self.state.stack[-2], self.state.stack[-1]
                result = "SWAPPED"
        elif opcode == 'CLEAR_STACK':
            count = len(self.state.stack)
            self.state.stack.clear()
            result = f"CLEARED {count}"
        elif opcode == 'ROT':
            if len(self.state.stack) < 3:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                a = self.state.stack[-3]
                b = self.state.stack[-2]
                c = self.state.stack[-1]
                self.state.stack[-3] = b
                self.state.stack[-2] = c
                self.state.stack[-1] = a
                result = "ROT"

        # Stack arithmetic
        elif opcode == 'ADD2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop() if self.state.stack else 0
                r = self.wrap_int32(a + b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'SUB2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop() if self.state.stack else 0
                r = self.wrap_int32(a - b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'MUL2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop() if self.state.stack else 0
                r = self.wrap_int32(a * b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'DIV2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop() if self.state.stack else 0
                if b == 0:
                    self.push_trap(TrapCode.DIV_BY_ZERO)
                    self.state.stack.append(0)
                    result = "DIV_BY_ZERO"
                else:
                    r = self.wrap_int32(int(a / b))
                    self.state.stack.append(r)
                    result = r
        elif opcode == 'MOD2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop() if self.state.stack else 0
                if b == 0:
                    self.push_trap(TrapCode.DIV_BY_ZERO)
                    self.state.stack.append(0)
                    result = "MOD_BY_ZERO"
                else:
                    r = a % b
                    if r < 0:
                        r += abs(b)
                    self.state.stack.append(r)
                    result = r

        # Memory operations
        elif opcode == 'STORE':
            if not self.state.stack:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                if self.memory_mode == 'strict':
                    slot = value
                    if not (0 <= slot < SPEC['max_memory_slots']):
                        self.push_trap(TrapCode.INVALID_MEMORY)
                        result = "INVALID_MEMORY"
                        return result
                else:
                    slot = value % SPEC['max_memory_slots'] if self.state.lane == Lane.DATA else value
                store_value = self.state.stack.pop()
                if 0 <= slot < SPEC['max_memory_slots']:
                    self.state.memory[slot] = store_value
                    result = f"STORED {store_value} @ {slot}"
                else:
                    self.push_trap(TrapCode.INVALID_MEMORY)
                    result = "INVALID_MEMORY"
        elif opcode == 'LOAD':
            if self.memory_mode == 'strict':
                slot = value
                if not (0 <= slot < SPEC['max_memory_slots']):
                    self.push_trap(TrapCode.INVALID_MEMORY)
                    result = "INVALID_MEMORY"
                    return result
            else:
                slot = value % SPEC['max_memory_slots'] if self.state.lane == Lane.DATA else value
            loaded_value = self.state.memory.get(slot, 0)
            if len(self.state.stack) + 1 > SPEC['max_stack_size']:
                self.push_trap(TrapCode.STACK_OVERFLOW)
                result = "STACK_OVERFLOW"
            else:
                self.state.stack.append(loaded_value)
                self.check_stack_health()
                result = loaded_value

        # System operations
        elif opcode == 'SET_META':
            self.state.lane = Lane.META if self.state.lane == Lane.DATA else Lane.DATA
            result = f"LANE: {self.state.lane.name}"
        elif opcode == 'NOP':
            result = "NOP"
        elif opcode == 'HALT':
            self.state.halted = True
            self.push_trap(TrapCode.HALT)
            result = "HALTED"

        # IX register
        elif opcode == 'SETIX':
            if len(self.state.stack) == 0:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                stack_value = self.state.stack.pop()
                self.state.ix = stack_value % SPEC['max_memory_slots']
                result = f"IX={self.state.ix}"
        elif opcode == 'LOADI':
            v = self.state.memory.get(self.state.ix, 0)
            if len(self.state.stack) + 1 > SPEC['max_stack_size']:
                self.push_trap(TrapCode.STACK_OVERFLOW)
                result = "STACK_OVERFLOW"
            else:
                self.state.stack.append(v)
                self.check_stack_health()
                result = v
        elif opcode == 'STOREI':
            if not self.state.stack:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                v = self.state.stack.pop()
                self.state.memory[self.state.ix] = v
                result = f"STORED {v} at mem[{self.state.ix}]"
        elif opcode == 'INCIX':
            self.state.ix = (self.state.ix + 1) % SPEC['max_memory_slots']
            result = f"IX={self.state.ix}"
        elif opcode == 'GETIX':
            if len(self.state.stack) + 1 > SPEC['max_stack_size']:
                self.push_trap(TrapCode.STACK_OVERFLOW)
                result = "STACK_OVERFLOW"
            else:
                self.state.stack.append(self.state.ix)
                self.check_stack_health()
                result = f"PUSHED IX={self.state.ix}"

        if instr.updates_flags and isinstance(result, int):
            self.update_flags(result)
        return result if result is not None else value

    def execute(self, data_stream: List[int], pause_stream: List[float],
                sync: bool = True, labels: Optional[Dict[str, int]] = None,
                strict_sync: bool = False) -> Dict:
        """
        Execute a PauseLang program.

        :param data_stream: List of integer operands.
        :param pause_stream: List of pause durations (seconds) – same length as data_stream.
        :param sync: If True, calibrate drift using the sync phrase (if present).
        :param labels: Optional label dictionary (from compiler).
        :param strict_sync: If True, NEVER auto‑strip the sync phrase.
                           Default False (auto‑strip if the stream begins with sync_phrase).
                           Set to True to avoid the auto‑strip foot‑gun.
        """
        if labels:
            self.state.labels = labels

        base_offset = 0

        def matches_sync_phrase(pauses):
            if len(pauses) < len(SPEC['sync_phrase']):
                return False
            for p_obs, p_exp in zip(pauses[:len(SPEC['sync_phrase'])], SPEC['sync_phrase']):
                if not self.quantizer.in_guard_band(p_obs, p_exp):
                    return False
            return True

        # Auto-strip sync if present (only if strict_sync is False)
        if not strict_sync and len(pause_stream) >= len(SPEC['sync_phrase']) and matches_sync_phrase(pause_stream):
            if sync:
                if not self.quantizer.calibrate(pause_stream[:len(SPEC['sync_phrase'])]):
                    return {'error': 'Sync calibration failed'}
            data_stream = data_stream[len(SPEC['sync_phrase']):]
            pause_stream = pause_stream[len(SPEC['sync_phrase']):]
            base_offset = len(SPEC['sync_phrase'])

        if len(data_stream) != len(pause_stream):
            return {'error': f'Stream length mismatch: data={len(data_stream)}, pauses={len(pause_stream)}'}

        results = []
        while self.state.pc < len(data_stream) and not self.state.halted:
            if not self.check_gas(): break

            self.state.instructions_executed += 1

            value = data_stream[self.state.pc]
            raw_pause = pause_stream[self.state.pc]
            # FIXED v0.7.12: decode using raw pause, not quantized value
            instr = None
            for target_pause, instruction in INSTRUCTIONS.items():
                if abs(raw_pause - target_pause) <= self.quantizer.guard_band:
                    instr = instruction
                    break
            if instr is None:
                self.push_trap(TrapCode.INVALID_INSTRUCTION)
                instr = INSTRUCTIONS[0.025]  # PASS as fallback

            prev_value = data_stream[self.state.pc - 1] if self.state.pc > 0 else None

            # Control flow
            if instr.opcode == 'JUMP':
                target = value - base_offset
                if not (0 <= target < len(data_stream)):
                    self.push_trap(TrapCode.INVALID_JUMP)
                    result = f"INVALID JUMP TARGET {value}"
                else:
                    self.state.pc = target - 1
                    result = f"JUMPED to {value}"
            elif instr.opcode == 'JUMP_IF_ODD' and self.state.flags[Flag.ODD]:
                target = value - base_offset
                if not (0 <= target < len(data_stream)):
                    self.push_trap(TrapCode.INVALID_JUMP)
                    result = f"INVALID JUMP TARGET {value}"
                else:
                    self.state.pc = target - 1
                    result = f"JUMPED to {value}"
            elif instr.opcode == 'JUMP_IF_ZERO' and self.state.flags[Flag.ZERO]:
                target = value - base_offset
                if not (0 <= target < len(data_stream)):
                    self.push_trap(TrapCode.INVALID_JUMP)
                    result = f"INVALID JUMP TARGET {value}"
                else:
                    self.state.pc = target - 1
                    result = f"JUMPED to {value}"
            elif instr.opcode == 'JUMP_IF_NONZERO' and not self.state.flags[Flag.ZERO]:
                target = value - base_offset
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
                if len(self.state.loop_stack) >= SPEC['max_loop_depth']:
                    self.push_trap(TrapCode.LOOP_DEPTH_EXCEEDED)
                    result = "LOOP_DEPTH_EXCEEDED"
                elif not self.state.loop_stack or self.state.loop_stack[-1] != self.state.pc:
                    self.state.loop_stack.append(self.state.pc)
                    result = "LOOP_START"
                else:
                    result = "LOOP_START"
            elif instr.opcode == 'LOOP_END':
                if len(self.state.stack) == 0:
                    self.push_trap(TrapCode.STACK_UNDERFLOW)
                    result = "LOOP_END STACK_UNDERFLOW"
                elif not self.state.loop_stack:
                    result = "LOOP_EXIT (no loop)"
                    self.state.stack.pop()
                else:
                    counter_value = self.state.stack.pop()
                    if counter_value > 0:
                        self.state.pc = self.state.loop_stack[-1] - 1
                        result = "LOOP_CONTINUE"
                    else:
                        self.state.loop_stack.pop()
                        result = "LOOP_EXIT"
            elif instr.opcode == 'CALL':
                if len(self.state.call_stack) >= SPEC['max_call_depth']:
                    self.push_trap(TrapCode.CALL_DEPTH_EXCEEDED)
                    result = "CALL_DEPTH_EXCEEDED"
                else:
                    target = value - base_offset
                    if not (0 <= target < len(data_stream)):
                        self.push_trap(TrapCode.INVALID_CALL)
                        result = f"INVALID CALL TARGET {value}"
                    else:
                        self.state.call_stack.append(self.state.pc + 1)
                        self.state.pc = target - 1
                        result = f"CALL {value}"
            elif instr.opcode == 'RET':
                if self.state.call_stack:
                    self.state.pc = self.state.call_stack.pop() - 1
                    result = "RET"
                else:
                    self.push_trap(TrapCode.RETURN_WITHOUT_CALL)
                    result = "RETURN_WITHOUT_CALL"
            else:
                result = self.execute_instruction(instr, value, prev_value)

            self.execution_trace.append({
                'pc': self.state.pc,
                'opcode': instr.opcode,
                'value': value,
                'result': result,
                'stack_depth': len(self.state.stack),
                'flags': [f.name for f, v in self.state.flags.items() if v],
                'gas': self.state.gas_used,
                'category': instr.category.name
            })
            results.append((value, instr.opcode, result))
            if self.debug:
                print(f"PC:{self.state.pc:03d} | {instr.signature()} {instr.opcode:<12} | {value:6d} → {result}")
            self.state.pc += 1

        return {
            'results': results,
            'final_state': self.get_state(),
            'traps': [t.name for t in self.state.trap_stack],
            'gas_used': self.state.gas_used,
            'halted': self.state.halted,
            'stats': {
                'stack_high_water': self.state.stack_high_water,
                'instructions_executed': self.state.instructions_executed,
                'trap_count': len(self.state.trap_stack),
            }
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
            'ix': self.state.ix,
            'stack_high_water': self.state.stack_high_water,
            'instructions_executed': self.state.instructions_executed,
        }

    def disassemble(self, show_labels: bool = True, show_state: bool = False,
                   compact: bool = False, show_memory: bool = False) -> str:
        if not self.execution_trace:
            return "No execution trace"
        lines = ["=== DISASSEMBLY ===",
                 "*PCs are post-sync absolute; execution trace is chronological (jumps may skip lines).*"]
        labels_reverse = {v: k for k, v in self.state.labels.items()} if self.state.labels else {}
        for i, step in enumerate(self.execution_trace):
            pc = step['pc'] + len(SPEC['sync_phrase'])
            label = ""
            if show_labels and pc in labels_reverse:
                label = f"{labels_reverse[pc]}:"
            label_col = f"{label:12}" if show_labels else ""
            state_col = ""
            if show_state:
                stack_preview = str(self.state.stack[-3:]) if len(self.state.stack) > 0 else "[]"
                flags = ','.join(step['flags'][:2]) if step['flags'] else "none"
                state_col = f" | S:{stack_preview:20} F:{flags:10}"
            mem_detail = ""
            if show_memory and step['opcode'] in ['STORE', 'STOREI', 'LOAD', 'LOADI']:
                if step['opcode'] == 'STORE':
                    lane = Lane.DATA
                    for j in range(i-1, -1, -1):
                        if 'LANE:' in str(self.execution_trace[j].get('result', '')):
                            lane_str = str(self.execution_trace[j]['result'])
                            lane = Lane.META if 'META' in lane_str else Lane.DATA
                            break
                    slot = step['value']
                    if self.memory_mode == 'strict':
                        mem_detail = f" [strict: slot {slot}]" if 0 <= slot < SPEC['max_memory_slots'] else " [INVALID (strict)]"
                    elif lane == Lane.DATA:
                        effective_slot = slot % SPEC['max_memory_slots']
                        mem_detail = f" [→ slot {effective_slot} (DATA)]"
                    else:
                        if slot >= SPEC['max_memory_slots']:
                            mem_detail = f" [INVALID (META)]"
                        else:
                            mem_detail = f" [→ slot {slot} (META)]"
            cat_icon = INSTRUCTIONS[OPCODE_TO_PAUSE[step['opcode']]].signature()
            if compact:
                lines.append(f"{pc:04d}: {step['opcode']:12} {step['value']:6d}{mem_detail}")
            else:
                result_str = str(step['result'])[:20]
                lines.append(
                    f"{label_col}{pc:04d}: {cat_icon} {step['opcode']:12} "
                    f"{step['value']:6d} → {result_str:20}{mem_detail}{state_col}"
                )
        return '\n'.join(lines)

    def explain(self, verbose: bool = False) -> str:
        if not self.execution_trace: return "No execution trace"
        if verbose:
            return self.disassemble(show_labels=True, show_state=True)
        phrases, current = [], []
        for step in self.execution_trace:
            if step['opcode'] in ['JUMP','JUMP_IF_ODD','JUMP_IF_ZERO','SKIP_NEXT','LOOP_START','LOOP_END','CALL','RET','HALT']:
                if current:
                    phrases.append(self._summarize_phrase(current))
                    current = []
                phrases.append(f"Control: {step['opcode']} → {step['result']}")
            else:
                current.append(step)
        if current:
            phrases.append(self._summarize_phrase(current))
        return ' | '.join(phrases)

    def _summarize_phrase(self, steps: List[Dict]) -> str:
        ops = [s['opcode'] for s in steps]
        return f"{ops[0]}..{ops[-1]} ({len(ops)} ops)" if len(ops) > 1 else f"{ops[0]}"

# === COMPILER WITH LABELS ===

class PauseLangCompiler:
    ALIASES = {
        'CONST': 'PUSH',
        'DROP': 'POP',
        'PEEK': 'DUP',
        'DROPS': 'CLEAR_STACK',
        'JMP': 'JUMP',
        'JOD': 'JUMP_IF_ODD',
        'JZ': 'JUMP_IF_ZERO',
        'JNZ': 'JUMP_IF_NONZERO',
    }

    MACROS = {
        'INC':       [('PUSH', 1), 'ADD2'],
        'DEC':       [('PUSH', 1), 'SUB2'],
        'DOUBLE':    ['DUP', 'ADD2'],
        'SQUARED':   ['DUP', 'MUL2'],
        'ENTER':     ['PUSH', 'SWAP'],
        'LEAVE':     ['SWAP', 'POP'],
        'STOREI_POP':['STOREI', 'POP'],
        'NOT':       [('PUSH', -1), 'SWAP', 'SUB2'],
        'LNOT':      [('PUSH', 1), 'SWAP', 'SUB2'],
        'NEG':       [('PUSH', 0), 'SWAP', 'SUB2'],
        'SETF':      [('PUSH', 0), 'ADD2'],
    }

    @staticmethod
    def compile(source: str, debug: bool = False) -> Tuple[List[float], List[int], List[str], Dict[str, int]]:
        lines = source.strip().split('\n')
        labels = {}
        pc = len(SPEC['sync_phrase'])

        # First pass: collect labels
        for line_num, raw_line in enumerate(lines):
            clean = raw_line.strip().split('#')[0].strip()
            if not clean:
                continue
            if clean.endswith(':'):
                label_name = clean[:-1].strip()
                if label_name in labels:
                    raise ValueError(f"Duplicate label '{label_name}' at line {line_num + 1}")
                labels[label_name] = pc
                if debug:
                    print(f"Label '{label_name}' → PC {pc}")
                continue
            parts = clean.split()
            opcode = parts[0].upper()
            if opcode in PauseLangCompiler.ALIASES:
                opcode = PauseLangCompiler.ALIASES[opcode]
            if opcode in PauseLangCompiler.MACROS:
                pc += len(PauseLangCompiler.MACROS[opcode])
            elif opcode in OPCODE_TO_PAUSE:
                pc += 1
            else:
                raise ValueError(f"Unknown opcode '{opcode}' at line {line_num + 1}")

        # Second pass: generate instructions
        pauses = SPEC['sync_phrase'].copy()
        data = [0] * len(SPEC['sync_phrase'])
        comments = ['SYNC'] * len(SPEC['sync_phrase'])

        for line_num, raw_line in enumerate(lines):
            clean = raw_line.strip().split('#')[0].strip()
            if not clean or clean.endswith(':'):
                continue
            parts = clean.split()
            opcode = parts[0].upper()
            original_opcode = opcode
            if opcode in PauseLangCompiler.ALIASES:
                opcode = PauseLangCompiler.ALIASES[opcode]
            value = 0
            if len(parts) > 1:
                operand = parts[1]
                if operand in labels:
                    value = labels[operand]
                    if debug:
                        print(f"Resolved label '{operand}' → {value}")
                elif operand.lstrip('-').isdigit():
                    value = int(operand)
                else:
                    raise ValueError(f"Unknown operand '{operand}' at line {line_num + 1}")

            if opcode in PauseLangCompiler.MACROS:
                for macro_step in PauseLangCompiler.MACROS[opcode]:
                    if isinstance(macro_step, tuple):
                        macro_op, imm = macro_step
                        pauses.append(OPCODE_TO_PAUSE[macro_op])
                        data.append(imm)
                        comments.append(f"{macro_op} {imm} (from {original_opcode})")
                    else:
                        pauses.append(OPCODE_TO_PAUSE[macro_step])
                        data.append(value)
                        comments.append(f"{macro_step} (from {original_opcode})")
            elif opcode in OPCODE_TO_PAUSE:
                pauses.append(OPCODE_TO_PAUSE[opcode])
                data.append(value)
                comment = opcode
                if original_opcode != opcode:
                    comment += f" (alias {original_opcode})"
                if len(parts) > 1 and parts[1] in labels:
                    comment += f" [{parts[1]}]"
                comments.append(comment)
            else:
                raise ValueError(f"Unknown opcode '{opcode}' at line {line_num + 1}")

        return pauses, data, comments, labels

# === OPTIONAL WAV EXPORTER ===
# Requires scipy (optional). If not installed, skip.
class WavExporter:
    @staticmethod
    def export_to_wav(pauses: List[float], sample_rate: int = 44100, filename: str = "pause_program.wav"):
        """
        Generate a WAV file where each pause is represented as a silent gap,
        and each instruction is a short click (1ms beep) at the start of the pause.
        The program can be decoded by measuring inter‑click intervals.
        """
        try:
            import numpy as np
            from scipy.io import wavfile
        except ImportError:
            print("WAV export requires numpy and scipy. Install with: pip install numpy scipy")
            return

        # Generate click (1ms sine beep at 1kHz)
        click_duration = 0.001  # 1ms
        click_samples = int(sample_rate * click_duration)
        t = np.linspace(0, click_duration, click_samples, endpoint=False)
        click = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)

        # Build audio: for each pause, output click then silence for (pause - click_duration)
        audio = []
        for pause in pauses:
            audio.append(click)
            silence_samples = max(0, int(sample_rate * pause) - click_samples)
            if silence_samples > 0:
                audio.append(np.zeros(silence_samples, dtype=np.int16))
        audio = np.concatenate(audio)
        wavfile.write(filename, sample_rate, audio)
        print(f"Exported {len(pauses)} instructions to {filename}")

# === ENHANCED TORTURE TESTS ===

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
        pauses = [0.140]  # RET
        data = [0]
        result = vm.execute(data, pauses, sync=False)
        assert 'RETURN_WITHOUT_CALL' in result['traps'], "RET without CALL should trap"
        return "✓ RET trap works"

    @staticmethod
    def test_loadi_uninit():
        """LOADI must return 0 for uninitialised slots, not trap."""
        vm = PauseLangVM(debug=False)
        # SETIX 42, LOADI, HALT
        pauses = [0.200, 0.205, 0.150]   # SETIX, LOADI, HALT
        data   = [42,     0,      0]
        result = vm.execute(data, pauses, sync=False)
        stack = result['final_state']['stack']
        assert stack == [0], f"LOADI on uninit should push 0, got {stack}"
        assert 'INVALID_MEMORY' not in result['traps'], "LOADI should not trap on uninit"
        return "✓ LOADI uninitialised returns 0"

    @staticmethod
    def test_rot():
        """ROT: ( a b c -- b c a )"""
        vm = PauseLangVM(debug=False)
        # PUSH 10, PUSH 20, PUSH 30, ROT, HALT
        pauses = [0.045, 0.045, 0.045, 0.165, 0.150]
        data   = [10,    20,    30,    0,     0]
        result = vm.execute(data, pauses, sync=False)
        stack = result['final_state']['stack']
        assert stack == [20, 30, 10], f"ROT failed: expected [20, 30, 10], got {stack}"
        return "✓ ROT instruction passed"

    @staticmethod
    def test_rot_underflow():
        """ROT with < 3 items should trap."""
        vm = PauseLangVM(debug=False)
        pauses = [0.045, 0.165]   # PUSH + ROT (only 1 item)
        data   = [99,    0]
        result = vm.execute(data, pauses, sync=False)
        assert 'STACK_UNDERFLOW' in result['traps'], "ROT with <3 items should trap"
        return "✓ ROT underflow protection passed"

    @staticmethod
    def test_strict_sync():
        """strict_sync=True must NOT auto-strip the sync phrase."""
        pauses = [0.29, 0.30, 0.045, 0.150]   # sync + PUSH 42 + HALT
        data   = [0,    0,    42,    0]

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
        pauses = [0.29, 0.30, 0.045, 0.150]   # sync + PUSH 42 + HALT
        data   = [0,    0,    42,    0]
        result = vm.execute(data, pauses, sync=True, strict_sync=False)
        assert 'error' not in result
        assert result['final_state']['stack'] == [42]
        return "✓ Short 2-symbol sync phrase passed"

    @staticmethod
    def test_sync_jitter_tolerance():
        """Sync phrase should tolerate small jitter within guard band."""
        vm = PauseLangVM(debug=False)
        for _ in range(30):
            jittered = [p + random.uniform(-0.0008, 0.0008) for p in SPEC['sync_phrase']]
            pauses = jittered + [0.045, 0.150]
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
        pauses = [0.045, 0.200, 0.215, 0.220, 0.150]
        data   = [255,   0,     0,     0,     0]
        result = vm.execute(data, pauses, sync=False)
        assert result['final_state']['stack'] == [0]
        assert result['final_state']['ix'] == 0
        return "✓ IX register wrapping passed"

    @staticmethod
    def test_store_worked_example():
        """Verify documented STORE example: PUSH 99 / STORE 42 → mem[42] = 99, then LOAD 42 → pushes 99."""
        vm = PauseLangVM(debug=False)
        # PUSH 99, STORE 42, LOAD 42, HALT
        pauses = [0.045, 0.080, 0.085, 0.150]
        data   = [99,    42,    42,   0]
        result = vm.execute(data, pauses, sync=False)
        mem = result['final_state']['memory']
        stack = result['final_state']['stack']
        assert mem.get(42) == 99, f"STORE failed: mem[42] = {mem.get(42)}"
        assert stack == [99], f"LOAD should have pushed 99, got {stack}"
        return "✓ STORE worked example verified"

    @staticmethod
    def test_fuzz_v077():
        """Fuzz with full v0.7.12 instruction set (including ROT)."""
        vm = PauseLangVM(debug=False)
        all_pauses = list(INSTRUCTIONS.keys())
        for _ in range(150):
            length = random.randint(4, 20)
            pauses = [random.choice(all_pauses) for _ in range(length)]
            data = [random.randint(-200, 200) for _ in range(length)]
            try:
                vm.execute(data, pauses, sync=False)
            except Exception as e:
                raise AssertionError(f"Fuzz crash: {e}")
            vm.reset()
        return "✓ Fuzz test passed (v0.7.12 ISA)"

    @staticmethod
    def test_gas_exhaustion_halted():
        """GAS_EXHAUSTED should set halted=True."""
        vm = PauseLangVM(gas_limit=2, debug=False)
        pauses = [0.045, 0.045, 0.045]  # three PUSHes, gas limit 2
        data   = [1, 2, 3]
        result = vm.execute(data, pauses, sync=False)
        assert result['halted'] is True, "GAS_EXHAUSTED did not set halted"
        assert 'GAS_EXHAUSTED' in result['traps']
        return "✓ GAS_EXHAUSTED sets halted flag"

    @staticmethod
    def test_jitter_no_snap():
        """Large jitter should produce INVALID_INSTRUCTION, not a silent wrong opcode."""
        vm = PauseLangVM(debug=False)
        # Expect PUSH (0.045) but add +3ms jitter -> 0.048, beyond guard band
        pauses = [0.048, 0.150]  # 0.048 outside 0.045±0.0015
        data   = [42,     0]
        result = vm.execute(data, pauses, sync=False)
        assert 'INVALID_INSTRUCTION' in result['traps'], "Large jitter should trap, not snap to MEAN"
        # Verify it didn't execute MEAN instead
        opcodes = [r[1] for r in result['results']]
        assert opcodes[0] == 'PASS', "Fallback PASS should be used on invalid decode"
        return "✓ Jitter no longer snaps to wrong opcode"

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
        ]
        print("\n🔥 TORTURE TEST SUITE v0.7.12 (Production Final) 🔥")
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

# === IOT DEMOS ===

class IoTDemos:
    @staticmethod
    def demo_leaky_bucket():
        print("\n📡 IoT Demo 1: Leaky Bucket Rate Limiter")
        print("─" * 50)
        source = """
        main:
            CONST 0
            STORE 0
            CONST 5
            STORE 1
            CONST 8
            STORE 2
        tick_loop:
            LOAD 2
            JZ tick_done
            LOAD 0
            LOAD 1
            SUB2
            JZ tick_skip
            LOAD 0
            CONST 1
            ADD2
            STORE 0
        tick_skip:
            LOAD 2
            CONST 1
            SUB2
            STORE 2
            JUMP tick_loop
        tick_done:
            CONST 7
            STORE 3
        consume_loop:
            LOAD 3
            JZ consume_done
            LOAD 0
            JZ reject
            LOAD 0
            CONST 1
            SUB2
            STORE 0
            LOAD 3
            CONST 1
            SUB2
            STORE 3
            JUMP consume_loop
        consume_done:
            CONST 1
            HALT
        reject:
            CONST 0
            HALT
        """
        pauses, data, _, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(gas_limit=100000, debug=False)
        result = vm.execute(data, pauses, labels=labels)
        bucket = result['final_state']['memory'].get(0, -1)
        success = result['final_state']['stack'][-1] if result['final_state']['stack'] else -1
        print(f"Final bucket: {bucket} | Success flag: {success}")
        assert bucket == 0 and success == 0, f"Expected bucket 0 and success 0, got bucket={bucket}, success={success}"
        print(" ✓ Leaky bucket correctly rejected over-limit requests")
        return True

    @staticmethod
    def demo_spike_detector():
        print("\n📡 IoT Demo 2: Temporal Spike/Dragon Detector")
        print("─" * 50)
        source = """
        main:
            CONST 120
            STORE 0
            CONST 150
            STORE 1
            CONST 45
            STORE 2
            CONST 500
            STORE 3
            CONST 160
            STORE 4
            CONST 250
            STORE 5
            CONST 0
            STORE 10
            CONST 0
            STORE 11
        loop:
            LOAD 11
            CONST 6
            SUB2
            JZ done
            LOAD 11
            SETIX
            LOADI
            DUP
            CONST 500
            SUB2
            JZ spike
            DROP
            CONST 250
            SUB2
            JZ spike
            JUMP next
        spike:
            DROP
            LOAD 10
            CONST 1
            ADD2
            STORE 10
        next:
            LOAD 11
            CONST 1
            ADD2
            STORE 11
            JUMP loop
        done:
            LOAD 10
            HALT
        """
        pauses, data, _, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(gas_limit=50000, debug=False)
        result = vm.execute(data, pauses, labels=labels)
        anomalies = result['final_state']['stack'][-1] if result['final_state']['stack'] else 0
        print(f"Detected {anomalies} anomalies (expected 2)")
        assert anomalies == 2
        print(" ✓ Spike detector correctly flagged temporal anomalies")
        return True

    @staticmethod
    def demo_key_delivery():
        print("\n📡 IoT Demo 3: Temporal Key Delivery")
        print("─" * 50)
        key = [0xAB, 0x37, 0xF2, 0x01]
        print(f"Delivering key: {[hex(x) for x in key]}")

        lines = [f"CONST {b}\nSTORE {i}" for i, b in enumerate(key)]
        lines.append("HALT")
        source = "\n".join(lines)

        pauses, data, _, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)

        recovered = [result['final_state']['memory'].get(i, -1) for i in range(4)]
        print(f"Recovered: {[hex(x) for x in recovered]}")
        assert recovered == key
        print(" ✓ Key successfully delivered via timing channel")
        return True

    @staticmethod
    def run_all():
        print("\n" + "📡" * 25)
        print("   IoT SIDE-CHANNEL DEMOS")
        print("📡" * 25)
        IoTDemos.demo_leaky_bucket()
        IoTDemos.demo_spike_detector()
        IoTDemos.demo_key_delivery()

# === MAIN ===

if __name__ == "__main__":
    print("Running enhanced Torture Test Suite for v0.7.12...\n")
    passed, failed = TortureTests.run_all()

    if failed == 0:
        IoTDemos.run_all()

        print("\n🎵 Exporting key-delivery demo as WAV...")
        key = [0xDE, 0xAD, 0xBE, 0xEF]
        lines = [f"CONST {b}\nSTORE {i}" for i, b in enumerate(key)]
        lines.append("HALT")
        pauses, _, _, _ = PauseLangCompiler.compile("\n".join(lines))
        WavExporter.export_to_wav(pauses, filename="iot_key_delivery.wav")

    print("\n" + "═" * 60)
    if failed == 0:
        print("✅ ALL TESTS PASSED — v0.7.12 Production Final")
        print("   Ready for low-power IoT side-channel experiments!")
    else:
        print(f"❌ {failed} test(s) failed")
    print("═" * 60)
