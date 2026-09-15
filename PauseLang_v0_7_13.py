"""
PauseLang v0.7.14 (legacy module filename retained)
=========================
A bounded stack VM with a temporally encoded instruction/control stream.
See CHANGELOG.md for v0.7.14 fixes and benchmarks/RESULTS.md for measurements.
The module filename is retained for compatibility with existing imports.
"""

import time
import random
import struct
from typing import List, Tuple, Any, Dict, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from math import exp, isfinite
import re

# === FORMAL SPECIFICATION ===
SPEC = {
    'version': '0.7.14',
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
    """Decode only uniquely accepted symbols; calibration is per VM/session.

    Offset correction is the default. Affine correction is opt-in because
    noise on the 10 ms sync baseline is amplified when estimating clock skew.
    Calibration bounds also keep the sync phrase distinct from ordinary ops.
    """
    def __init__(self, quantum: Optional[float] = None,
                 guard_band: Optional[float] = None, calibration: str = 'offset'):
        self.quantum = SPEC['time_quantum'] if quantum is None else quantum
        self.guard_band = SPEC['guard_band'] if guard_band is None else guard_band
        if not isfinite(self.quantum) or self.quantum <= 0:
            raise ValueError('quantum must be finite and positive')
        if not isfinite(self.guard_band) or self.guard_band < 0:
            raise ValueError('guard_band must be finite and non-negative')
        if calibration not in ('offset', 'affine'):
            raise ValueError("calibration must be 'offset' or 'affine'")
        self.calibration = calibration
        self.calibration_history = deque(maxlen=10)
        self.reset()

    def reset(self):
        self.drift_estimate = 0.0  # legacy name: additive offset in seconds
        self.scale_estimate = 1.0
        self.calibration_history.clear()

    def adjusted(self, pause: float) -> float:
        return (pause - self.drift_estimate) / self.scale_estimate

    def quantize(self, pause: float) -> float:
        return round(self.adjusted(pause) / self.quantum) * self.quantum

    def in_guard_band(self, pause: float, target: float) -> bool:
        if not isfinite(pause) or pause <= 0:
            return False
        adjusted_us = int(round(self.adjusted(pause) * 1_000_000))
        target_us = int(round(target * 1_000_000))
        guard_us = int(round(self.guard_band * 1_000_000))
        return abs(adjusted_us - target_us) <= guard_us

    def decode(self, pause: float):
        matches = [instr for instr in INSTRUCTIONS.values()
                   if self.in_guard_band(pause, instr.pause)]
        # Never resolve overlapping windows by instruction-table order.
        return matches[0] if len(matches) == 1 else None

    def calibration_parameters(self, sync_pauses: List[float]):
        expected = SPEC['sync_phrase']
        if (len(sync_pauses) != len(expected) or len(expected) < 2
                or any(not isfinite(p) or p <= 0 for p in sync_pauses)):
            return None
        scale = 1.0
        if self.calibration == 'affine':
            baseline = expected[-1] - expected[0]
            if baseline <= 0:
                return None
            scale = (sync_pauses[-1] - sync_pauses[0]) / baseline
        offset = sum(p - scale * e for p, e in zip(sync_pauses, expected)) / len(expected)
        if not (0.9 <= scale <= 1.1) or abs(offset) > 0.020:
            return None
        # Residual acceptance uses the same canonical-time guard as decode.
        if any(abs((p - offset) / scale - e) > self.guard_band + 0.0000005
               for p, e in zip(sync_pauses, expected)):
            return None
        return scale, offset

    def calibrate(self, sync_pauses: List[float]) -> bool:
        parameters = self.calibration_parameters(sync_pauses)
        if parameters is None:
            return False
        self.scale_estimate, self.drift_estimate = parameters
        self.calibration_history.append(self.drift_estimate)
        return True

    def get_drift_trend(self) -> float:
        if not self.calibration_history:
            return 0.0
        return sum(self.calibration_history) / len(self.calibration_history)

# === ENUMS ===

class Flag(Enum):
    ZERO = auto()
    ODD = auto()
    NEGATIVE = auto()
    OVERFLOW = auto()
    # CARRY intentionally omitted: no instruction in this ISA produces or
    # consumes a carry bit.  Add it here only if carry-aware arithmetic ops
    # (ADDC, SUBB, …) are introduced.

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
    LOOP_MISMATCH = 15

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

# INSTRUCTIONS is keyed by pause duration in integer milliseconds to avoid
# IEEE 754 float-key ambiguity (e.g. 0.045 is not exactly representable).
# The `pause` field on each Instruction stores the canonical float (seconds)
# for human-readable output only.  All decode/lookup logic uses the int key.
INSTRUCTIONS = {
    # Arithmetic - STREAM OPS
    5:   Instruction('ADD',            0.005, '[STREAM] Add current and previous', OpCategory.STREAM),
    10:  Instruction('MEAN',           0.010, '[STREAM] Average of current and previous', OpCategory.STREAM),
    15:  Instruction('DIFF',           0.015, '[STREAM] Subtract previous from current', OpCategory.STREAM),
    20:  Instruction('SQUARE',         0.020, '[STREAM] Square current value', OpCategory.STREAM),

    # Conditional - STREAM OPS
    25:  Instruction('PASS',           0.025, '[STREAM] Pass unchanged (NO FLAG UPDATE)', OpCategory.STREAM, updates_flags=False),
    30:  Instruction('IF_GT_15_SQUARE',0.030, '[STREAM] Square if > 15', OpCategory.STREAM),
    35:  Instruction('DOUBLE_IF_EVEN', 0.035, '[STREAM] Double if even', OpCategory.STREAM),
    40:  Instruction('NEGATE_IF_ODD',  0.040, '[STREAM] Negate if odd', OpCategory.STREAM),

    # Stack - PURE STACK OPS
    45:  Instruction('PUSH',           0.045, '[STACK] Push operand to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),
    50:  Instruction('POP',            0.050, '[STACK] Pop from stack', OpCategory.STACK, requires_stack=1, stack_delta=-1),
    55:  Instruction('DUP',            0.055, '[STACK] Duplicate top', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=1),

    # Control - CONTROL FLOW
    60:  Instruction('JUMP_IF_ODD',    0.060, '[CONTROL] Jump if ODD flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    65:  Instruction('SKIP_NEXT',      0.065, '[CONTROL] Skip next instruction', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    70:  Instruction('LOOP_START',     0.070, '[CONTROL] Mark loop start', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    75:  Instruction('LOOP_END',       0.075, '[CONTROL] Pop TOS; loop if > 0', OpCategory.CONTROL, updates_flags=False, modifies_flow=True, requires_stack=1, stack_delta=-1),

    # Memory - HYBRID OPS
    # WORKED EXAMPLE: STORE takes its slot from the DATA stream, not the stack.
    #   Example: PUSH 99 / STORE 42  →  mem[42] = 99
    80:  Instruction('STORE',          0.080, '[HYBRID] Store TOS at mem[operand] (operand from data stream, not stack). Example: PUSH 99 / STORE 42 → mem[42]=99', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    85:  Instruction('LOAD',           0.085, '[HYBRID] Load mem[operand] to stack', OpCategory.HYBRID, stack_delta=1),
    90:  Instruction('SWAP',           0.090, '[STACK] Swap top two', OpCategory.STACK, updates_flags=False, requires_stack=2),
    95:  Instruction('CLEAR_STACK',    0.095, '[STACK] Clear entire stack', OpCategory.STACK, updates_flags=False),

    # Stack Arithmetic - PURE STACK OPS
    100: Instruction('ADD2',           0.100, '[STACK] Pop 2, push sum', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    105: Instruction('SUB2',           0.105, '[STACK] Pop 2, push difference', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    110: Instruction('MUL2',           0.110, '[STACK] Pop 2, push product', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    115: Instruction('DIV2',           0.115, '[STACK] Pop 2, push quotient', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    120: Instruction('MOD2',           0.120, '[STACK] Pop 2, push modulo', OpCategory.STACK, requires_stack=2, stack_delta=-1),

    # Meta - SYSTEM OPS
    125: Instruction('SET_META',       0.125, '[SYSTEM] Toggle meta mode', OpCategory.SYSTEM, updates_flags=False),
    130: Instruction('JUMP_IF_ZERO',   0.130, '[CONTROL] Jump if ZERO flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    135: Instruction('CALL',           0.135, '[CONTROL] Call subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    140: Instruction('RET',            0.140, '[CONTROL] Return from subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),

    # System
    145: Instruction('NOP',            0.145, '[SYSTEM] No operation', OpCategory.SYSTEM, updates_flags=False),
    150: Instruction('HALT',           0.150, '[SYSTEM] Halt execution', OpCategory.SYSTEM, updates_flags=False, modifies_flow=True),

    # Unconditional Jump
    155: Instruction('JUMP',           0.155, '[CONTROL] Unconditional jump', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),

    # Jump if not zero (kept in natural numeric order)
    160: Instruction('JUMP_IF_NONZERO',0.160, '[CONTROL] Jump if not ZERO', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),

    # ROT
    165: Instruction('ROT',            0.165, '[STACK] Rotate top three: ( a b c -- b c a )', OpCategory.STACK, updates_flags=False, requires_stack=3, stack_delta=0),

    # IX Register
    200: Instruction('SETIX',          0.200, '[STACK] Pop stack → IX register', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=-1),
    205: Instruction('LOADI',          0.205, '[STACK] Push mem[IX] to stack (returns 0 if uninitialised)', OpCategory.STACK, updates_flags=True, stack_delta=1),
    210: Instruction('STOREI',         0.210, '[HYBRID] Store TOS at mem[IX]', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    215: Instruction('INCIX',          0.215, '[SYSTEM] IX = (IX + 1) % max_slots', OpCategory.SYSTEM, updates_flags=False),
    220: Instruction('GETIX',          0.220, '[STACK] Push IX register to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),
}

# Maps opcode name → pause in integer milliseconds.
OPCODE_TO_PAUSE = {instr.opcode: key_ms for key_ms, instr in INSTRUCTIONS.items()}

def _pause_to_ms(pause_s: float) -> int:
    """Convert a pause in seconds to the nearest integer-millisecond key."""
    return int(round(pause_s * 1000))

# Backward-compatible alias for v0.7.13's misnamed private helper.
_pause_to_us = _pause_to_ms

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
    def __init__(self, gas_limit: int = 20000, trap_policy: str = 'continue', memory_mode: str = 'wrap', debug: bool = False,
                 guard_band: Optional[float] = None, calibration: str = 'offset'):
        if trap_policy not in ('continue', 'halt', 'raise'):
            raise ValueError('Invalid trap_policy')
        if memory_mode not in ('wrap', 'strict'):
            raise ValueError('Invalid memory_mode')
        if not isinstance(gas_limit, int) or gas_limit < 0:
            raise ValueError('gas_limit must be a non-negative integer')
        self.state = VMState()
        self.gas_limit = gas_limit
        self.trap_policy = trap_policy
        self.memory_mode = memory_mode
        self.debug = debug
        self.quantizer = TimeQuantizer(guard_band=guard_band, calibration=calibration)
        self.execution_trace = []

    def reset(self):
        self.state = VMState()
        self.execution_trace = []
        self.quantizer.reset()

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
            self.state.trap_stack[-1] = TrapCode.TRAP_STORM
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
        if self.state.gas_used >= self.gas_limit:
            self.push_trap(TrapCode.GAS_EXHAUSTED)
            self.state.halted = True   # <-- FIXED v0.7.12: explicitly halt on gas exhaustion
            return False
        self.state.gas_used += 1
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
                a = self.state.stack.pop()
                r = self.wrap_int32(a + b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'SUB2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop()
                r = self.wrap_int32(a - b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'MUL2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop()
                r = self.wrap_int32(a * b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'DIV2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop()
                if b == 0:
                    self.push_trap(TrapCode.DIV_BY_ZERO)
                    self.state.stack.append(0)
                    result = "DIV_BY_ZERO"
                else:
                    quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                    r = self.wrap_int32(quotient)
                    self.state.stack.append(r)
                    result = r
        elif opcode == 'MOD2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b = self.state.stack.pop()
                a = self.state.stack.pop()
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
                if not (0 <= slot < SPEC['max_memory_slots']):
                    self.push_trap(TrapCode.INVALID_MEMORY)
                    return 'INVALID_MEMORY'
                store_value = self.state.stack.pop()
                self.state.memory[slot] = store_value
                result = f"STORED {store_value} @ {slot}"
        elif opcode == 'LOAD':
            if self.memory_mode == 'strict':
                slot = value
                if not (0 <= slot < SPEC['max_memory_slots']):
                    self.push_trap(TrapCode.INVALID_MEMORY)
                    result = "INVALID_MEMORY"
                    return result
            else:
                slot = value % SPEC['max_memory_slots'] if self.state.lane == Lane.DATA else value
            if not (0 <= slot < SPEC['max_memory_slots']):
                self.push_trap(TrapCode.INVALID_MEMORY)
                return 'INVALID_MEMORY'
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
        :param sync: If True, calibrate offset/skew using the sync phrase (if present).
        :param labels: Optional label dictionary (from compiler).
        :param strict_sync: If True, NEVER auto‑strip the sync phrase.
                           Default False (auto‑strip if the stream begins with sync_phrase).
                           Set to True to avoid the auto‑strip foot‑gun.
        """
        if len(data_stream) != len(pause_stream):
            return {'error': f'Stream length mismatch: data={len(data_stream)}, pauses={len(pause_stream)}'}
        if any(not isinstance(value, int) for value in data_stream):
            return {'error': 'Operands must be integers'}
        if labels:
            self.state.labels = labels

        base_offset = 0

        def matches_sync_phrase(pauses):
            phrase = pauses[:len(SPEC['sync_phrase'])]
            if sync:
                return self.quantizer.calibration_parameters(phrase) is not None
            # Detect an uncalibrated phrase independently of earlier sessions.
            raw_quantizer = TimeQuantizer(guard_band=self.quantizer.guard_band)
            return (len(phrase) == len(SPEC['sync_phrase']) and
                    all(raw_quantizer.in_guard_band(p, e)
                        for p, e in zip(phrase, SPEC['sync_phrase'])))

        # Auto-strip sync if present (only if strict_sync is False)
        if not strict_sync and len(pause_stream) >= len(SPEC['sync_phrase']) and matches_sync_phrase(pause_stream):
            if sync:
                if not self.quantizer.calibrate(pause_stream[:len(SPEC['sync_phrase'])]):
                    return {'error': 'Sync calibration failed'}
            data_stream = data_stream[len(SPEC['sync_phrase']):]
            pause_stream = pause_stream[len(SPEC['sync_phrase']):]
            base_offset = len(SPEC['sync_phrase'])

        results = []
        while self.state.pc < len(data_stream) and not self.state.halted:
            if not self.check_gas(): break

            self.state.instructions_executed += 1

            executed_pc = self.state.pc
            value = data_stream[self.state.pc]
            raw_pause = pause_stream[self.state.pc]
            instr = self.quantizer.decode(raw_pause)
            if instr is None:
                self.push_trap(TrapCode.INVALID_INSTRUCTION)
                instr = INSTRUCTIONS[25]  # PASS as fallback (25 ms key)

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
                if self.state.loop_stack and self.state.loop_stack[-1] == self.state.pc:
                    result = 'LOOP_START'
                elif len(self.state.loop_stack) >= SPEC['max_loop_depth']:
                    self.push_trap(TrapCode.LOOP_DEPTH_EXCEEDED)
                    result = "LOOP_DEPTH_EXCEEDED"
                else:
                    self.state.loop_stack.append(self.state.pc)
                    result = "LOOP_START"
            elif instr.opcode == 'LOOP_END':
                if len(self.state.stack) == 0:
                    self.push_trap(TrapCode.STACK_UNDERFLOW)
                    result = "LOOP_END STACK_UNDERFLOW"
                elif not self.state.loop_stack:
                    # No matching LOOP_START — trap rather than silently consuming TOS.
                    self.push_trap(TrapCode.LOOP_MISMATCH)
                    result = "LOOP_MISMATCH"
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
                'pc': executed_pc,
                'absolute_pc': executed_pc + base_offset,
                'opcode': instr.opcode,
                'value': value,
                'result': result,
                'stack_depth': len(self.state.stack),
                'stack_snapshot': self.state.stack.copy(),  # per-step snapshot for disassemble
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
                 "*PCs are absolute positions in the supplied stream; trace is chronological.*"]
        labels_reverse = {v: k for k, v in self.state.labels.items()} if self.state.labels else {}
        for i, step in enumerate(self.execution_trace):
            pc = step.get('absolute_pc', step['pc'] + len(SPEC['sync_phrase']))
            label = ""
            if show_labels and pc in labels_reverse:
                label = f"{labels_reverse[pc]}:"
            label_col = f"{label:12}" if show_labels else ""
            state_col = ""
            if show_state:
                # Use the per-step snapshot captured during execution, not current state.
                snap = step.get('stack_snapshot', [])
                stack_preview = str(snap[-3:]) if snap else "[]"
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
                        if not (0 <= slot < SPEC['max_memory_slots']):
                            mem_detail = f" [INVALID (META)]"
                        else:
                            mem_detail = f" [→ slot {slot} (META)]"
            cat_icon = INSTRUCTIONS[OPCODE_TO_PAUSE[step['opcode']]].signature()  # key is integer milliseconds
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
            if step['opcode'] in ['JUMP','JUMP_IF_ODD','JUMP_IF_ZERO','JUMP_IF_NONZERO','SKIP_NEXT','LOOP_START','LOOP_END','CALL','RET','HALT']:
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
        'STOREI_POP':['STOREI'],
        'NOT':       [('PUSH', -1), 'SWAP', 'SUB2'],
        'LNOT':      [('PUSH', 1), 'SWAP', 'SUB2'],
        'NEG':       [('PUSH', 0), 'SWAP', 'SUB2'],
        'SETF':      [('PUSH', 0), 'ADD2'],
    }

    @staticmethod
    def compile(source: str, debug: bool = False) -> Tuple[List[float], List[int], List[str], Dict[str, int]]:
        lines = source.splitlines()
        labels = {}
        pc = len(SPEC['sync_phrase'])

        # First pass: collect labels
        for line_num, raw_line in enumerate(lines):
            clean = raw_line.strip().split('#')[0].strip()
            if not clean:
                continue
            if clean.endswith(':'):
                label_name = clean[:-1].strip()
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', label_name):
                    raise ValueError(f"Invalid label '{label_name}' at line {line_num + 1}")
                if label_name in labels:
                    raise ValueError(f"Duplicate label '{label_name}' at line {line_num + 1}")
                labels[label_name] = pc
                if debug:
                    print(f"Label '{label_name}' → PC {pc}")
                continue
            parts = clean.split()
            if len(parts) > 2:
                raise ValueError(f'One instruction per line; extra tokens at line {line_num + 1}')
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
            if len(parts) > 2:
                raise ValueError(f'One instruction per line; extra tokens at line {line_num + 1}')
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
                elif re.fullmatch(r'[+-]?[0-9]+', operand):
                    value = int(operand)
                else:
                    raise ValueError(f"Unknown operand '{operand}' at line {line_num + 1}")

            if opcode in PauseLangCompiler.MACROS:
                for macro_step in PauseLangCompiler.MACROS[opcode]:
                    if isinstance(macro_step, tuple):
                        macro_op, imm = macro_step
                        pauses.append(INSTRUCTIONS[OPCODE_TO_PAUSE[macro_op]].pause)
                        data.append(imm)
                        comments.append(f"{macro_op} {imm} (from {original_opcode})")
                    else:
                        pauses.append(INSTRUCTIONS[OPCODE_TO_PAUSE[macro_step]].pause)
                        data.append(value)
                        comments.append(f"{macro_step} (from {original_opcode})")
            elif opcode in OPCODE_TO_PAUSE:
                pauses.append(INSTRUCTIONS[OPCODE_TO_PAUSE[opcode]].pause)
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
        A terminal click makes the final interval measurable too.
        """
        try:
            import numpy as np
            from scipy.io import wavfile
        except ImportError:
            print("WAV export requires numpy and scipy. Install with: pip install numpy scipy")
            return

        if not isinstance(sample_rate, int) or sample_rate < 4000:
            raise ValueError('sample_rate must be an integer >= 4000 Hz')
        if any(not isfinite(p) or p < 0.001 for p in pauses):
            raise ValueError('WAV pauses must be finite and at least 1 ms')

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
        audio.append(click)  # terminal marker measures the final pause too
        audio = np.concatenate(audio)
        wavfile.write(filename, sample_rate, audio)
        print(f"Exported {len(pauses)} instructions to {filename}")

# === ENHANCED TORTURE TESTS ===

class TortureTests:
    """Compatibility runner; tests now live outside the VM implementation."""
    @staticmethod
    def run_all():
        from tests.legacy_torture import TortureTests as LegacyTests
        return LegacyTests.run_all()

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
    import sys
    # Redirected Windows output may use cp1252. Escape unsupported decoration
    # without changing the caller's encoding or crashing an otherwise valid run.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    print("Running PauseLang v0.7.14 test suite...\n")
    import unittest
    from pathlib import Path
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'),
                                               top_level_dir=str(Path(__file__).parent))
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = len(test_result.failures) + len(test_result.errors)

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
        print("✅ ALL TESTS PASSED — v0.7.14")
        print("   Ready for low-power IoT side-channel experiments!")
    else:
        print(f"❌ {failed} test(s) failed")
    print("═" * 60)
    raise SystemExit(1 if failed else 0)
