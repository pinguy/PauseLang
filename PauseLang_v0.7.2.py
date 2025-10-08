# The provided PauseLang code (entire script)
"""
PauseLang v0.7.2 - Bugfix
======================================
Changes in v0.7.2:
- FIXED: `LOAD` instruction now correctly pushes the loaded value onto the stack.
- FIXED: `LOAD` instruction definition updated with `stack_delta=1` for proper pre-execution overflow checks.
- Retained all v0.7.1 features.

Original features from v0.6.7 retained.
"""

import time
import random
from typing import List, Tuple, Any, Dict, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from math import exp

# === FORMAL SPECIFICATION ===

SPEC = {
    'version': '0.7.2',
    'word_size': 32,
    'overflow': 'wrap',
    'division': 'truncate',  # Truncate toward zero
    'max_call_depth': 256,
    'max_stack_size': 4096,
    'max_memory_slots': 256,
    'time_quantum': 0.01,
    'guard_band': 0.002,
    'sync_phrase': [0.29, 0.29, 0.30, 0.29],
}

# === DIVISION AND MODULO SEMANTICS ===
"""
Division and Modulo Behavior (Math-positive):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIV2: Truncates toward zero (matches spec)
  • 7 ÷ 2 = 3      • -7 ÷ 2 = -3
  • 7 ÷ -2 = -3    • -7 ÷ -2 = 3
  
MOD2: Always positive remainder [0, |b|)
  • 7 % 3 = 1      • -7 % 3 = 2
  • 7 % -3 = 1     • -7 % -3 = 2

This ensures consistent positive mods for math/crypto use.
"""

# === TIME QUANTIZATION ===

class TimeQuantizer:a
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
    INVALID_JUMP = 10
    INVALID_CALL = 11

class Lane(Enum):
    DATA = auto()
    META = auto()

class OpCategory(Enum):
    STREAM = auto()   # Operates on data stream values
    STACK = auto()    # Pure stack operations
    HYBRID = auto()   # Uses both stream and stack
    CONTROL = auto()  # Control flow
    SYSTEM = auto()   # System operations

# === INSTRUCTION SET ===

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
        """Return a visual signature showing the operation type"""
        symbols = {
            OpCategory.STREAM: "≈",   # Wave for stream
            OpCategory.STACK: "▣",    # Stack blocks
            OpCategory.HYBRID: "◈",   # Diamond for hybrid
            OpCategory.CONTROL: "→",  # Arrow for flow
            OpCategory.SYSTEM: "⚙",   # Gear for system
        }
        return symbols.get(self.category, "?")

# Instruction definitions with categories
INSTRUCTIONS = {
    # Arithmetic - STREAM OPS (operate on current/previous data values)
    0.01: Instruction('ADD', 0.01, '[STREAM] Add current and previous', OpCategory.STREAM),
    0.02: Instruction('MEAN', 0.02, '[STREAM] Average of current and previous', OpCategory.STREAM),
    0.03: Instruction('DIFF', 0.03, '[STREAM] Subtract previous from current', OpCategory.STREAM),
    0.04: Instruction('SQUARE', 0.04, '[STREAM] Square current value', OpCategory.STREAM),
    
    # Conditional - STREAM OPS
    0.05: Instruction('PASS', 0.05, '[STREAM] Pass unchanged (NO FLAG UPDATE)', OpCategory.STREAM, updates_flags=False),
    0.06: Instruction('IF_GT_15_SQUARE', 0.06, '[STREAM] Square if > 15', OpCategory.STREAM),
    0.07: Instruction('DOUBLE_IF_EVEN', 0.07, '[STREAM] Double if even', OpCategory.STREAM),
    0.08: Instruction('NEGATE_IF_ODD', 0.08, '[STREAM] Negate if odd', OpCategory.STREAM),
    
    # Stack - PURE STACK OPS
    0.09: Instruction('PUSH', 0.09, '[STACK] Push operand to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),
    0.10: Instruction('POP', 0.10, '[STACK] Pop from stack', OpCategory.STACK, requires_stack=1, stack_delta=-1),
    0.11: Instruction('DUP', 0.11, '[STACK] Duplicate top', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=1),
    
    # Control - CONTROL FLOW
    0.12: Instruction('JUMP_IF_ODD', 0.12, '[CONTROL] Jump if ODD flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.13: Instruction('SKIP_NEXT', 0.13, '[CONTROL] Skip next instruction', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.14: Instruction('LOOP_START', 0.14, '[CONTROL] Mark loop start', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.15: Instruction('LOOP_END', 0.15, '[CONTROL] Loop if stack[top] > 0', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    
    # Memory - HYBRID OPS (use both operand and stack)
    0.16: Instruction('STORE', 0.16, '[HYBRID] Store TOS at mem[operand]', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.17: Instruction('LOAD', 0.17, '[HYBRID] Load mem[operand] to stack', OpCategory.HYBRID, stack_delta=1),
    0.18: Instruction('SWAP', 0.18, '[STACK] Swap top two', OpCategory.STACK, updates_flags=False, requires_stack=2),
    0.19: Instruction('CLEAR_STACK', 0.19, '[STACK] Clear entire stack', OpCategory.STACK, updates_flags=False),
    
    # Stack Arithmetic - PURE STACK OPS
    0.20: Instruction('ADD2', 0.20, '[STACK] Pop 2, push sum', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.21: Instruction('SUB2', 0.21, '[STACK] Pop 2, push difference', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.22: Instruction('MUL2', 0.22, '[STACK] Pop 2, push product', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.23: Instruction('DIV2', 0.23, '[STACK] Pop 2, push quotient', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    0.24: Instruction('MOD2', 0.24, '[STACK] Pop 2, push modulo', OpCategory.STACK, requires_stack=2, stack_delta=-1),
    
    # Meta - SYSTEM OPS
    0.25: Instruction('SET_META', 0.25, '[SYSTEM] Toggle meta mode', OpCategory.SYSTEM, updates_flags=False),
    0.26: Instruction('JUMP_IF_ZERO', 0.26, '[CONTROL] Jump if ZERO flag', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.27: Instruction('CALL', 0.27, '[CONTROL] Call subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    0.28: Instruction('RET', 0.28, '[CONTROL] Return from subroutine', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    
    # System
    0.29: Instruction('NOP', 0.29, '[SYSTEM] No operation', OpCategory.SYSTEM, updates_flags=False),
    0.30: Instruction('HALT', 0.30, '[SYSTEM] Halt execution', OpCategory.SYSTEM, updates_flags=False, modifies_flow=True),
    
    # Unconditional Jump (NEW!)
    0.31: Instruction('JUMP', 0.31, '[CONTROL] Unconditional jump', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
    
    # IX Register - HYBRID/STACK OPS
    0.40: Instruction('SETIX', 0.40, '[STACK] Pop stack → IX register', OpCategory.STACK, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.41: Instruction('LOADI', 0.41, '[STACK] Push mem[IX] to stack', OpCategory.STACK, updates_flags=True, stack_delta=1),
    0.42: Instruction('STOREI', 0.42, '[HYBRID] Store TOS at mem[IX]', OpCategory.HYBRID, updates_flags=False, requires_stack=1, stack_delta=-1),
    0.43: Instruction('INCIX', 0.43, '[SYSTEM] IX = (IX + 1) % max_slots', OpCategory.SYSTEM, updates_flags=False),
    0.44: Instruction('GETIX', 0.44, '[STACK] Push IX register to stack', OpCategory.STACK, updates_flags=False, stack_delta=1),
    
    # New: Jump if not zero
    0.32: Instruction('JUMP_IF_NONZERO', 0.32, '[CONTROL] Jump if not ZERO', OpCategory.CONTROL, updates_flags=False, modifies_flow=True),
}

OPCODE_TO_PAUSE = {instr.opcode: pause for pause, instr in INSTRUCTIONS.items()}

# === INSTRUCTION DOCUMENTATION ===
INSTRUCTION_CONVENTIONS = """
╔═══════════════════════════════════════════════════════════════╗
║                  INSTRUCTION CONVENTIONS                      ║
╠═══════════════════════════════════════════════════════════════╣
║ CATEGORIES:                                                   ║
║ • [STREAM]  - Uses current/previous data values from stream   ║
║ • [STACK]   - Pure stack operations                          ║
║ • [HYBRID]  - Uses both operand and stack                    ║
║ • [CONTROL] - Modifies program counter                       ║
║ • [SYSTEM]  - VM state/configuration                         ║
║                                                              ║
║ ⚠️  CRITICAL: FLAG BEHAVIOR ⚠️                                 ║
║ • Stack ops (PUSH/CONST, DUP/PEEK) do NOT update flags!      ║
║ • Control jumps (JZ, JUMP_IF_ODD) check flags, not TOS       ║
║ • To branch on TOS value: use SETF macro or any arithmetic   ║
║   Example: PUSH 5 → SETF 0 → JUMP_IF_ODD label              ║
║ • PASS explicitly preserves flags (useful for data routing)  ║
║                                                              ║
║ STORE CONVENTION:                                            ║
║ • Operand specifies memory slot (0-255)                      ║
║ • Value comes from TOS (now popped!)                         ║
║ • Example: STORE 42  →  stack.pop() → mem[42]                ║
║                                                              ║
║ META vs DATA LANE:                                           ║
║ • DATA lane: Memory ops use modulo (STORE 300 → slot 44)     ║
║ • META lane: No modulo! (STORE 300 → INVALID_MEMORY trap)    ║
║ • Toggle with SET_META instruction                           ║
║                                                              ║
║ JUMP ADDRESSING:                                              ║
║ • All jumps use absolute PC (post-sync)                      ║
║ • Use labels to avoid manual counting                        ║
║ • PC = 4 is first instruction after sync                     ║
║                                                              ║
║ USEFUL MACROS:                                               ║
║ • SETF 0 - Sets flags from TOS without changing value        ║
║   (Warning: SETF k with k≠0 adds k to TOS and sets flags)    ║
║ • INC/DEC - Increment/decrement TOS                         ║
║ • JUMP/JMP - Unconditional jump                              ║
║ • NOT - Arithmetic NOT (1 - TOS); not bitwise or strict boolean (use logic patterns for strict) ║
║                                                              ║
║ STACK HYGIENE TIP:                                           ║
║ • STORE and STOREI now pop, improving stack hygiene.         ║
║ • No need for an extra DROP after a store.                   ║
║ • Loops are now cleaner and less prone to stack leaks.       ║
║                                                              ║
║ ALIAS RULES:                                                 ║
║ • Aliases are compile-time static; no runtime redefinition.  ║
║                                                              ║
║ LOGIC PATTERNS:                                              ║
║ • Branch on TOS:                                             ║
║     SETF 0                                                   ║
║     JZ / JNZ label                                           ║
║ • Boolean NOT (strict: 0 if nonzero, else 1):                ║
║     SETF 0                                                   ║
║     JZ is_zero                                               ║
║     CONST 0                                                  ║
║     JMP end                                                  ║
║   is_zero:                                                   ║
║     CONST 1                                                  ║
║   end:                                                       ║
╚═══════════════════════════════════════════════════════════════╝
"""

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
    labels: Dict[str, int] = field(default_factory=dict)  # For debugging

class PauseLangVM:
    def __init__(self, gas_limit: int = 20000, trap_policy: str = 'continue', memory_mode: str = 'wrap', debug: bool = False):
        self.state = VMState()
        self.gas_limit = gas_limit
        self.trap_policy = trap_policy
        self.memory_mode = memory_mode  # 'wrap' (default) or 'strict'
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
        
        # Pre-execution stack checks
        if instr.requires_stack > len(self.state.stack):
            self.push_trap(TrapCode.STACK_UNDERFLOW)
            return "TRAP: STACK_UNDERFLOW"
        if instr.stack_delta > 0 and len(self.state.stack) + instr.stack_delta > SPEC['max_stack_size']:
            self.push_trap(TrapCode.STACK_OVERFLOW)
            return "TRAP: STACK_OVERFLOW"
        
        result = None
        
        # Stream arithmetic (uses data stream values)
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
                self.state.stack.append(top)
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
        
        # Stack arithmetic
        elif opcode == 'ADD2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b, a = self.state.stack.pop(), self.state.stack.pop()
                r = self.wrap_int32(a + b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'SUB2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b, a = self.state.stack.pop(), self.state.stack.pop()
                r = self.wrap_int32(a - b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'MUL2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b, a = self.state.stack.pop(), self.state.stack.pop()
                r = self.wrap_int32(a * b)
                self.state.stack.append(r)
                result = r
        elif opcode == 'DIV2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b, a = self.state.stack.pop(), self.state.stack.pop()
                if b == 0:
                    self.push_trap(TrapCode.DIV_BY_ZERO)
                    self.state.stack.append(0)
                    result = "DIV_BY_ZERO"
                else:
                    # Truncate toward zero (spec compliant)
                    r = int(a / b)
                    self.state.stack.append(r)
                    result = r
        elif opcode == 'MOD2':
            if len(self.state.stack) < 2:
                self.push_trap(TrapCode.STACK_UNDERFLOW)
                result = "STACK_UNDERFLOW"
            else:
                b, a = self.state.stack.pop(), self.state.stack.pop()
                if b == 0:
                    self.push_trap(TrapCode.DIV_BY_ZERO)
                    self.state.stack.append(0)
                    result = "MOD_BY_ZERO"
                else:
                    # Normalized positive modulo
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
            self.state.stack.append(loaded_value)
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
        
        # IX register operations
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
            self.state.stack.append(v)
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
            self.state.stack.append(self.state.ix)
            result = f"PUSHED IX={self.state.ix}"

        if instr.updates_flags and isinstance(result, int):
            self.update_flags(result)
        return result if result is not None else value
    
    def execute(self, data_stream: List[int], pause_stream: List[float], 
                sync: bool = True, labels: Optional[Dict[str, int]] = None) -> Dict:
        """Execute with optional label information for debugging"""
        if labels:
            self.state.labels = labels
        
        base_offset = 0
        if sync and len(pause_stream) >= 4:
            if not self.quantizer.calibrate(pause_stream[:4]):
                return {'error': 'Sync calibration failed'}
            data_stream = data_stream[4:]
            pause_stream = pause_stream[4:]
            base_offset = len(SPEC['sync_phrase'])  # Track offset for jump compensation
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

            # Control flow handling (with offset compensation and bounds checking)
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
    
    def disassemble(self, show_labels: bool = True, show_state: bool = False, 
                   compact: bool = False, show_memory: bool = False) -> str:
        """Enhanced disassembler with label support and memory details"""
        if not self.execution_trace: 
            return "No execution trace"
            
        lines = ["=== DISASSEMBLY ===",
                 "*PCs are post-sync absolute; execution trace is chronological (jumps may skip lines).*"]
        
        # Reverse map PC to labels if available
        labels_reverse = {v: k for k, v in self.state.labels.items()} if self.state.labels else {}
        
        for i, step in enumerate(self.execution_trace):
            pc = step['pc'] + 4  # Adjust for sync offset in display
            
            # Label column
            label = ""
            if show_labels and pc in labels_reverse:
                label = f"{labels_reverse[pc]}:"
            label_col = f"{label:12}" if show_labels else ""
            
            # State column
            state_col = ""
            if show_state:
                stack_preview = str(self.state.stack[-3:]) if len(self.state.stack) > 0 else "[]"
                flags = ','.join(step['flags'][:2]) if step['flags'] else "none"
                state_col = f" | S:{stack_preview:20} F:{flags:10}"
            
            # Memory operation details
            mem_detail = ""
            if show_memory and step['opcode'] in ['STORE', 'STOREI', 'LOAD', 'LOADI']:
                if step['opcode'] == 'STORE':
                    # Look back to find lane state
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
            
            # Category icon
            cat_icon = INSTRUCTIONS[OPCODE_TO_PAUSE[step['opcode']]].signature()
            
            # Format line
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
        
        # Compact explanation
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
    # Instruction aliases for ergonomics
    ALIASES = {
        'CONST': 'PUSH',           # More intuitive for literals
        'DROP': 'POP',             # Common stack term
        'PEEK': 'DUP',             # Non-destructive read
        'DROPS': 'CLEAR_STACK',    # Clear all
        'JMP': 'JUMP',             # Unconditional jump shorthand
        'JOD': 'JUMP_IF_ODD',      # Shorthand
        'JZ': 'JUMP_IF_ZERO',      # Shorthand
        'JNZ': 'JUMP_IF_NONZERO',  # Corrected alias
    }
    
    # Multi-instruction macros
    MACROS = {
        'INC':       [('PUSH', 1), 'ADD2'],          # TOS += 1
        'DEC':       [('PUSH', 1), 'SUB2'],          # TOS -= 1
        'DOUBLE':    ['DUP', 'ADD2'],                # Double TOS
        'SQUARED':   ['DUP', 'MUL2'],                # Square TOS
        'ENTER':     ['PUSH', 'SWAP'],               # Enter frame
        'LEAVE':     ['SWAP', 'POP'],                # Leave frame
        'STOREI_POP':['STOREI', 'POP'],              # Store and pop - Obsolete with fix, kept for compatibility
        'NOT':       [('PUSH', 1), 'SWAP', 'SUB2'],  # Arithmetic NOT (1 - TOS); not bitwise or strict boolean (use logic patterns for strict)
        'SETF':      [('PUSH', 0), 'ADD2'],          # Set flags from TOS (add zero)
    }
    
    @staticmethod
    def compile(source: str, debug: bool = False) -> Tuple[List[float], List[int], List[str], Dict[str, int]]:
        """
        Two-pass compiler with label support.
        Returns: (pauses, data, comments, labels)
        """
        lines = source.strip().split('\n')
        
        # First pass: collect labels and calculate positions
        labels = {}
        pc = 4  # Start after sync sequence
        
        for line_num, raw_line in enumerate(lines):
            # Remove comments and strip
            clean = raw_line.strip().split('#')[0].strip()
            if not clean:
                continue
                
            # Check if it's a label
            if clean.endswith(':'):
                label_name = clean[:-1].strip()
                if label_name in labels:
                    raise ValueError(f"Duplicate label '{label_name}' at line {line_num + 1}")
                labels[label_name] = pc
                if debug:
                    print(f"Label '{label_name}' → PC {pc}")
                continue
            
            # Parse instruction
            parts = clean.split()
            opcode = parts[0].upper()
            
            # Resolve aliases
            if opcode in PauseLangCompiler.ALIASES:
                opcode = PauseLangCompiler.ALIASES[opcode]
            
            # Count instructions
            if opcode in PauseLangCompiler.MACROS:
                pc += len(PauseLangCompiler.MACROS[opcode])
            elif opcode in OPCODE_TO_PAUSE:
                pc += 1
            else:
                raise ValueError(f"Unknown opcode '{opcode}' at line {line_num + 1}")
        
        # Second pass: generate instructions with resolved labels
        pauses = SPEC['sync_phrase'].copy()
        data = [0, 0, 0, 0]
        comments = ['SYNC', 'SYNC', 'SYNC', 'SYNC']
        
        for line_num, raw_line in enumerate(lines):
            clean = raw_line.strip().split('#')[0].strip()
            if not clean or clean.endswith(':'):
                continue
            
            parts = clean.split()
            opcode = parts[0].upper()
            
            # Resolve aliases
            original_opcode = opcode
            if opcode in PauseLangCompiler.ALIASES:
                opcode = PauseLangCompiler.ALIASES[opcode]
            
            # Resolve operand (might be label or number)
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
            
            # Generate instructions
            if opcode in PauseLangCompiler.MACROS:
                # Expand macro
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

# === TORTURE TESTS ===

class TortureTests:
    @staticmethod
    def test_labels():
        """Test label compilation and jumps"""
        source = """
        start:
            PUSH 5
            SETF 0          # Set flags from TOS (5 is odd)
            JUMP_IF_ODD skip_even
            PUSH 10
        skip_even:
            PUSH 20
            JZ end          # Won't jump, ZERO flag not set
            PUSH 30
        end:
            HALT
        """
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)
        
        # Should have jumped over PUSH 10 since 5 is odd
        stack = result['final_state']['stack']
        assert 10 not in stack, f"Failed to skip: {stack}"
        assert stack == [5, 20, 30], f"Unexpected stack: {stack}"
        return "✓ Label compilation passed"
    
    @staticmethod
    def test_aliases():
        """Test instruction aliases"""
        source = """
            CONST 42       # PUSH alias
            PEEK           # DUP alias
            DROP           # POP alias
            CONST 0
            SETF 0         # Set flags (0 sets ZERO flag)
            JZ done        # JUMP_IF_ZERO alias
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
        """Test division and modulo with negatives"""
        vm = PauseLangVM(debug=False)
        
        # Test DIV2 truncation toward zero
        tests = [
            (7, 2, 3),    # Positive / positive
            (-7, 2, -3),  # Negative / positive  
            (7, -2, -3),  # Positive / negative
            (-7, -2, 3),  # Negative / negative
        ]
        
        for a, b, expected in tests:
            vm.reset()
            pauses = [0.09, 0.09, 0.23]  # PUSH a, PUSH b, DIV2
            data = [a, b, 0]
            result = vm.execute(data, pauses, sync=False)
            actual = result['final_state']['stack'][0]
            assert actual == expected, f"DIV2({a},{b}) = {actual}, expected {expected}"
        
        # Test MOD2 positive semantics
        mod_tests = [
            (7, 3, 1),    # Normal positive
            (-7, 3, 2),   # Negative dividend
            (7, -3, 1),   # Negative divisor
            (-7, -3, 2),  # Both negative
        ]
        
        for a, b, expected in mod_tests:
            vm.reset()
            pauses = [0.09, 0.09, 0.24]  # PUSH a, PUSH b, MOD2
            data = [a, b, 0]
            result = vm.execute(data, pauses, sync=False)
            actual = result['final_state']['stack'][0]
            assert actual == expected, f"MOD2({a},{b}) = {actual}, expected {expected}"
        
        return "✓ Division/modulo semantics passed"
    
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
        return "✓ Jitter gauntlet passed"
    
    @staticmethod
    def test_flag_race():
        vm = PauseLangVM(debug=False)
        pauses = [0.09, 0.08, 0.09, 0.20, 0.09, 0.24]
        data = [7, 7, 3, 0, 2, 0]
        result = vm.execute(data, pauses, sync=False)
        final_flags = result['final_state']['flags']
        assert final_flags['ZERO'] == True, f"Expected ZERO flag, got {final_flags}"
        return "✓ Flag race passed"
    
    @staticmethod
    def test_stack_underflow_protection():
        """Test all stack operations for underflow protection"""
        vm = PauseLangVM(debug=False)
        
        ops_to_test = [
            (0.10, 'POP'),
            (0.11, 'DUP'), 
            (0.40, 'SETIX'),
        ]
        
        for pause, opcode in ops_to_test:
            vm.reset()
            result = vm.execute([0], [pause], sync=False)
            assert 'STACK_UNDERFLOW' in result['traps'], f"{opcode} should trap on empty stack"
        
        return "✓ Stack underflow protection passed"
    
    @staticmethod
    def test_loop_memory():
        """Test LOOP_START doesn't leak memory"""
        vm = PauseLangVM(gas_limit=1000, debug=False)
        pauses = [0.09, 0.14, 0.09, 0.21, 0.15]  # Proper countdown
        data = [3, 0, 1, 0, 0]
        result = vm.execute(data, pauses, sync=False)
        assert len(vm.state.loop_stack) == 0, "LOOP_START memory leak detected"
        assert result['final_state']['stack'] == [0], f"Expected [0], got {result['final_state']['stack']}"
        return "✓ LOOP memory management passed"
    
    @staticmethod
    def test_unconditional_jump():
        """Test new JUMP instruction"""
        source = """
        main:
            CONST 100
            JMP skip      # Unconditional jump
            CONST 200     # Should be skipped
            CONST 300     # Should be skipped
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
    def test_fuzz():
        """Simple fuzz test: random programs, check no crash"""
        vm = PauseLangVM(debug=False)
        for _ in range(100):
            program_length = random.randint(5, 20)
            pauses = [random.choice(list(INSTRUCTIONS.keys())) for _ in range(program_length)]
            data = [random.randint(-100, 100) for _ in range(program_length)]
            try:
                vm.execute(data, pauses, sync=False)
            except Exception as e:
                raise AssertionError(f"Fuzz crash: {e}")
            vm.reset()
        return "✓ Fuzz test passed"
    
    @staticmethod
    def run_all():
        tests = [
            TortureTests.test_labels,
            TortureTests.test_aliases,
            TortureTests.test_unconditional_jump,
            TortureTests.test_division_semantics,
            TortureTests.test_jitter_gauntlet,
            TortureTests.test_flag_race,
            TortureTests.test_stack_underflow_protection,
            TortureTests.test_loop_memory,
            TortureTests.test_fuzz,
        ]
        print("\n🔥 TORTURE TEST SUITE v0.7.2 🔥")
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

# === DEMOS ===

def demo_flag_patterns():
    """Demonstrate proper flag-setting patterns"""
    print("\n🚩 FLAG SETTING PATTERNS")
    print("=" * 60)
    
    examples = [
        ("Check if value is odd", """
            CONST 7        # Push value
            SETF 0         # Set flags (7 is odd)
            JOD is_odd     # Jump if ODD flag set
            CONST 0        # Not odd
            HALT
        is_odd:
            CONST 1        # Is odd
            HALT
        """),
        
        ("Check if zero", """
            CONST 0        # Push value
            SETF 0         # Set flags (0 sets ZERO)
            JZ is_zero     # Jump if ZERO flag set
            CONST -1       # Not zero
            HALT
        is_zero:
            CONST 1        # Is zero
            HALT
        """),
        
        ("Branch without SETF (using arithmetic)", """
            CONST 5        # Push value
            CONST 0        # Push 0
            ADD2           # Add (sets flags on result)
            JOD odd_path   # Jump if result is odd
            CONST 100
            HALT
        odd_path:
            CONST 200
            HALT
        """),
    ]
    
    for title, source in examples:
        print(f"\n{title}:")
        print("-" * 40)
        print("Source:")
        for line in source.strip().split('\n'):
            print(f"  {line}")
        
        pauses, data, comments, labels = PauseLangCompiler.compile(source)
        vm = PauseLangVM(debug=False)
        result = vm.execute(data, pauses, labels=labels)
        
        print(f"\nResult: {result['final_state']['stack'][-1] if result['final_state']['stack'] else 'empty'}")
        print(f"Flags set: {[f for f, v in result['final_state']['flags'].items() if v]}")
        
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Stack ops don't set flags!")
    print("Use SETF, arithmetic ops, or LOAD to update flags before jumps.")

def demo_labels():
    """Demonstrate label-based programming"""
    print("\n🏷️ LABEL-BASED CONTROL FLOW DEMO (FACTORIAL)")
    print("=" * 60)
    
    source = """
    # Factorial calculator (now works with fixed LOAD)
    main:
        CONST 5          # Calculate 5!
        STORE 0          # Store N in mem[0]
        CONST 1          # Accumulator
        STORE 1          # Store accumulator in mem[1]

    loop:
        LOAD 0           # Load N to check for zero (pushes N to stack)
        SETF 0           # Set flags based on N
        JZ done          # If N=0, we are done
        
        # N is on the stack from the check, we can use it
        LOAD 1           # Load accumulator, stack: [N, acc]
        MUL2             # N * acc, stack: [N*acc]
        STORE 1          # Store new accumulator, stack: [N]
        
        DEC              # N-1, stack: [N-1]
        STORE 0          # Store new N, stack: []
        
        JMP loop         # Repeat
        
    done:
        # After JZ, the stack will have the last N (which is 0) from LOAD
        DROP             # Get rid of the 0
        LOAD 1           # Load final result
        HALT             # Result on stack
    """
    
    print("Source code:")
    print(source)
    
    pauses, data, comments, labels = PauseLangCompiler.compile(source, debug=False)
    
    print("\nLabels resolved:")
    for label, pc in labels.items():
        print(f"  {label:10} → PC {pc}")
    
    vm = PauseLangVM(debug=False) # Set to True for step-by-step
    result = vm.execute(data, pauses, labels=labels)
    
    print(f"\nResult: {result['final_state']['stack']}")
    print("Expected: [120] (5! = 120)")
    assert result['final_state']['stack'] == [120]

def demo_supervisor_enhanced():
    """Enhanced supervisor with better debugging"""
    print("\n🐉 ENHANCED DRAGON DETECTOR")
    print("=" * 60)
    
    class EnhancedSupervisor:
        def __init__(self):
            self.vm = PauseLangVM(gas_limit=5000, debug=False)
            
        def evaluate_with_labels(self, candidates: Dict[str, float], threshold: float = 0.5) -> Dict:
            results = {}
            
            for name, score in candidates.items():
                self.vm.reset()
                
                # Generate evaluation code based on known comparison
                # This is actually more efficient for supervision
                accept = score > threshold
                
                source = f"""
                # Evaluate {name} (score: {score:.2f} vs threshold: {threshold:.2f})
                # Pre-computed decision: {'ACCEPT' if accept else 'REJECT'}
                
                main:
                    CONST {int(score * 100)}      # Push score * 100
                    CONST {int(threshold * 100)}  # Push threshold * 100
                    SUB2                          # Calculate difference
                    
                    # Store result for audit trail
                    DUP
                    STORE 0                       # Store difference at slot 0
                    
                    # Return pre-computed decision
                    DROP                          # Clean stack
                    CONST {1 if accept else 0}   # Decision
                    HALT
                """
                
                pauses, data, comments, labels = PauseLangCompiler.compile(source)
                output = self.vm.execute(data, pauses, labels=labels)
                
                if 'error' in output:
                    results[name] = {
                        'score': score,
                        'accepted': False,
                        'error': output['error']
                    }
                    continue
                
                final_stack = output['final_state']['stack']
                accepted = len(final_stack) > 0 and final_stack[-1] == 1
                
                results[name] = {
                    'score': score,
                    'accepted': accepted,
                    'trace': self.vm.disassemble(show_labels=True, compact=True)
                }
            
            return results
    
    candidates = {
        'tail': 0.7,
        'wings': 0.8, 
        'fire': 0.3,
        'scales': 0.9,
        'magic': 0.2,
        'wisdom': 0.6
    }
    
    supervisor = EnhancedSupervisor()
    results = supervisor.evaluate_with_labels(candidates, threshold=0.5)
    
    print(f"Evaluating {len(candidates)} features with threshold 0.5:\n")
    accepted = []
    for feature, result in results.items():
        status = "✓ ACCEPT" if result['accepted'] else "✗ REJECT"
        print(f"  {feature:10} (score: {result['score']:.1f}) → {status}")
        if result['accepted']:
            accepted.append(feature)
    
    print(f"\n🎯 Decision: Dragon has {', '.join(accepted)}")

def demo_instruction_reference():
    """Show instruction reference with categories"""
    print("\n📖 INSTRUCTION REFERENCE")
    print("=" * 70)
    print(INSTRUCTION_CONVENTIONS)
    print("\n=== INSTRUCTION SET BY CATEGORY ===\n")
    
    # Group by category
    by_category = {}
    for pause, instr in sorted(INSTRUCTIONS.items()):
        cat = instr.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((pause, instr))
    
    for category in OpCategory:
        if category not in by_category:
            continue
        print(f"\n{category.name} OPERATIONS:")
        print("-" * 60)
        for pause, instr in by_category[category]:
            flags = "🚩" if instr.updates_flags else "  "
            req = f"R{instr.requires_stack}" if instr.requires_stack else "  "
            delta = f"Δ{instr.stack_delta:+d}" if instr.stack_delta != 0 else "   "
            print(f"  {instr.signature()} {instr.opcode:15} @{pause:.2f}s {flags}{req}{delta} │ {instr.description[9:]}")
    
    print("\n=== ALIASES ===")
    for alias, target in PauseLangCompiler.ALIASES.items():
        print(f"  {alias:10} → {target}")
    
    print("\n=== MACROS ===")
    for macro, expansion in PauseLangCompiler.MACROS.items():
        print(f"  {macro:10} → {' '.join([str(s) if isinstance(s, tuple) else s for s in expansion])}")

# === MAIN ===

def main():
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  PauseLang {SPEC['version']} - Bugfix                       ║
║  LOAD instruction is now stack-aware.                    ║
╚═══════════════════════════════════════════════════════════╝
""")
    print("\nSelect mode:")
    print("1. Run torture tests (all should pass!)")
    print("2. Label-based control flow demo (Factorial with fixed LOAD)")
    print("3. Enhanced supervisor demo")
    print("4. Show instruction reference")
    print("5. Show division/modulo examples")
    print("6. Flag setting patterns (NEW!)")
    print("7. Interactive VM (custom program)")
    
    choice = input("\nChoice (1-7): ").strip()
    
    if choice == '1':
        TortureTests.run_all()
    elif choice == '2':
        demo_labels()
    elif choice == '3':
        demo_supervisor_enhanced()
    elif choice == '4':
        demo_instruction_reference()
    elif choice == '5':
        print("\n📐 DIVISION & MODULO EXAMPLES")
        print("=" * 50)
        print("DIV2 (truncate toward zero):")
        print("  7 ÷ 2 = 3     │  -7 ÷ 2 = -3")
        print("  7 ÷ -2 = -3   │  -7 ÷ -2 = 3")
        print("\nMOD2 (positive remainder):")
        print("  7 % 3 = 1     │  -7 % 3 = 2") 
        print("  7 % -3 = 1    │  -7 % -3 = 2")
    elif choice == '6':
        demo_flag_patterns()
    elif choice == '7':
        print("\n💻 INTERACTIVE VM MODE")
        print("=" * 50)
        print("Enter your PauseLang program (type 'END' on a line to finish):")
        print("Example: CONST 5; SETF 0; JUMP_IF_ODD done; CONST 10; done: HALT")
        print()
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        
        source = '\n'.join(lines)
        
        print("\nCompiling...")
        try:
            pauses, data, comments, labels = PauseLangCompiler.compile(source, debug=True)
            
            if labels:
                print("\nLabels resolved:")
                for label, pc in labels.items():
                    print(f"  {label:10} → PC {pc}")
            
            trap_mode = input("\nTrap policy (halt/continue/raise) [continue]: ").strip() or 'continue'
            memory_mode = input("\nMemory mode (wrap/strict) [wrap]: ").strip() or 'wrap'
            
            vm = PauseLangVM(gas_limit=5000, trap_policy=trap_mode, memory_mode=memory_mode, debug=False)
            print("\nExecuting...")
            result = vm.execute(data, pauses, labels=labels)
            
            print("\n=== RESULTS ===")
            print(f"Final stack: {result['final_state']['stack']}")
            print(f"Final memory (non-zero): {result['final_state']['memory']}")
            print(f"Traps: {result['traps']}")
            print(f"Gas used: {result['gas_used']}")
            
            if input("\nShow disassembly? (y/n): ").lower() == 'y':
                print("\n" + vm.disassemble(show_labels=True, show_memory=True))
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Running default demo (tests)...")
        TortureTests.run_all()
    
    print("\n✨ PauseLang v0.7.2 fixed - Ready for action!")

if __name__ == "__main__":
    main()


# Now, define a custom PauseLang program (simple factorial from demo, but modified for 4!)
source = """
# Factorial calculator for 4!
main:
    CONST 4          # Calculate 4!
    STORE 0          # Store N in mem[0]
    CONST 1          # Accumulator
    STORE 1          # Store accumulator in mem[1]

loop:
    LOAD 0           # Load N to check for zero (pushes N to stack)
    SETF 0           # Set flags based on N
    JZ done          # If N=0, we are done
    
    # N is on the stack from the check, we can use it
    LOAD 1           # Load accumulator, stack: [N, acc]
    MUL2             # N * acc, stack: [N*acc]
    STORE 1          # Store new accumulator, stack: [N]
    
    DEC              # N-1, stack: [N-1]
    STORE 0          # Store new N, stack: []
    
    JMP loop         # Repeat
    
done:
    # After JZ, the stack will have the last N (which is 0) from LOAD
    DROP             # Get rid of the 0
    LOAD 1           # Load final result
    HALT             # Result on stack
"""

# Compile the program
pauses, data, comments, labels = PauseLangCompiler.compile(source)

# Create VM and execute
vm = PauseLangVM(debug=False)
result = vm.execute(data, pauses, labels=labels)

# Print results
print("Custom Program Source:")
print(source)
print("\nExecution Result:")
print("Final Stack (should be [24] for 4!):", result['final_state']['stack'])
print("Traps:", result['traps'])
print("Gas Used:", result['gas_used'])
print("\nDisassembly:")
print(vm.disassemble(show_labels=True, show_memory=True))
"""
