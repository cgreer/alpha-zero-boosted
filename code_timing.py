from contextlib import contextmanager
from dataclasses import dataclass
import time
import random
from copy import deepcopy, copy
import numpy

BASE_ITERATIONS = 1_000_000
BASE_ITERATIONS = 100_000
operation_info = []


@dataclass
class Operation:
    name: str
    num_interations: int
    elapsed: float = -1.0
    operations_per_second: float = -1.0


@contextmanager
def time_operation(name, num_interations):
    operation = Operation(name, num_interations)

    st_time = time.time()
    yield operation
    operation.elapsed = time.time() - st_time

    operation.operations_per_second = BASE_ITERATIONS / operation.elapsed
    operation_info.append((operation.name, operation.operations_per_second))


def display_results(message, ops_per_sec):
    time_per_op = 1.0 / ops_per_sec
    ops_per_sec = "{:,.1f}".format(ops_per_sec)
    print("{:<90}{:>30}{:>30}".format(message, ops_per_sec, time_per_op))


# Warm up
x = 0
for i in range(BASE_ITERATIONS):
    x += 1

op = "dirichlet.rvs() (scipy frozen distribution, size 9)"
import scipy.stats # noqa
d = scipy.stats.dirichlet([.2] * 9)
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        d.rvs()

from agents import NOISE_MAKER # noqa
op = "NOISE_MAKER.make_noise(.2, 10)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        NOISE_MAKER.make_noise(.2, 10)

op = "random.randint(0, 9999)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        random.randint(0, 9999)

op = "[0.0 for x in agents]"
agents = [0, 1]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        s = [0.0 for x in agents]

op = "[0.0] * len(agents)"
agents = [0, 1]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        s = [0.0] * len(agents)

op = "tuple(generator) (10)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = tuple(x for x in range(10))

op = "tuple(appended list) (10)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        s = []
        for i in range(10):
            s.append(i)
        a = tuple(s)

op = "enumerate(s) (100)"
s = list(range(100))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        for i, val in enumerate(s):
            pass

op = "manual enumerate(s) (100)"
s = list(range(100))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        i = 0
        for val in s:
            i += 1
            pass

op = "try/except IndexError:"
s = [0]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        try:
            a = s[1]
        except IndexError:
            pass

import numpy # noqa
op = "numpy.random.dirichlet (size 9)"
noise_alpha = .2
num_child_edges = 9
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        noise = numpy.random.dirichlet([noise_alpha] * num_child_edges)

import math # noqa
op = "math.sqrt"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        math.sqrt(42)

op = "a = s[500] (tuple, 1000)"
s = tuple(range(1000))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = s[500]

'''
op = "quoridor full path"
from quoridor import victory_distance_2 as victory_distance, victory_path # noqa
blocked_passages = set()
blocked_passages.add((4, 1, 4, 2))
blocked_passages.add((4, 2, 4, 1))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        initial_x = 4
        initial_y = 0
        distance, final_x, final_y, visited = victory_distance(initial_x, initial_y, blocked_passages, 8)
        vic_path = victory_path(initial_x, initial_y, final_x, final_y, visited)
'''

s = tuple(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = s[90]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = s[90] (tuple)", ops_per_sec))

s = list(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = s[90]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = s[90] (list)", ops_per_sec))

s = set((0, 1) for x in range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = (0, 0) in s
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = (0, 0) in s (set)", ops_per_sec))

s = 0
st_time = time.time()
for i in range(BASE_ITERATIONS):
    s += 1
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append((
    "s += 1",
    ops_per_sec,
))

s = list(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    s[90] = 1
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("s[90] = 1 (list, 180)", ops_per_sec))

st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = [True] * 81
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = [True] * 91", ops_per_sec))

st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = [True for _ in range(81)]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = [True for _ in range(81)]", ops_per_sec))

x = 16
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = x % 9
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = x % 9", ops_per_sec))

s = set(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    s.add(90)
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("s.add(90)", ops_per_sec))

x = 0
y = 1
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = min(x, y)
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("min(x, y)", ops_per_sec))

x = 0
y = 1
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = x if x < y else y
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("x if x < y else y", ops_per_sec))

s = list(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = s[:]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = s[:] (180)", ops_per_sec))

s = list(range(90))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = s[:]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = s[:] (90)", ops_per_sec))

st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = i
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = i", ops_per_sec))

s = list(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = copy(s)
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = copy(s)", ops_per_sec))

s = list(range(180))
st_time = time.time()
for i in range(BASE_ITERATIONS // 10):
    a = list(x for x in s)
elapsed = time.time() - st_time
ops_per_sec = (BASE_ITERATIONS // 10) / elapsed
operation_info.append(("a = list(x for x in s)", ops_per_sec))


def foo():
    pass


st_time = time.time()
for i in range(BASE_ITERATIONS):
    foo()
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("foo()", ops_per_sec))

visited = [[False for x in range(9)] for y in range(9)]
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = visited[5][5]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = visited[5][5] (9x9)", ops_per_sec))

x = 5
y = 5
visited = [False] * 81
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = visited[9 * y + x]
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = visited[9 * y + x] (81)", ops_per_sec))

s = [1] * 10
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = len(s)
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("a = len(s) (list, 10)", ops_per_sec))

s = [1] * 10
st_time = time.time()
for i in range(BASE_ITERATIONS):
    a = s.pop()
    s.append(a)
elapsed = time.time() - st_time
ops_per_sec = BASE_ITERATIONS / elapsed
operation_info.append(("s.pop, s.append (list, 10)", ops_per_sec))

op = "if s > 0"
s = 0
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        if s > 0:
            pass

op = "if s == 0"
s = 0
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        if s == 0:
            pass

op = "[x[:] for x in s] (s is [6][7]int)"
s = [[0] * 6 for _ in range(7)]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = [x[:] for x in s]

op = "for x in s: append(x[:]) (s is [6][7]int)"
s = [[0] * 6 for _ in range(7)]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = []
        for x in s:
            a.append(x[:])

op = "[copy(x) for x in s] (s is [6][7]int)"
s = [[0] * 6 for _ in range(7)]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = [copy(x) for x in s]

op = "deepcopy(s) (s is [6][7]int)"
s = [[0] * 6 for _ in range(7)]
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a = deepcopy(s)

op = "for x in (0, 1, 2, 3, 4, 5, 6)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        for x in (0, 1, 2, 3, 4, 5, 6):
            pass

op = "for x in range(7)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        for x in range(7):
            pass

op = "numpy.zeros(84, dtype=numpy.float32)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        numpy.zeros(84, dtype=numpy.float32)

s = numpy.zeros(84, dtype=numpy.float32)
op = "s[40] = 1.0 (numpy float32 array)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        s[40] = 1.0

s = [0] * 84
op = "numpy.array([s], dtype=numpy.float32) (s is list, 84 ints)"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        numpy.array([s], dtype=numpy.float32)

from treelite.runtime import ( # noqa
    Batch as TreeliteBatch,
)
features = [0] * 87 # size of connect feature array
features = [features, features]
op = "TreeliteBatch.from_npy2d(numpy.array(features, dtype=numpy.float32))"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        TreeliteBatch.from_npy2d(numpy.array(features, dtype=numpy.float32))

features = [0] * 87 # size of connect feature array
features = [features, features]
op = "TreeliteBatch.from_npy2d(numpy.asarray(features, dtype=numpy.float32))"
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        TreeliteBatch.from_npy2d(numpy.asarray(features, dtype=numpy.float32))

op = "b = actions / actions.sum()"
actions = numpy.array([.2, .3, .1, 0.5, .1, .2, .2], dtype=numpy.float32)
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        b = actions / actions.sum()

op = "b = actions / s (manual sum)"
actions = numpy.array([.2, .3, .1, 0.5, .1, .2, .2], dtype=numpy.float32)
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        s = 0.0
        for x in actions:
            s += x
        b = actions / s

op = "a * (1 - .25) (a is numpy.float32)"
a = numpy.float32(0.6)
b = numpy.float32(1.0)
c = numpy.float32(0.25)
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a * (1 - .25)

op = "a * (1 - .25) (a is python float)"
a = 0.6
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a * (1 - .25)

op = "a * (b - c) (a,b,c are numpy.float32)"
a = numpy.float32(0.6)
b = numpy.float32(1.0)
c = numpy.float32(0.25)
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a * (b - c)

op = "a * (b - c) (a,b,c are python floats)"
a = 0.6
b = 1.0
c = 0.25
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        a * (b - c)

op = "b = copy.copy(s)"
s = set((x, 1, 2, 3) for x in range(180))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        b = copy(s)

op = "b = set(s)"
s = set((x, 1, 2, 3) for x in range(180))
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        b = set(s)

op = "if s: (s is {})"
s = {}
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        if s:
            pass

op = "if s: (s is [])"
s = []
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        if s:
            pass

op = "if s: (s is None)"
s = None
with time_operation(op, BASE_ITERATIONS) as op:
    for i in range(op.num_interations):
        if s:
            pass


###############
# Display all results
###############
# operation_info.sort(key=lambda x: x[1], reverse=True)
for op, ops_per_sec in operation_info:
    display_results(op, ops_per_sec)
