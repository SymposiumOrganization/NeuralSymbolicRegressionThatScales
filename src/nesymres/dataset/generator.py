# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import re
import sys
import math
import itertools
from collections import OrderedDict
import numpy as np
import numexpr as ne
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.integrals.risch import NonElementaryIntegral
from sympy.calculus.util import AccumBounds
from sympy.core.rules import Transform
from sympy import sympify, Symbol
from sympy import Float
from random import random
from .sympy_utils import (
    remove_root_constant_terms,
    reduce_coefficients,
    reindex_coefficients,
)
from .sympy_utils import (
    extract_non_constant_subtree,
    simplify_const_with_coeff,
    simplify_equa_diff,
    clean_degree2_solution,
)
from .sympy_utils import remove_mul_const, has_inf_nan, has_I, simplify


CLEAR_SYMPY_CACHE_FREQ = 10000




class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace("Derivative(f(x),x)", "1")
    s = s.replace("Derivative(1,x)", "1")
    s = s.replace("(E)", "(exp(1))")
    s = s.replace("(I)", "(1)")
    s = s.replace("(pi)", "(1)")
    s = re.sub(
        r"(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(",
        "(",
        s,
    )
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i) for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False


def eval_test_zero(eq):
    """
    Evaluate an equation by replacing all its free symbols with random values.
    """
    variables = eq.free_symbols
    assert len(variables) <= 3
    outputs = []
    for values in itertools.product(*[TEST_ZERO_VALUES for _ in range(len(variables))]):
        _eq = eq.subs(zip(variables, values)).doit()
        outputs.append(float(sp.Abs(_eq.evalf())))
    return outputs


class Generator(object):

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",

        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",

        # Trigonometric Inverses
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",

        # Hyperbolic Functions
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",

    }

    OPERATORS = {
        # Elementary functions
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "pow": 2,
        #'rac': 2,
        "inv": 1,
        "pow2": 1,
        "pow3": 1,
        "pow4": 1,
        "pow5": 1,
        "sqrt": 1,
        "exp": 1,
        "ln": 1,

        # Trigonometric Functions
        "sin": 1,
        "cos": 1,
        "tan": 1,

        # Trigonometric Inverses
        "asin": 1,
        "acos": 1,
        "atan": 1,

        # Hyperbolic Functions
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        "coth": 1,
    }

    def __init__(self, params):
        self.max_ops = params.max_ops
        self.int_base = params.int_base
        self.precision = params.precision
        self.variables = params.variables
        self.n_coefficients = params.n_coefficients
        self.max_len = params.max_len
        #assert self.max_int >= 1
        assert abs(self.int_base) >= 2
        assert self.precision >= 2

        # parse operators with their weights
        self.operators = sorted(list(self.OPERATORS.keys()))
        ops = params.operators.split(",")
        ops = sorted([x.split(":") for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 1]
        ).astype(np.float64)
        self.bin_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 2]
        ).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.constants = ["pi", "E"]
        self.variables = OrderedDict(
            {
                "x": sp.Symbol("x", real=True, nonzero=True),  # , positive=True
                "y": sp.Symbol("y", real=True, nonzero=True),  # , positive=True
                "z": sp.Symbol("z", real=True, nonzero=True),
                "c": sp.Symbol("c", real=True, nonzero=True),
            }
        )
        # , positive=True
        #'p': sp.Symbol('p', real=True, nonzero=True),  # , positive=True
        #'t': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        # })
        self.coefficients = OrderedDict({})

        self.symbols = [
            "I",
            "INT+",
            "INT-",
            "INT",
            "FLOAT",
            "-",
            ".",
            "10^",
            "Y",
            "Y'",
            "Y''",
        ]
        # if self.balanced:
        #     assert self.int_base > 2
        #     max_digit = (self.int_base + 1) // 2
        #     self.elements = [str(i) for i in range(max_digit - abs(self.int_base), max_digit)]
        # else:
        #     self.elements = [str(i) for i in range(-5,abs(self.int_base))]
        assert 1 <= len(self.variables)
        # assert 0 <= self.n_coefficients <= len(self.coefficients)
        # assert all(k in self.OPERATORS for k in self.functions.keys())
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(
            self.variables.items()
        ):  # + list(self.coefficients.items()): #+ list(self.functions.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # vocabulary
        # import pdb
        # pdb.set_trace()
        # self.words = self.constants + list(self.variables.keys()) + self.operators  + self.elements #list(self.coefficients.keys()) +SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.coefficients.keys()) + self.operators + self.symbols + self.elements
        digits = [str(i) for i in range(-3, abs(6))]
        self.words = (
            list(self.variables.keys())
            + [
                x
                for x in self.operators
                if x not in ("pow2", "pow3", "pow4", "pow5", "sub", "inv")
            ]
            + digits
        )  # + self.elements
        # import pdb
        # pdb.set_trace()
        self.id2word = {i: s for i, s in enumerate(self.words, 3)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        # ADD Start and Finish
        self.word2id["P"] = 0
        self.word2id["S"] = 1
        self.word2id["F"] = 2
        self.id2word[1] = "S"
        self.id2word[2] = "F"
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)


        # leaf probabilities
        s = [float(x) for x in params.leaf_probs.split(",")]
        assert len(s) == 4 and all(x >= 0 for x in s)
        self.leaf_probs = np.array(s).astype(np.float64)
        self.leaf_probs = self.leaf_probs / self.leaf_probs.sum()
        assert self.leaf_probs[0] > 0
        assert (self.leaf_probs[1] == 0) == (self.n_coefficients == 0)

        # possible leaves
        self.n_leaves = len(self.variables) + self.n_coefficients
        if self.leaf_probs[2] > 0:
            self.n_leaves += 2 #-1, 1
        if self.leaf_probs[3] > 0:
            self.n_leaves += len(self.constants)

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = [
            x for x in params.rewrite_functions.split(",") if x != ""
        ]
        assert len(self.rewrite_functions) == len(set(self.rewrite_functions))
        assert all(
            x in ["expand", "factor", "expand_log", "logcombine", "powsimp", "simplify"]
            for x in self.rewrite_functions
        )

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = self.int_base
        balanced = self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append("INT")
        else:
            res.append("INT-" if neg else "INT+")
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (
            balanced
            and lst[0] == "INT"
            or base >= 2
            and lst[0] in ["INT+", "INT-"]
            or base <= -2
            and lst[0] == "INT"
        ):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == "-" and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == "INT-":
            val = -val
        return val, i + 1

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, max_int, rng):
        """
        Generate a leaf.
        """
        self.leaf_probs
        leaf_type = rng.choice(4, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.variables.keys())[rng.randint(self.n_variables)]]
        elif leaf_type == 1:
            return [list(self.coefficients.keys())[rng.randint(self.n_coefficients)]]
        elif leaf_type == 2:
            c = rng.randint(1, max_int + 1)
            c = c if (self.positive or rng.randint(2) == 0) else -c
            return self.write_int(c)
        else:
            return [self.constants[rng.randint(len(self.constants))]]

    def _generate_expr(
        self,
        nb_total_ops,
        max_int,
        rng,
        require_x=False,
        require_y=False,
        require_z=False,
    ):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += (
                self.OPERATORS[op] - 1 - skipped
            )  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [None for _ in range(self.OPERATORS[op])]
                + stack[pos + 1 :]
            )

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        # optionally add variables x, y, z if possible
        assert not require_z or require_y
        assert not require_y or require_x
        leaves = [self.get_leaf(max_int, rng) for _ in range(t_leaves)]
        if require_z and t_leaves >= 2:
            leaves[1] = ["z"]
        if require_y:
            leaves[0] = ["y"]
        if require_x and not any(len(leaf) == 1 and leaf[0] == "x" for leaf in leaves):
            leaves[-1] = ["x"]
        rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1 :]
        assert len(leaves) == 0

        return stack
        
    def add_contants(self,pred_str):
        temp = self.sympy_to_prefix(sympify(pred_str))
        temp2 = self._prefix_to_infix_with_constants(temp)[0]
        # num = self.count_number_of_constants(temp2)
        # costs = [random() for x in range(num)]
        # example = temp2.format(*tuple(costs))
        # pred_str = str(self.constants_to_placeholder(example))
        # c=0
        # expre = list(pred_str)
        # breakpoint()
        # for j,i in enumerate(list(pred_str)):
        #     try:
        #         if i == 'c' and list(pred_str)[j+1] != 'o':
        #             expre[j] = 'c{}'.format(str(c))
        #             c=c+1
        #     except IndexError:
        #         if i == 'c':
        #             expre[j] = 'c{}'.format(str(c))
        #             c=c+1        
        # example = "".join(list(expre))
        return temp2

    def tokenize(self, prefix_expr):
        tokenized_expr = []
        tokenized_expr.append(self.word2id["S"])
        for i in prefix_expr:
            # try:
            tokenized_expr.append(self.word2id[i])
            # except:
            # breakpoint()
            # print("Exception with {} in Tokenization".format(prefix_expr))
            # return None
        tokenized_expr.append(self.word2id["F"])
        return tokenized_expr

    def de_tokenize(self, tokenized_expr):
        prefix_expr = []
        for i in tokenized_expr:
            if i == self.word2id["F"]:
                break
            else:
                prefix_expr.append(self.id2word[i])
        return prefix_expr

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        elif token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "rac":
            return f"({args[0]})**(1/({args[1]}))"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token == "pow4":
            return f"({args[0]})**4"
        elif token == "pow5":
            return f"({args[0]})**5"
        elif token in [
            "sign",
            "sqrt",
            "exp",
            "ln",
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "asin",
            "acos",
            "atan",
            "acot",
            "asec",
            "acsc",
            "sinh",
            "cosh",
            "tanh",
            "coth",
            "sech",
            "csch",
            "asinh",
            "acosh",
            "atanh",
            "acoth",
            "asech",
            "acsch",
        ]:
            return f"{token}({args[0]})"
        elif token == "derivative":
            return f"Derivative({args[0]},{args[1]})"
        elif token == "f":
            return f"f({args[0]})"
        elif token == "g":
            return f"g({args[0]},{args[1]})"
        elif token == "h":
            return f"h({args[0]},{args[1]},{args[2]})"
        elif token.startswith("INT"):
            return f"{token[-1]}{args[0]}"
        else:
            return token
        raise InvalidPrefixExpression(
            f"Unknown token in prefix expression: {token}, with arguments {args}"
        )

    def _prefix_to_infix_with_constants(self, expr, is_const=1):
        """
        Return string with constants
        """
        if not expr or len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for i in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix_with_constants(
                    l1, is_const and not (t == "pow" and i > 0)
                )
                args.append(i1)
            if self.OPERATORS[t] == 1:
                return ["", "{}*"][is_const] + self.write_infix(t, args), l1
            else:
                return self.write_infix(t, args), l1
        elif t in self.variables:
            return ["", "{}*"][is_const] + t, expr[1:]
        elif t in self.coefficients or t in self.constants or t == "I":
            return t, expr[1:]
        else:
            val = int(expr[0])
            # sign = lambda x: (1, -1)[x < 0]
            return [val, self.sign(val) + "{}"][is_const], expr[1:]
            # val, i = self.parse_int(expr)
            # return str(val), expr[i:]

    def sign(self, x):
        return ("", "-")[x < 0]

    def _prefix_to_infix_benchmark(self, expr):
        """
        Return string with constants
        """
        # breakpoint()
        if not expr or len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix_benchmark(l1)
                args.append(i1)
            if self.OPERATORS[t] == 1:
                return self.write_infix(t, args), l1
            else:
                return self.write_infix(t, args), l1
        elif t in self.variables:
            return t, expr[1:]
        elif t in self.coefficients or t in self.constants or t == "I":
            return t, expr[1:]
        else:
            val = int(expr[0])
            return val, expr[1:]
            # val, i = self.parse_int(expr)
            # return str(val), expr[i:]

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):  # Arity
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif (
            t in self.variables
            or t in self.coefficients
            or t in self.constants
            or t == "I"
        ):
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def _prefix_to_edges(self, expr):
        t = expr[0][1]
        edges = []
        li = expr[1:]
        if t in self.operators:
            args = []
            for _ in range(self.OPERATORS[t]):
                new_edge = [expr[0][0], li[0][0]]
                edges.append(new_edge)
                inner_edges, li = self._prefix_to_edges(li)
                edges.extend(inner_edges)
        return edges, li

    def prefix_to_infix_with_constants(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix_with_constants(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"

    def rewrite_sympy_expr(self, expr):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in self.rewrite_functions:
            if f == "expand":
                expr_rw = sp.expand(expr_rw)
            elif f == "factor":
                expr_rw = sp.factor(expr_rw)
            elif f == "expand_log":
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == "logcombine":
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == "powsimp":
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == "simplify":
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    def infix_to_sympy(self, infix, no_rewrite=False, check_if_valid=True):
        """
        Convert an infix expression to SymPy.
        """
        if check_if_valid:
            if not is_valid_expr(infix):
                raise ValueErrorExpression
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = self.rewrite_sympy_expr(expr)
        return expr

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        if op == "derivative":
            assert n_args >= 2
            assert all(
                len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1
                for arg in expr.args[1:]
            ), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = (
                    ["derivative" for _ in range(int(degree))]
                    + parse_list
                    + [str(var) for _ in range(int(degree))]
                )
            return parse_list

        assert (
            (op == "add" or op == "mul")
            and (n_args >= 2)
            or (op != "add" and op != "mul")
            and (1 <= n_args <= 2)
        )

        # square root
        if (
            op == "pow"
            and isinstance(expr.args[1], sp.Rational)
            and expr.args[1].p == 1
            and expr.args[1].q == 2
        ):
            return ["sqrt"] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]  # self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return (
                ["div"] + [str(expr.p)] + [str(expr.q)]
            )  # self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            return ["I"]
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # environment function
        for func_name, func in self.functions.items():
            if isinstance(expr, func):
                return self._sympy_to_prefix(func_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def reduce_coefficients(self, expr):
        return reduce_coefficients(
            expr, self.variables.values(), self.coefficients.values()
        )

    def reindex_coefficients(self, expr):
        if self.n_coefficients == 0:
            return expr
        return reindex_coefficients(
            expr, list(self.coefficients.values())[: self.n_coefficients]
        )

    def extract_non_constant_subtree(self, expr):
        return extract_non_constant_subtree(expr, self.variables.values())

    def simplify_const_with_coeff(self, expr, coeffs=None):
        if coeffs is None:
            coeffs = self.coefficients.values()
        for coeff in coeffs:
            expr = simplify_const_with_coeff(expr, coeff)
        return expr

    # @timeout(3)
    @staticmethod
    def count_number_of_constants(format_string):
        return len(re.findall(r"({})", format_string))

    def process_equation(self, infix, check_if_valid=True):
        f = self.infix_to_sympy(infix, check_if_valid=check_if_valid)
        # skip constant expressions
        if self.variables["x"] not in f.free_symbols:
            return None, "X not in free symbols"
        # remove additive constant, re-index coefficients
        # if rng.randint(2) == 0:
        f = extract_non_constant_subtree(f, list(self.variables.values()))
        # f = remove_root_constant_terms(f, list(self.variables.values()), 'add')
        # f = remove_root_constant_terms(f, list(self.variables.values()), 'mul')
        f = self.reduce_coefficients(f)
        f = self.simplify_const_with_coeff(f)
        f = self.reindex_coefficients(f)

        # skip invalid expressions
        if has_inf_nan(f):
            return None, "There are nans"

        return f

    def generate_equation(self, rng):
        """
        Generate pairs of (function, primitive).
        Start by generating a random function f, and use SymPy to compute F.
        """
        x = self.variables["x"]
        if rng.randint(40) == 0:
            nb_ops = rng.randint(0, 3)
        else:
            nb_ops = rng.randint(3, self.max_ops + 1)


        # try:
        # generate an expression and rewrite it,
        # avoid issues in 0 and convert to SymPy
        f_expr = self._generate_expr(nb_ops, self.max_int, rng)

        infix = self.prefix_to_infix(f_expr)
        f = self.process_equation(infix)
        f_prefix = self.sympy_to_prefix(f)
        # skip too long sequences
        if len(f_expr) + 2 > self.max_len:
            return None, "Sequence longer than max length"

        # skip when the number of operators is too far from expected
        real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in f_expr)
        if real_nb_ops < nb_ops / 2:
            return None, "Too many operators"

        return f_prefix, f


    def constants_to_placeholder(self, s):
        try:
            sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
            sympy_expr = sympy_expr.xreplace(
                Transform(
                    lambda x: Symbol("c", real=True, nonzero=True),
                    lambda x: isinstance(x, Float),
                )
            )
        except:
            breakpoint()
        return sympy_expr






