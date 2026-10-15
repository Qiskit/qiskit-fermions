# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ternary-tree bindings.

The encoding mathematics is tested in the ``core`` Rust crate: leg counts, the Bonsai pairing, Pauli
weights, agreement with Jordan-Wigner on a chain, the Majorana route for the other three algebras,
total parity, mode maps and padding all have Rust tests. Re-deriving any of that here would test the
same code twice through a longer path, so this module tests the **Python boundary** instead: that
arguments cross it in the right form, that results come back as usable
:class:`~qiskit.quantum_info.SparseObservable` objects, that Rust errors surface as Python exceptions,
and that the objects behave like Python objects.
"""

from __future__ import annotations

import importlib
import pickle
import sys

import pytest
from qiskit.quantum_info import SparseObservable
from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
from qiskit_fermions.mappers.library import (
    edge_vertex_jordan_wigner,
    edge_vertex_ternary_tree,
    fermion_jordan_wigner,
    fermion_ternary_tree,
    majorana_jordan_wigner,
    majorana_ternary_tree,
    ternary_tree,
    transfer_vertex_jordan_wigner,
    transfer_vertex_ternary_tree,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)


def assert_obs_equal(got: SparseObservable, expected: SparseObservable) -> None:
    """Asserts two observables represent the same operator."""
    diff = (got - expected).simplify()
    assert diff == SparseObservable.zero(got.num_qubits)


def fermion_op() -> FermionOperator:
    """A four-mode operator with one-, two- and constant terms."""
    return FermionOperator.from_dict(
        {
            (): 2.0,
            ((True, 0), (False, 0)): 0.1,
            ((True, 1), (False, 2), (True, 2), (False, 1)): -1.0j,
            ((True, 0), (True, 3), (False, 3), (False, 1)): 0.25 + 0.5j,
        }
    )


def majorana_op() -> MajoranaOperator:
    return MajoranaOperator.from_dict({(0, 3): 0.5, (2, 5): 0.25j, (4, 1, 6, 7): -1.5})


def edge_op() -> EdgeVertexOperator:
    return EdgeVertexOperator.from_dict(
        {((0, 0),): 1.0, ((1, 2), (2, 1)): 0.5j, ((0, 3), (3, 1)): -0.75}
    )


def transfer_op() -> TransferVertexOperator:
    return TransferVertexOperator.from_dict(
        {((1, 1),): 1.0, ((0, 2), (2, 0)): 0.5j, ((3, 1), (1, 3)): -0.75}
    )


class TestMapperBindings:
    """Each mapper must reach its Rust implementation and return a usable observable.

    The Jordan-Wigner tree is used as the reference because its expected result is already pinned by
    an independently tested mapper on the Python side. A binding that dropped an argument, mismatched
    a type or mapped the wrong operator would fail here; the encoding mathematics itself is covered in
    Rust.
    """

    def test_fermion(self):
        op = fermion_op()
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert_obs_equal(fermion_ternary_tree(op, enc, 4), fermion_jordan_wigner(op, 4))

    def test_majorana(self):
        op = majorana_op()
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert_obs_equal(majorana_ternary_tree(op, enc, 4), majorana_jordan_wigner(op, 4))

    def test_edge_vertex(self):
        op = edge_op()
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert_obs_equal(edge_vertex_ternary_tree(op, enc, 4), edge_vertex_jordan_wigner(op, 4))

    def test_transfer_vertex(self):
        op = transfer_op()
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert_obs_equal(
            transfer_vertex_ternary_tree(op, enc, 4), transfer_vertex_jordan_wigner(op, 4)
        )

    def test_returns_a_real_sparse_observable(self):
        """The result must be a genuine ``SparseObservable``, not an opaque handle."""
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        obs = fermion_ternary_tree(fermion_op(), enc, 4)
        assert isinstance(obs, SparseObservable)
        assert obs.num_qubits == 4
        # Usable in Qiskit's own arithmetic and simplification.
        assert (obs - obs).simplify() == SparseObservable.zero(4)

    @pytest.mark.parametrize("num_qubits", [4, 6, 9])
    def test_num_qubits_argument_is_honoured(self, num_qubits):
        """``num_qubits`` crosses the boundary and widens the result, padding with the identity."""
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        obs = fermion_ternary_tree(fermion_op(), enc, num_qubits)
        assert obs.num_qubits == num_qubits
        assert_obs_equal(obs, fermion_jordan_wigner(fermion_op(), num_qubits))


class TestTreeBindings:
    """Constructors, accessors and label handling across the boundary."""

    def test_general_constructor_takes_a_spec(self):
        """The ``(parent, label)`` specification is the general interface, and crosses as a list."""
        # A mixed-branching shape neither convenience constructor can produce.
        tree = TernaryTree([None, (0, "X"), (0, "Y"), (0, "Z"), (1, "X")])
        assert tree.num_nodes == 5
        assert tree.num_legs == 2 * 5 + 1

    def test_chain_and_breadth_first(self):
        assert TernaryTree.chain(4, "Z").num_nodes == 4
        assert TernaryTree.breadth_first(13, 3).num_nodes == 13

    @pytest.mark.parametrize("label", ["X", "x", "Y", "y", "Z", "z"])
    def test_labels_accept_either_case(self, label):
        """Labels cross as single characters, and the parser accepts both cases."""
        assert TernaryTree.chain(3, label).num_nodes == 3

    def test_root_and_children(self):
        tree = TernaryTree([None, (0, "X"), (0, "Z")])
        assert tree.root == 0
        assert sorted(tree.children(0)) == [("X", 1), ("Z", 2)]
        assert tree.children(1) == []

    def test_leg_paths_round_trip_as_tuples(self):
        """Leg paths come back as ``(qubit, label)`` tuples, with labels as characters."""
        paths = TernaryTree.chain(2, "Z").leg_paths()
        assert paths == [
            [(0, "X")],
            [(0, "Y")],
            [(1, "X"), (0, "Z")],
            [(1, "Y"), (0, "Z")],
            [(1, "Z"), (0, "Z")],
        ]

    def test_max_weight_is_exposed(self):
        assert TernaryTree.chain(5, "Z").max_weight == 5
        assert TernaryTree.breadth_first(13, 3).max_weight == 3

    def test_repr(self):
        assert "TernaryTree" in repr(TernaryTree.chain(3, "Z"))


class TestEncodingBindings:
    """The compiled encoding's accessors and its mode-map argument."""

    def test_attributes(self):
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        assert enc.num_modes == 4
        assert enc.max_weight == 2

    def test_majorana_image_round_trips_as_tuples(self):
        """The Pauli image comes back as ascending ``(qubit, label)`` tuples."""
        enc = TernaryTreeEncoding(TernaryTree.chain(3, "Z"))
        assert enc.majorana_image(4) == [(0, "Z"), (1, "Z"), (2, "X")]

    def test_mode_map_argument_crosses(self):
        """A ``mode_map`` list reaches Rust and changes which mode each node serves."""
        tree = TernaryTree.breadth_first(4, 3)
        identity = TernaryTreeEncoding(tree)
        reversed_map = TernaryTreeEncoding(tree, mode_map=[3, 2, 1, 0])
        # Node u serves mode 3-u under the reversed map, so the images swap accordingly.
        for node in range(4):
            for offset in range(2):
                assert identity.majorana_image(2 * node + offset) == reversed_map.majorana_image(
                    2 * (3 - node) + offset
                )

    def test_default_mode_map_is_the_identity(self):
        tree = TernaryTree.breadth_first(4, 3)
        assert TernaryTreeEncoding(tree).majorana_image(0) == TernaryTreeEncoding(
            tree, mode_map=[0, 1, 2, 3]
        ).majorana_image(0)

    def test_total_parity_is_a_method_on_the_encoding(self):
        """Parity is derived from the encoding, so it is a method rather than a library function."""
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert_obs_equal(
            enc.total_parity(4),
            SparseObservable.from_sparse_list([("ZZZZ", [0, 1, 2, 3], 1.0)], 4),
        )
        # On the balanced tree the leftover leg is not the all-Z string.
        balanced = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        assert_obs_equal(
            balanced.total_parity(4),
            SparseObservable.from_sparse_list([("ZZ", [0, 3], 1.0)], 4),
        )
        # `num_qubits` pads with the identity, as it does for the mappers.
        assert enc.total_parity(7).num_qubits == 7

    def test_one_encoding_serves_several_widths(self):
        """The encoding carries no qubit count, so it can be reused across registers."""
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        assert fermion_ternary_tree(fermion_op(), enc, 4).num_qubits == 4
        assert fermion_ternary_tree(fermion_op(), enc, 7).num_qubits == 7

    def test_repr(self):
        assert "TernaryTreeEncoding" in repr(TernaryTreeEncoding(TernaryTree.chain(3, "Z")))


class TestPythonObjectBehaviour:
    """The pyclasses must behave like Python objects, not bare handles."""

    @pytest.mark.parametrize("cls", [TernaryTree, TernaryTreeEncoding])
    def test_logical_module_path_resolves(self, cls):
        """The public ``__module__`` must name a module that actually exists.

        A native pyclass declares its logical dotted path for ``repr`` and error messages, which is not
        the physical ``_lib``-rooted path it is importable from. ``qiskit_fermions/__init__.py`` aliases
        the two, and anything resolving a class by name (``importlib``, ``pickle``, Sphinx) depends on
        that alias being in place.
        """
        assert cls.__module__ in sys.modules
        assert importlib.import_module(cls.__module__) is not None

    @pytest.mark.parametrize(
        "tree",
        [
            TernaryTree.chain(5, "Z"),
            TernaryTree.chain(4, "X"),
            TernaryTree.breadth_first(13, 3),
            TernaryTree.breadth_first(7, 2, "ZXY"),
            # An irregular shape neither convenience constructor produces.
            TernaryTree([None, (0, "X"), (0, "Y"), (0, "Z"), (1, "X"), (2, "Z")]),
        ],
        ids=["chain", "parity", "balanced", "binary", "irregular"],
    )
    def test_tree_is_pickleable(self, tree):
        """``__getnewargs__`` must hand back a specification that rebuilds the same tree."""
        restored = pickle.loads(pickle.dumps(tree))
        assert restored.num_nodes == tree.num_nodes
        assert restored.num_legs == tree.num_legs
        assert restored.root == tree.root
        assert restored.max_weight == tree.max_weight
        # The leg paths are the encoding-defining content, so compare those rather than the summary.
        assert restored.leg_paths() == tree.leg_paths()

    def test_encoding_is_not_pickleable(self):
        """The encoding has no pickle protocol, by design.

        It is a derived artifact: compiling discards the tree and mode map to build the Pauli tables, so
        supporting pickle would mean retaining those inputs purely to serialize them. The tree is what a
        search produces and what defines the encoding, so that is what round-trips; callers recompile.
        """
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(enc)

    def test_recompiling_a_pickled_tree_reproduces_the_encoding(self):
        """Pickling the tree is sufficient: recompiling it gives an equivalent encoding."""
        tree = TernaryTree.breadth_first(4, 3)
        mode_map = [2, 0, 3, 1]
        enc = TernaryTreeEncoding(tree, mode_map=mode_map)

        restored = TernaryTreeEncoding(pickle.loads(pickle.dumps(tree)), mode_map=mode_map)
        for majorana in range(2 * enc.num_modes):
            assert restored.majorana_image(majorana) == enc.majorana_image(majorana)

        op = fermion_op()
        assert_obs_equal(fermion_ternary_tree(op, restored, 4), fermion_ternary_tree(op, enc, 4))

    def test_classes_are_importable_from_the_framework_module(self):
        """These are framework pieces, so they live on ``qiskit_fermions.mappers``."""
        import qiskit_fermions.mappers as mappers
        import qiskit_fermions.mappers.library as library

        assert mappers.TernaryTree is TernaryTree
        assert mappers.TernaryTreeEncoding is TernaryTreeEncoding
        # ... and not on the library module, which holds the mapper functions.
        assert not hasattr(library, "TernaryTree")
        assert not hasattr(library, "TernaryTreeEncoding")

    def test_public_module_path(self):
        """The public ``__module__`` is the logical path, not the private ``_lib`` one."""
        assert TernaryTree.__module__ == "qiskit_fermions.mappers.ternary_tree"
        assert TernaryTreeEncoding.__module__ == "qiskit_fermions.mappers.ternary_tree"

    def test_trees_are_frozen(self):
        """Both pyclasses are frozen: their invariants are checked at construction."""
        tree = TernaryTree.chain(3, "Z")
        with pytest.raises(AttributeError):
            tree.root = 1


class TestErrors:
    """Every Rust error must surface as a Python exception with a useful message."""

    def test_rejects_bad_label(self):
        with pytest.raises(ValueError, match="invalid Pauli label"):
            TernaryTree.chain(3, "Q")
        with pytest.raises(ValueError, match="invalid Pauli label"):
            TernaryTree([None, (0, "Q")])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one node"):
            TernaryTree.chain(0, "Z")
        with pytest.raises(ValueError, match="at least one node"):
            TernaryTree([])

    def test_rejects_duplicate_label(self):
        with pytest.raises(ValueError, match="more than one link labelled"):
            TernaryTree([None, (0, "X"), (0, "X")])

    def test_rejects_multiple_roots(self):
        with pytest.raises(ValueError, match="exactly one root"):
            TernaryTree([None, None])

    def test_rejects_disconnected(self):
        # One root plus a two-node cycle: the root count is right but the cycle is unreachable.
        with pytest.raises(ValueError, match="not reachable"):
            TernaryTree([None, (2, "X"), (1, "X")])

    def test_rejects_out_of_range_parent(self):
        with pytest.raises(ValueError, match="references node"):
            TernaryTree([None, (7, "X")])

    def test_rejects_bad_branching_order(self):
        with pytest.raises(ValueError, match="exactly 3 labels"):
            TernaryTree.breadth_first(4, 2, "XY")

    def test_rejects_bad_mode_map(self):
        tree = TernaryTree.chain(3, "Z")
        with pytest.raises(ValueError, match="mode map has"):
            TernaryTreeEncoding(tree, mode_map=[0, 1])
        with pytest.raises(ValueError, match="not a permutation"):
            TernaryTreeEncoding(tree, mode_map=[0, 1, 1])
        # An out-of-range entry is a mode, so it is reported as one rather than as a node.
        with pytest.raises(ValueError, match="references mode 9"):
            TernaryTreeEncoding(tree, mode_map=[0, 1, 9])

    def test_rejects_too_few_qubits(self):
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(ValueError, match="too small"):
            fermion_ternary_tree(fermion_op(), enc, 3)

    def test_rejects_too_few_qubits_for_a_low_mode_operator(self):
        """A register below the encoding's extent is rejected, whatever modes the operator touches.

        A mode's Pauli image spans its whole path to the root, so on the balanced tree even
        ``gamma_0`` acts on qubit 1. Clamping the bound to ``num_qubits`` used to let this reach
        ``qk_obs_new`` and abort the process.
        """
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        cases = [
            (fermion_ternary_tree, FermionOperator.from_dict({((True, 0), (False, 0)): 1.0})),
            (majorana_ternary_tree, MajoranaOperator.from_dict({(0,): 1.0})),
            (edge_vertex_ternary_tree, EdgeVertexOperator.from_dict({((0, 0),): 1.0})),
            (transfer_vertex_ternary_tree, TransferVertexOperator.from_dict({((0, 0),): 1.0})),
        ]
        for mapper, op in cases:
            for num_qubits in range(1, 4):
                with pytest.raises(ValueError, match="too small for an encoding of 4 modes"):
                    mapper(op, enc, num_qubits)
            # The exact register and a padded one are both fine.
            for num_qubits in (4, 6):
                assert mapper(op, enc, num_qubits).num_qubits == num_qubits

    def test_reports_the_encoding_bound_when_the_register_is_ample(self):
        """An operator outside the encoding is reported against the encoding, not the register.

        Raising ``num_qubits`` cannot widen the encoding, so reporting the clamped bound would send
        the caller after the wrong fix.
        """
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(ValueError, match=r"num_qubits \(4\) is too small.*mode index 7"):
            fermion_ternary_tree(FermionOperator.from_dict({((True, 7),): 1.0}), enc, 10)

    def test_total_parity_rejects_too_few_qubits(self):
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(ValueError, match="too small for an encoding of 4 modes"):
            enc.total_parity(2)
        # The exact register and a padded one are both fine.
        for num_qubits in (4, 6):
            assert enc.total_parity(num_qubits).num_qubits == num_qubits

    def test_rejects_excess_branching(self):
        """``branching`` above 3 is an error, not silently the ``branching=3`` tree."""
        for branching in (4, 5, 255):
            with pytest.raises(ValueError, match="at most 3"):
                TernaryTree.breadth_first(7, branching)
        # Three is still the largest accepted value.
        assert TernaryTree.breadth_first(7, 3).num_nodes == 7

    def test_rejects_operator_outside_the_encoding(self):
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(ValueError, match="too small"):
            fermion_ternary_tree(FermionOperator.from_dict({((True, 7),): 1.0}), enc, 4)
        with pytest.raises(ValueError, match="too small"):
            majorana_ternary_tree(MajoranaOperator.from_dict({(15,): 1.0}), enc, 4)

    def test_rejects_out_of_range_majorana(self):
        enc = TernaryTreeEncoding(TernaryTree.chain(3, "Z"))
        with pytest.raises(ValueError, match="out of range"):
            enc.majorana_image(6)

    def test_rejects_out_of_range_node(self):
        with pytest.raises(ValueError, match="out of range"):
            TernaryTree.chain(3, "Z").children(9)


class TestTypeAgnosticWrapper:
    """The ``ternary_tree`` wrapper must dispatch on the operator type and nothing more.

    It implements no conversion logic of its own, so every arm is pinned against the direct
    implementation it is supposed to delegate to. A swapped or dropped argument shows up as a
    mismatch, not as a silent wrong answer.
    """

    @pytest.mark.parametrize(
        "op_factory,direct",
        [
            (fermion_op, fermion_ternary_tree),
            (majorana_op, majorana_ternary_tree),
            (edge_op, edge_vertex_ternary_tree),
            (transfer_op, transfer_vertex_ternary_tree),
        ],
    )
    def test_dispatches_to_direct_implementation(self, op_factory, direct):
        op = op_factory()
        # A non-uniform tree, so the wrapper cannot appear to work by collapsing onto the chain, and
        # a non-identity mode map, so a dropped `encoding` argument cannot pass either.
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3), mode_map=[2, 0, 3, 1])
        assert_obs_equal(ternary_tree(op, enc, 4), direct(op, enc, 4))

    def test_num_qubits_is_forwarded(self):
        # Padding with the identity is the encoding's own behaviour; what is checked here is that the
        # argument reaches it rather than being replaced by the encoding's mode count.
        enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
        assert ternary_tree(fermion_op(), enc, 6).num_qubits == 6

    def test_unsupported_type_raises(self):
        # All four operator types have a direct implementation, so the guard is exercised with
        # something that is not a fermionic operator at all. It must raise rather than fail obscurely
        # deeper in, and it is what any future operator type will land on until it gains one.
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(TypeError, match="SparseObservable"):
            ternary_tree(SparseObservable.identity(4), enc, 4)

    def test_propagates_errors_from_the_implementation(self):
        # The wrapper must not swallow or re-wrap the underlying ValueError.
        enc = TernaryTreeEncoding(TernaryTree.chain(4, "Z"))
        with pytest.raises(ValueError, match="too small"):
            ternary_tree(fermion_op(), enc, 3)
