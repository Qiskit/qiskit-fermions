// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use super::library::into_py_obs;

use qiskit_fermions_core::mappers::ternary_tree::encoding::{ModeMap, TernaryTreeEncoding};
use qiskit_fermions_core::mappers::ternary_tree::{NodeId, PauliLabel, Slot, TernaryTree};

/// Parses a Pauli label given as a single character.
fn parse_label(label: char) -> PyResult<PauliLabel> {
    match label {
        'X' | 'x' => Ok(PauliLabel::X),
        'Y' | 'y' => Ok(PauliLabel::Y),
        'Z' | 'z' => Ok(PauliLabel::Z),
        other => Err(crate::value_err(format!(
            "invalid Pauli label {other:?}; expected one of 'X', 'Y', 'Z'"
        ))),
    }
}

/// Renders a Pauli label as a single character.
fn label_char(label: PauliLabel) -> char {
    match label {
        PauliLabel::X => 'X',
        PauliLabel::Y => 'Y',
        PauliLabel::Z => 'Z',
    }
}

/// A ternary tree defining a fermion-to-qubit encoding.
///
/// A ternary tree on :math:`N` nodes encodes :math:`N` fermionic modes onto :math:`N` qubits. Each node
/// is a qubit carrying three downward **links**, labelled ``X``, ``Y`` and ``Z``. A link either
/// descends to a child node (an *edge*) or terminates (a *leg*). Each leg's upward path to the root
/// spells a Pauli string: crossing the link labelled :math:`P` that descends *from* node :math:`u`
/// contributes :math:`P` on qubit :math:`u`, with the identity everywhere else.
///
/// Every tree of this shape yields a valid encoding, so the constructor accepts an arbitrary one. The
/// familiar encodings are particular shapes:
///
/// .. list-table::
///    :header-rows: 1
///
///    * - Encoding
///      - Construction
///      - Pauli weight
///    * - Jordan-Wigner
///      - ``TernaryTree.chain(n, "Z")``
///      - :math:`n`
///    * - Parity
///      - ``TernaryTree.chain(n, "X")``
///      - :math:`n`
///    * - Balanced [1]_
///      - ``TernaryTree.breadth_first(n, 3)``
///      - :math:`\lceil \log_3(2n+1) \rceil`, optimal
///    * - Binary-branching
///      - ``TernaryTree.breadth_first(n, 2, "ZXY")``
///      - :math:`\lfloor \log_2 n \rfloor + 1`
///
/// Those two conveniences cover the *uniform* families only, which is why the constructor also takes an
/// arbitrary tree: a parent-and-label specification, one entry per node, giving that node's parent and
/// the label of the link descending to it (``None`` for the root). The trees that motivate the framework
/// need it -- the Bonsai algorithm [2]_ grows a spanning tree of a device's coupling graph, so its
/// branching follows that connectivity and is neither a chain nor uniform.
///
/// Not every encoding in the family has a convenience constructor: Bravyi-Kitaev, for instance, is a tree
/// that is not uniform, so it is built by computing its specification. The
/// :ref:`mappers guide <bravyi_kitaev_by_hand>` works that case through as an example.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers import TernaryTree
///     >>> tree = TernaryTree.breadth_first(4, 3)
///     >>> tree.num_nodes, tree.num_legs, tree.max_weight
///     (4, 9, 2)
///
/// The balanced tree's weight grows logarithmically where Jordan-Wigner's grows linearly:
///
/// .. doctest::
///
///     >>> TernaryTree.chain(13, "Z").max_weight
///     13
///     >>> TernaryTree.breadth_first(13, 3).max_weight
///     3
///
/// An arbitrary tree is given as one ``(parent, label)`` pair per node, with ``None`` for the root:
///
/// .. doctest::
///
///     >>> tree = TernaryTree([None, (0, "X"), (0, "Y"), (1, "Z")])
///     >>> tree.num_nodes, tree.num_legs
///     (4, 9)
///
/// .. [1] Z. Jiang, A. Kalev, W. Mruczkiewicz and H. Neven, Optimal fermion-to-qubit mapping via
///        ternary trees with applications to reduced quantum states learning, Quantum 4, 276 (2020),
///        `arXiv:1910.10746 <https://arxiv.org/abs/1910.10746>`_.
/// .. [2] A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, Bonsai algorithm:
///        grow your own fermion-to-qubit mappings, PRX Quantum 4, 030314 (2023),
///        `arXiv:2212.09731 <https://arxiv.org/abs/2212.09731>`_.
#[gen_stub_pyclass]
#[gen_stub(module = "qiskit_fermions._lib.mappers.ternary_tree_structures")]
#[pyclass(
    module = "qiskit_fermions.mappers.ternary_tree",
    name = "TernaryTree",
    frozen
)]
#[derive(Clone)]
pub struct PyTernaryTree {
    pub inner: TernaryTree,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTernaryTree {
    /// Build a tree from a parent-and-label specification.
    ///
    /// Args:
    ///     spec: one entry per node, giving that node's parent and the label of the link descending to
    ///         it, or ``None`` for the root. Exactly one entry must be ``None``.
    ///
    /// Raises:
    ///     ValueError: if the specification does not describe a single connected ternary tree, or if a
    ///         label is not one of ``"X"``, ``"Y"``, ``"Z"``.
    #[new]
    fn new(spec: Vec<Option<(NodeId, char)>>) -> PyResult<Self> {
        let parsed: PyResult<Vec<Option<(NodeId, PauliLabel)>>> = spec
            .into_iter()
            .map(|entry| match entry {
                None => Ok(None),
                Some((parent, label)) => Ok(Some((parent, parse_label(label)?))),
            })
            .collect();
        Ok(Self {
            inner: TernaryTree::try_new(&parsed?).map_err(crate::value_err)?,
        })
    }

    /// Build a linear chain descending along ``label``.
    ///
    /// ``label="Z"`` is the Jordan-Wigner encoding and ``label="X"`` the parity encoding; both have
    /// Pauli weight ``num_nodes``.
    ///
    /// Args:
    ///     num_nodes: the number of modes, and hence qubits.
    ///     label: the link the chain descends along.
    ///
    /// Raises:
    ///     ValueError: if ``num_nodes`` is zero or ``label`` is not a Pauli label.
    #[staticmethod]
    fn chain(num_nodes: u32, label: char) -> PyResult<Self> {
        Ok(Self {
            inner: TernaryTree::chain(num_nodes, parse_label(label)?).map_err(crate::value_err)?,
        })
    }

    /// Build a tree by attaching up to ``branching`` children to each node in turn.
    ///
    /// ``branching=3`` gives the balanced tree, whose Pauli weight
    /// :math:`\lceil \log_3(2n+1) \rceil` is optimal over all fermion-to-qubit mappings.
    ///
    /// Args:
    ///     num_nodes: the number of modes, and hence qubits.
    ///     branching: how many children to attach per node, at most 3.
    ///     order: the labels to attach children along, as a 3-character string. Changing it changes
    ///         which Pauli strings the encoding produces, though not their weight.
    ///
    /// Raises:
    ///     ValueError: if ``num_nodes`` is zero, ``branching`` exceeds 3, ``order`` is not 3 Pauli
    ///         labels, or its first ``branching`` labels are not distinct.
    ///
    /// Note:
    ///     ``branching=2`` gives a binary-branching tree of weight :math:`\lfloor \log_2 n \rfloor + 1`.
    ///     That matches the Bravyi-Kitaev weight but is *not* that encoding, except when ``num_nodes``
    ///     is a power of two: Bravyi-Kitaev's tree is the irregular partial-sum shape. See
    ///     :class:`.TernaryTree` for how to build it.
    #[staticmethod]
    #[pyo3(signature = (num_nodes, branching, order = "XYZ"))]
    fn breadth_first(num_nodes: u32, branching: u8, order: &str) -> PyResult<Self> {
        let labels: Vec<PauliLabel> = order.chars().map(parse_label).collect::<PyResult<_>>()?;
        let order: [PauliLabel; 3] = labels.try_into().map_err(|_| {
            crate::value_err(format!("order must name exactly 3 labels, got {order:?}"))
        })?;
        Ok(Self {
            inner: TernaryTree::breadth_first(num_nodes, branching, order)
                .map_err(crate::value_err)?,
        })
    }

    /// The number of nodes, which is both the number of modes and the number of qubits.
    #[getter]
    fn num_nodes(&self) -> u32 {
        self.inner.num_nodes()
    }

    /// The number of legs, always ``2 * num_nodes + 1``.
    ///
    /// This holds for every shape: a node with fewer children has more legs, not fewer strings. Of
    /// these, ``2 * num_nodes`` pair into the Majorana operators and one is left over, carrying the
    /// total fermionic parity.
    #[getter]
    fn num_legs(&self) -> u32 {
        self.inner.num_legs()
    }

    /// The root node.
    #[getter]
    fn root(&self) -> NodeId {
        self.inner.root()
    }

    /// The largest Pauli weight of any single Majorana operator this tree generates.
    ///
    /// A fermionic creation or annihilation operator is a sum of two Majorana operators, so its image
    /// may reach twice this.
    #[getter]
    fn max_weight(&self) -> u32 {
        self.inner.max_weight()
    }

    /// Return the Pauli string of every leg, as ``(qubit, label)`` pairs ordered leaf-to-root.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.mappers import TernaryTree
    ///     >>> TernaryTree.chain(2, "Z").leg_paths()
    ///     [[(0, 'X')], [(0, 'Y')], [(1, 'X'), (0, 'Z')], [(1, 'Y'), (0, 'Z')], [(1, 'Z'), (0, 'Z')]]
    ///
    /// The qubits within a path are *not* sorted: they follow the tree's depth rather than the node
    /// numbering. Use :meth:`TernaryTreeEncoding.majorana_image` for the sorted form the mapped observables
    /// use.
    fn leg_paths(&self) -> Vec<Vec<(NodeId, char)>> {
        (0..self.inner.num_legs())
            .map(|leg| {
                self.inner
                    .leg_path(leg)
                    .into_iter()
                    .map(|(qubit, label)| (qubit, label_char(label)))
                    .collect()
            })
            .collect()
    }

    /// Return the children of ``node``, as a ``{label: child}`` mapping over its edges.
    ///
    /// Links that terminate in a leg are omitted.
    fn children(&self, node: NodeId) -> PyResult<Vec<(char, NodeId)>> {
        if node >= self.inner.num_nodes() {
            return Err(crate::value_err(format!(
                "node {node} is out of range for a tree with {} nodes",
                self.inner.num_nodes()
            )));
        }
        Ok(PauliLabel::iter()
            .filter_map(|label| match self.inner.child_of(node, label) {
                Slot::Edge(child) => Some((label_char(label), child)),
                Slot::Leg(_) => None,
            })
            .collect())
    }

    /// Returns the constructor arguments needed to pickle this tree.
    ///
    /// The parent-and-label specification is recovered from the tree itself rather than stored: it is
    /// exactly what the tree records per node, so there is nothing to keep in sync.
    ///
    /// The tree is the thing worth carrying between processes, since it is what a search produces and
    /// what defines the encoding. :class:`TernaryTreeEncoding` deliberately has no pickle protocol: it
    /// is a derived artifact that compiling discards its inputs to build, so supporting it would mean
    /// retaining the tree and mode map purely to serialize them. Pickle the tree and recompile.
    fn __getnewargs__(&self) -> (Vec<Option<(NodeId, char)>>,) {
        let spec = (0..self.inner.num_nodes())
            .map(|node| {
                self.inner
                    .parent_of(node)
                    .map(|(parent, label)| (parent, label_char(label)))
            })
            .collect();
        (spec,)
    }

    fn __repr__(&self) -> String {
        format!(
            "<TernaryTree with {} nodes and {} legs, max weight {}>",
            self.inner.num_nodes(),
            self.inner.num_legs(),
            self.inner.max_weight()
        )
    }
}

/// A ternary tree compiled into the Pauli images of its Majorana operators.
///
/// Pairing the tree's legs into Majorana operators and building their Pauli strings is done once, here,
/// rather than per mapped operator. Pass one of these to the ternary-tree mappers.
///
/// The pairing follows Algorithm 1 of the Bonsai paper [1]_: from each node, descend its ``X`` link and
/// then ``Z`` links until reaching a leg, which carries :math:`\gamma` of that node's mode; the ``Y``
/// link likewise gives :math:`\gamma'`. This is what makes :math:`|0\cdots0\rangle` the fermionic
/// vacuum, so it is not an adjustable convention.
///
/// .. [1] A. Miller, Z. Zimborás, S. Knecht, S. Maniscalco and G. García-Pérez, Bonsai algorithm:
///        grow your own fermion-to-qubit mappings, PRX Quantum 4, 030314 (2023),
///        `arXiv:2212.09731 <https://arxiv.org/abs/2212.09731>`_.
///
/// Usage
/// =====
///
/// .. doctest::
///
///     >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
///     >>> enc = TernaryTreeEncoding(TernaryTree.breadth_first(4, 3))
///     >>> enc.num_modes, enc.max_weight
///     (4, 2)
///
/// The encoding is built once and reused across every operator mapped through it, so the pairing and
/// string construction are not repeated per call. The qubit count of a mapped operator is given to the
/// mapper rather than fixed here, so one encoding serves registers of any width from
/// :attr:`num_modes` upwards.
#[gen_stub_pyclass]
#[gen_stub(module = "qiskit_fermions._lib.mappers.ternary_tree_structures")]
#[pyclass(
    module = "qiskit_fermions.mappers.ternary_tree",
    name = "TernaryTreeEncoding",
    frozen
)]
#[derive(Clone)]
pub struct PyTernaryTreeEncoding {
    pub inner: TernaryTreeEncoding,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTernaryTreeEncoding {
    /// Compile ``tree`` into the Pauli images of its Majorana operators.
    ///
    /// Args:
    ///     tree: the tree defining the encoding.
    ///     mode_map: which mode each node serves, as a permutation of ``range(tree.num_nodes)``
    ///         indexed by node. Defaults to the identity. Reordering the modes of a fixed tree is one
    ///         of the cheapest ways to lower the weight of a particular Hamiltonian.
    ///
    /// Raises:
    ///     ValueError: if ``mode_map`` is not a permutation of the tree's nodes.
    #[new]
    #[pyo3(signature = (tree, mode_map = None))]
    fn new(tree: &PyTernaryTree, mode_map: Option<Vec<u32>>) -> PyResult<Self> {
        let mode_map = match mode_map {
            Some(modes) => ModeMap::Permutation(modes),
            None => ModeMap::Identity,
        };
        Ok(Self {
            inner: TernaryTreeEncoding::new(&tree.inner, &mode_map).map_err(crate::value_err)?,
        })
    }

    /// The number of fermionic modes this encoding covers.
    #[getter]
    fn num_modes(&self) -> u32 {
        self.inner.num_modes()
    }

    /// The largest Pauli weight of any single Majorana operator.
    #[getter]
    fn max_weight(&self) -> u32 {
        self.inner.max_weight()
    }

    /// Return the Pauli image of Majorana operator ``majorana``, as ``(qubit, label)`` pairs.
    ///
    /// The index is ``2 * mode + is_prime``, matching
    /// :func:`~qiskit_fermions.operators.gamma`. The string is ordered by ascending qubit.
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
    ///     >>> enc = TernaryTreeEncoding(TernaryTree.chain(3, "Z"))
    ///     >>> enc.majorana_image(4)
    ///     [(0, 'Z'), (1, 'Z'), (2, 'X')]
    ///
    /// The coefficient is always exactly :math:`1`: a leg's string is a product of Paulis on distinct
    /// qubits, so it is Hermitian and squares to the identity with no phase to correct.
    fn majorana_image(&self, majorana: u32) -> PyResult<Vec<(u32, char)>> {
        if majorana >= 2 * self.inner.num_modes() {
            return Err(crate::value_err(format!(
                "Majorana index {majorana} is out of range for an encoding of {} modes",
                self.inner.num_modes()
            )));
        }
        let (qubits, labels) = self.inner.majorana_image(majorana);
        Ok(qubits
            .iter()
            .zip(labels)
            .map(|(&qubit, &label)| (qubit, label_char(label)))
            .collect())
    }

    /// Build the total fermionic parity observable of this encoding.
    ///
    /// A tree's :math:`2N+1` legs pair into :math:`2N` Majorana operators, leaving exactly one over, and
    /// that leftover is the total parity :math:`\prod_j (1 - 2 n_j)`.
    ///
    /// Args:
    ///     num_qubits: the number of qubits for the resulting observable. Must be at least the encoding's
    ///         mode count; a larger value pads with the identity.
    ///
    /// Returns:
    ///     The total-parity observable: a single Pauli string with a :math:`\pm 1` coefficient.
    ///
    /// Raises:
    ///     ValueError: if ``num_qubits`` is below the encoding's mode count.
    ///
    /// An observable to measure, not a constraint to impose
    /// ====================================================
    ///
    /// Both eigenvalues of this operator are physical, so do not project a state onto the :math:`+1`
    /// sector. A ternary-tree encoding spends one qubit per mode and is a unitary isomorphism onto the
    /// *whole* :math:`2^N`-dimensional space: both parity sectors are represented, each with multiplicity
    /// :math:`2^{N-1}`, and there is no unphysical subspace to project away.
    ///
    /// Local encodings that add ancilla qubits do work that way. The flow sets in the
    /// :ref:`1D <1d_fermi_hubbard>` and :ref:`2D <2d_fermi_hubbard>` Fermi-Hubbard guides use more qubits
    /// than modes, which makes the extra dimensions unphysical and a stabilizer subspace the thing that
    /// keeps the encoding faithful. Ternary trees add no such qubits and have no such subspace.
    ///
    /// Note also that the string is *not* in general the all-:math:`Z` string. It is the leg reached by
    /// the all-:math:`Z` *path*, which carries the identity on every node not on that path.
    ///
    /// Usage
    /// =====
    ///
    /// On the Jordan-Wigner chain it is the full :math:`Z` string:
    ///
    /// .. doctest::
    ///
    ///     >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
    ///     >>> TernaryTreeEncoding(TernaryTree.chain(4, "Z")).total_parity(4)
    ///     <SparseObservable with 1 term on 4 qubits: (1+0j)(Z_3 Z_2 Z_1 Z_0)>
    ///
    /// On the balanced tree at four modes it is not:
    ///
    /// .. doctest::
    ///
    ///     >>> TernaryTreeEncoding(TernaryTree.breadth_first(4, 3)).total_parity(4)
    ///     <SparseObservable with 1 term on 4 qubits: (1+0j)(Z_3 Z_0)>
    ///
    /// The coefficient is :math:`\pm 1` and depends on the tree, so it is computed rather than assumed.
    #[gen_stub(override_return_type(type_repr="qiskit.quantum_info.SparseObservable", imports=("qiskit.quantum_info")))]
    fn total_parity(&self, num_qubits: u32) -> PyResult<Py<PyAny>> {
        let obs = self
            .inner
            .total_parity(num_qubits)
            .map_err(crate::value_err)?;
        Ok(unsafe { into_py_obs(obs) })
    }

    fn __repr__(&self) -> String {
        format!(
            "<TernaryTreeEncoding of {} modes, max weight {}>",
            self.inner.num_modes(),
            self.inner.max_weight()
        )
    }
}

/// The physical module holding the ternary-tree structures.
///
/// Its name differs from the `ternary_tree` mapper module under
/// [`library`](crate::mappers::library::ternary_tree) because `#[pymodule]` derives a `PyInit_` symbol
/// from the module's *Python* name, so two modules anywhere in the extension cannot share one. Only
/// this physical `_lib` path is affected: the classes declare their public path via `#[pyclass(module
/// = ...)]`, and `python/qiskit_fermions/mappers/__init__.py` re-exports them, so a user sees
/// `qiskit_fermions.mappers.TernaryTree` either way.
#[pymodule]
pub mod ternary_tree_structures {
    #[pymodule_export]
    use super::PyTernaryTree;

    #[pymodule_export]
    use super::PyTernaryTreeEncoding;
}
