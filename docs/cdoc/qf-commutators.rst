*********************
Commutator Generators
*********************

The :ref:`operator representations <qf_operators>` provide efficient functions
for the computation of various commutators. Since these functions work with
generic operator representations (a notion that does not exist in C), they are
explained generically in the following three sections.
The actual functions contained in the C API for the various operator
representations are listed at the bottom of this page.

Commutator
^^^^^^^^^^

.. c:function:: OpType qf_op_type_commutator(const OpType *op_a, const OpType *op_b)

    .. caution::
       The function signature here is **generic**! A real one will replace
       ``OpType`` with an actual :ref:`operator representations <qf_operators>`
       and ``op_type`` with its matching prefix (e.g. ``ferm_op`` for
       :c:struct:`QfFermionOperator`).

    The commutator is defined as:

    .. math::

       [A, B] = AB - BA

    :param op_a: A pointer to the operator :math:`A`.
    :param op_b: A pointer to the operator :math:`B`.
    :return: A pointer to the constructed operator commutator.

Example
.......

.. code-block:: c
   :linenos:

   QfFermionOperator *op1 = qf_ferm_op_zero();
   QkComplex64 coeff1 = {1.0, 0.0};
   bool action1[2] = {true, false};
   uint32_t modes1[2] = {0, 0};
   qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
   QfFermionOperator *op2 = qf_ferm_op_zero();
   QkComplex64 coeff2 = {2.0, 0.0};
   bool action2[2] = {false, true};
   uint32_t modes2[2] = {0, 0};
   qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);

   QfFermionOperator *comm = qf_ferm_op_commutator(op1, op2);

Anti-Commutator
^^^^^^^^^^^^^^^

.. c:function:: OpType qf_op_type_anti_commutator(const OpType *op_a, const OpType *op_b)

    .. caution::
       The function signature here is **generic**! A real one will replace
       ``OpType`` with an actual :ref:`operator representations <qf_operators>`
       and ``op_type`` with its matching prefix (e.g. ``ferm_op`` for
       :c:struct:`QfFermionOperator`).

    The anti-commutator is defined as:

    .. math::

       \{A, B\} = AB + BA

    :param op_a: A pointer to the operator :math:`A`.
    :param op_b: A pointer to the operator :math:`B`.
    :return: A pointer to the constructed operator anti-commutator.

Example
.......

.. code-block:: c
   :linenos:

   QfFermionOperator *op1 = qf_ferm_op_zero();
   QkComplex64 coeff1 = {1.0, 0.0};
   bool action1[2] = {true, false};
   uint32_t modes1[2] = {0, 0};
   qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
   QfFermionOperator *op2 = qf_ferm_op_zero();
   QkComplex64 coeff2 = {2.0, 0.0};
   bool action2[2] = {false, true};
   uint32_t modes2[2] = {0, 0};
   qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);

   QfFermionOperator *anti_comm = qf_ferm_op_anti_commutator(op1, op2);

Double-Commutator
^^^^^^^^^^^^^^^^^

.. c:function:: OpType qf_op_type_double_commutator(const OpType *op_a, const OpType *op_b, const OpType *op_c, bool sign)

    .. caution::
       The function signature here is **generic**! A real one will replace
       ``OpType`` with an actual :ref:`operator representations <qf_operators>`
       and ``op_type`` with its matching prefix (e.g. ``ferm_op`` for
       :c:struct:`QfFermionOperator`).

    The double-commutator is defined as follows (see also Chapter 13.6, *Equation of motion
    methods*, page 479 of [1]_; the ``sign=false`` case is the symmetric double commutator
    :math:`[A, B, C]` appearing in Equation (2) of [2]_):

    If ``sign`` is ``false``, it returns

    .. math::
         [[A, B], C]/2 + [A, [B, C]]/2 = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If ``sign`` is ``true``, it returns

    .. math::
         \{[A, B], C\}/2 + \{A, [B, C]\}/2 = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.


    .. [1] R. McWeeny. Methods of Molecular Quantum Mechanics.
           2nd Edition, Academic Press, 1992. ISBN 0-12-486552-6.

    .. [2] P. J. Ollitrault, A. Miessen, I. Tavernelli. Molecular Quantum Dynamics: A Quantum
           Computing Perspective. Acc. Chem. Res. 54, 4229-4238 (2021).
           https://doi.org/10.1021/acs.accounts.1c00514

    :param op_a: A pointer to the operator :math:`A`.
    :param op_b: A pointer to the operator :math:`B`.
    :param op_c: A pointer to the operator :math:`C`.
    :param sign: the nature of the outer (anti-)commutator as per the definition above.
    :return: A pointer to the constructed operator anti-commutator.


Example
.......

.. code-block:: c
   :linenos:

   QfFermionOperator *op1 = qf_ferm_op_zero();
   QkComplex64 coeff1 = {1.0, 0.0};
   bool action1[2] = {true, false};
   uint32_t modes1[2] = {0, 0};
   qf_ferm_op_add_term(op1, 2, action1, modes1, &coeff1);
   QfFermionOperator *op2 = qf_ferm_op_zero();
   QkComplex64 coeff2 = {2.0, 0.0};
   bool action2[2] = {false, true};
   uint32_t modes2[2] = {0, 0};
   qf_ferm_op_add_term(op2, 2, action2, modes2, &coeff2);
   QfFermionOperator *op3 = qf_ferm_op_zero();
   qf_ferm_op_add_term(op3, 2, action1, modes1, &coeff1);
   QkComplex64 coeff3 = {2.0, 0.5};
   qf_ferm_op_add_term(op3, 2, action2, modes2, &coeff3);

   QfFermionOperator *double_comm = qf_ferm_op_double_commutator(op1, op2, op3, false);

Functions
^^^^^^^^^

.. doxygengroup:: qf_commutators
   :content-only:
