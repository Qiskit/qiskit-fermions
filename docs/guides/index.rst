########
Overview
########

This page summarizes the guides that are available in addition to the extensive API documentation of
this package.

.. note::
   These guides might refer to specific Python modules when explaining core concepts.
   Unless otherwise stated, the C API provides matching functionality, although it is not structured
   into modules. However, the function names will correspond naturally and should be easy to navigate.

Get started
===========

This section lists some simple goal-oriented guides to get you started with
specific example applications:

.. toctree::
   :maxdepth: 1
   :glob:

   sqdrift
   skqd
   lucj

You can also start with these more extensive end-to-end examples:

* :ref:`1d_fermi_hubbard`
* :ref:`2d_fermi_hubbard`

Explanations
============

This section lists guides for explaining various concepts and components from
this package:

.. toctree::
   :maxdepth: 1
   :glob:

   operators
   grouping
   mappers
   circuit
   transpilation
   ffsim

How-tos
=======

This section lists goal-oriented guides for accomplishing specific tasks with
particular components of this package:

Circuits
--------

.. toctree::
   :maxdepth: 1
   :glob:

   vqe_outer_loop

Transpiler passes
-----------------

.. toctree::
   :maxdepth: 1
   :glob:

   merge_slater_determinant
