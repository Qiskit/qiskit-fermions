{#
   Type aliases are module-level data, not classes. We render them with
   ``autodata`` so that Sphinx documents the alias itself (and its docstring)
   rather than trying to enumerate the members of the aliased class -- the
   latter emits "don't know which module to import" warnings for every
   inherited member.
-#}

{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autodata:: {{ objname }}
