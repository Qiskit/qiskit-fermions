{#
   We show all the class's methods and attributes on the same page.
   This template does _NOT_ document attributes and methods inherited from
   parent classes, but _only_ the ones explicitly defined by this class.
   This is not the default template (see `class.rst` for that, which _does_
   documented inherited attributes and methods, too) and must be chosen
   explicitly when writing the `autosummary::` directive.
-#}

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :no-members:
   :no-inherited-members:
   :special-members:

{% block attributes_summary %}
  {% set wanted_attributes = (attributes | reject('in', inherited_members) | list) %}
  {% if wanted_attributes %}
   .. rubric:: Attributes
    {% for item in wanted_attributes %}
   .. autoattribute:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock -%}

{#
   ``autosummary`` unconditionally excludes single-underscore names from ``methods`` (see
   https://github.com/sphinx-doc/sphinx/issues/8922 -- ``:meta public:`` is not honored here, only
   by a plain ``autoclass :members:``, which this template does not use). Our own protocol methods
   (e.g. ``_apply_unitary_``, ``_linear_operator_``) follow a leading-and-trailing-single-underscore
   naming convention (never used for anything else in this namespace), so recover them here by name
   from the unfiltered ``members`` list instead. ``select``/``reject`` only dispatch to registered
   Jinja tests (not arbitrary string methods), so this filters via a plain loop instead.
-#}
{% set protocol_members = [] %}
{% for item in members %}
  {% if item not in attributes and item not in methods and item != '__init__'
        and item not in inherited_members
        and item.startswith('_') and not item.startswith('__')
        and item.endswith('_') and not item.endswith('__') %}
    {% set _ = protocol_members.append(item) %}
  {% endif %}
{% endfor %}

{% block methods_summary %}
  {% set wanted_methods = (methods | reject('==', '__init__') | reject('in', inherited_members) | list) %}
  {% if wanted_methods %}
   .. rubric:: Methods
    {% for item in wanted_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
  {% if protocol_members %}
   .. rubric:: Protocol Methods
    {% for item in protocol_members %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
