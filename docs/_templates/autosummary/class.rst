{#
   We show all the class's methods and attributes on the same page. By default, we document
   all methods, including those defined by parent classes.
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
  {% set inherited_attributes = (attributes | select('in', inherited_members) | list) %}
  {% if inherited_attributes %}
   .. rubric:: Inherited Attributes
    {% for item in inherited_attributes %}
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
  {% set wanted_protocol_methods = (protocol_members | reject('in', inherited_members) | list) %}
  {% if wanted_protocol_methods %}
   .. rubric:: Protocol Methods
    {% for item in wanted_protocol_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
  {% set inherited_methods = (methods | reject('==', '__init__') | select('in', inherited_members) | list) %}
  {% if inherited_methods %}
   .. rubric:: Inherited Methods
    {% for item in inherited_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
  {% set inherited_protocol_methods = (protocol_members | select('in', inherited_members) | list) %}
  {% if inherited_protocol_methods %}
   .. rubric:: Inherited Protocol Methods
    {% for item in inherited_protocol_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
