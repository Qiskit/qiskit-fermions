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

{% block methods_summary %}
  {% set wanted_methods = (methods | reject('==', '__init__') | reject('in', inherited_members) | list) %}
  {% if wanted_methods %}
   .. rubric:: Methods
    {% for item in wanted_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
