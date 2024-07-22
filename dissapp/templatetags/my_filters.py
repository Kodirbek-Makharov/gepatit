from django import template

register = template.Library()


@register.filter(name="is_float")
def is_float(value):
    return isinstance(value, float)

