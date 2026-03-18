def generate_suggestions(*args, **kwargs):
    from .engine import generate_suggestions as _generate_suggestions

    return _generate_suggestions(*args, **kwargs)


__all__ = ["generate_suggestions"]
