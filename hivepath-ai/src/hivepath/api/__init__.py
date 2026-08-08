"""HTTP interface.

Deliberately empty of eager imports. ``application`` builds a FastAPI instance
at module scope, and ``hivepath.planning`` imports ``hivepath.api.schemas``; if
this module pulled in ``application`` then importing the schemas would drag the
whole app in and close a circular import. Import what you need directly:

    from hivepath.api.application import create_app
"""

__all__: list[str] = []
