"""Interactive dashboard for exploring technique results."""

try:
    import dash  # noqa: F401
    import dash_bootstrap_components  # noqa: F401
    from arth.dashboard.app import create_app

    __all__ = ["create_app"]
except ImportError:
    # dash/plotly/dash-bootstrap-components not installed -- degrade gracefully
    __all__ = []
