"""Single source of truth for the build number.

It exists because "am I actually running the new files?" turned out to be a
real question — the version is shown in the UI header and served at
/api/version so it can always be answered in one glance.
"""

__version__ = "0.20.0"
