"""
Utility subpackage.

Keep this module free of eager imports: importing submodules here can create
circular import chains (e.g. protocol -> utils.api_models -> utils.__init__).

Import utilities directly from their modules, e.g.:
  from talisman_ai.utils.uids import get_random_uids
"""
