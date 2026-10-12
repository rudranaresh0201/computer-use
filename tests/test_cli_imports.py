import importlib


def test_cli_module_compiles_and_imports():
    # every subcommand lives in this one file; a syntax error here breaks all of them at once
    assert importlib.import_module("computeruse.cli")
