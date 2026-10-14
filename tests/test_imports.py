def test_every_module_imports():
    """
    Every module of the package imports with the dependencies of pyproject.toml only (no tensorflow or other optional packages)
    """
    import importlib
    import pkgutil
    import SynBPS
    
    modules = [m.name for m in pkgutil.walk_packages(SynBPS.__path__, "SynBPS.")]
    assert len(modules) > 10
    
    for name in modules:
        importlib.import_module(name)
