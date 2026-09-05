import importlib
import importlib.util
import logging
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def load_custom_app(app_import: str, base_dir: Path | None = None) -> FastAPI:
    """Load custom FastAPI app from import path.

    Supports both file-based and module-based imports:
    - File path: "./custom_routes.py:app" or "/path/to/file.py:app"
    - Module path: "my_package.custom:app"
    """
    if ":" not in app_import:
        raise ValueError(
            f"Invalid app import path format: {app_import}. "
            "Expected format: 'path/to/file.py:variable' or 'module.path:variable'"
        )

    path, name = app_import.rsplit(":", 1)

    path_obj = Path(path)
    is_file_path = path_obj.suffix == ".py" or path.startswith("./") or path.startswith("../")

    if is_file_path:
        if not path_obj.is_absolute() and base_dir is not None:
            path_obj = (base_dir / path_obj).resolve()

        if not path_obj.exists():
            raise FileNotFoundError(f"Custom app file not found: {path_obj}")

        spec = importlib.util.spec_from_file_location("custom_app_module", str(path_obj))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {path_obj}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(path)

    if not hasattr(module, name):
        raise AttributeError(
            f"App '{name}' not found in module '{path}'. "
            f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
        )

    user_app = getattr(module, name)

    if not isinstance(user_app, FastAPI):
        raise TypeError(
            f"Object '{name}' in module '{path}' is not a FastAPI application. "
            "Custom apps must be FastAPI instances. Use: from fastapi import FastAPI; app = FastAPI()"
        )

    logger.info("Successfully loaded custom app '%s' from %s", name, path)
    return user_app
