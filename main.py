import logging

from app.app_factory import create_app


logging.basicConfig(level=logging.INFO)

app = create_app()
