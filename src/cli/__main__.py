"""Allow running CLI as: python -m src.cli"""

import sys

from dotenv import load_dotenv

load_dotenv()

from .main import main

sys.exit(main())
