"""Allow `python -m kloter` invocation."""

from dotenv import load_dotenv

load_dotenv()

from kloter.cli import main  # noqa: E402

main()
