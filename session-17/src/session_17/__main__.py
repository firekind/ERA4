import logging

from session_17 import App

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)

    with App() as a:
        a.run()
