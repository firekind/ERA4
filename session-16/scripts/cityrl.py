import sys

import qdarktheme
from PySide6.QtWidgets import QApplication

from session_16.cityrl import Dashboard


def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    dashboard = Dashboard()
    dashboard.show()

    app.exec()


if __name__ == "__main__":
    main()
