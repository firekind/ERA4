from typing import Any

import dearpygui.dearpygui as dpg


def set_value(item: int | str | None, value: Any, **kwargs) -> None:
    if item is None:
        return
    dpg.set_value(item, value, **kwargs)


def set_item_label(item: str | int | None, label: str) -> None:
    if item is None:
        return
    dpg.set_item_label(item, label)


def disable_item(item: str | int | None) -> None:
    if item is None:
        return
    dpg.disable_item(item)


def enable_item(item: str | int | None) -> None:
    if item is None:
        return
    dpg.enable_item(item)


def configure_item(item: str | int | None, **kwargs) -> None:
    if item is None:
        return
    dpg.configure_item(item, **kwargs)


def fit_axis_data(item: str | int | None, **kwargs) -> None:
    if item is None:
        return
    dpg.fit_axis_data(item, **kwargs)
