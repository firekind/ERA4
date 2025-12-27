import cv2
import numpy as np
from numpy.typing import NDArray


def render_car(
    frame: NDArray[np.uint8],
    pos: tuple[int, int],
    angle: float = 0,
    width: int = 14,
    height: int = 8,
):
    # Car body color (Nord blue #88C0D0)
    car_color = (136, 192, 208)  # RGB
    outline_color = (255, 255, 255)  # white

    # Create rotation matrix
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Define car body corners (centered at origin)
    half_w, half_h = width / 2, height / 2
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]
    )

    # Rotate and translate corners
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ rotation_matrix.T
    translated = rotated + np.array(pos)

    # Draw car body
    cv2.fillPoly(frame, [translated.astype(np.int32)], car_color)
    cv2.polylines(frame, [translated.astype(np.int32)], True, outline_color, 1)

    # Draw front indicator
    front_corners = np.array(
        [[half_w - 2, -3], [half_w, -3], [half_w, 3], [half_w - 2, 3]]
    )
    rotated_front = front_corners @ rotation_matrix.T
    translated_front = rotated_front + np.array(pos)
    cv2.fillPoly(frame, [translated_front.astype(np.int32)], outline_color)


def render_waypoint(
    frame: NDArray[np.uint8], pos: tuple[int, int], label: str, targeted: bool = False
):
    # Colors in RGB
    waypoint_color = (0, 150, 255)  # Cyan-ish
    white = (255, 255, 255)

    if targeted:
        # Draw outer ring
        outer_color = (100, 200, 255)  # Lighter cyan
        cv2.circle(frame, pos, 16, outer_color, -1, cv2.LINE_AA)  # Filled outer circle

        # Draw inner circle
        cv2.circle(frame, pos, 10, waypoint_color, -1, cv2.LINE_AA)  # Filled
        cv2.circle(frame, pos, 10, white, 2, cv2.LINE_AA)  # White outline
    else:
        # Draw dimmed waypoint
        dimmed_color = (150, 200, 255)  # Lighter/dimmed cyan
        cv2.circle(frame, pos, 8, dimmed_color, -1, cv2.LINE_AA)  # Filled
        cv2.circle(frame, pos, 8, white, 1, cv2.LINE_AA)  # White outline

    # Draw label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    # Get text size to center it
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_pos = (pos[0] - text_width // 2, pos[1] + text_height // 2)

    cv2.putText(
        frame, label, text_pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
    )


def render_sensor(
    frame: NDArray[np.uint8], pos: tuple[int, int], colliding: bool = False
) -> None:
    # Colors in RGB
    if colliding:
        color = (191, 97, 106)  # Red (#BF616A)
    else:
        color = (163, 190, 140)  # Green (#A3BE8C)

    cv2.circle(frame, pos, 2, color, -1, cv2.LINE_AA)
