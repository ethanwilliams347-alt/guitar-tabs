"""Synthetic tab images for unit tests. Every position is known exactly."""
import cv2
import numpy as np

TAB_TOP_Y = 375          # first tab line occupies rows 375 and 376, so its centroid is 375.5
TAB_GAP = 18
LINE_THICK = 2
STAFF_TOP_Y = 225
STAFF_GAP = 14


def blank(h=484, w=1920):
    return np.full((h, w), 255, np.uint8)


def tab_line_ys(top=TAB_TOP_Y, gap=TAB_GAP):
    return [top + k * gap for k in range(6)]


def hline(img, y, x0, x1, thick=LINE_THICK, level=0):
    img[y:y + thick, x0:x1] = level


def vline(img, x, y0, y1, thick=2, level=0):
    img[y0:y1, x:x + thick] = level


def draw_staff(img, x0, x1, top=STAFF_TOP_Y, gap=STAFF_GAP):
    for k in range(5):
        hline(img, top + k * gap, x0, x1, thick=1)


def draw_chord_grids(img, xs, top=87, gap=18, width=85):
    """Chord diagrams: 5 short horizontal lines and 6 verticals each."""
    for x in xs:
        for k in range(5):
            hline(img, top + k * gap, x, x + width, thick=1)
        for k in range(6):
            vline(img, x + k * width // 5, top, top + 4 * gap, thick=1)


def draw_tab_lines(img, x0, x1, ys=None):
    for y in ys or tab_line_ys():
        hline(img, y, x0, x1)


def draw_digit(img, x, y_line, text, scale=0.6):
    """A fret number centred on a line, over a white box that blanks the line."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cy = y_line + LINE_THICK // 2
    img[cy - th // 2 - 2:cy + th // 2 + 3, x - tw // 2 - 3:x + tw // 2 + 4] = 255
    cv2.putText(img, text, (x - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_8)


def draw_bar(img, x, ys=None, thick=2, extend_up=0):
    ys = ys or tab_line_ys()
    vline(img, x, ys[0] - extend_up, ys[-1] + LINE_THICK, thick=thick)


def draw_arpeggio(img, xc, ys=None, amp=1.5, period=6.0, thick=3):
    """A thick wavy stroke with an arrowhead above the top line. Its centre column is fully inked."""
    ys = ys or tab_line_ys()
    for y in range(ys[0] - 2, ys[-1] + LINE_THICK):
        x = xc + int(round(amp * np.sin(2 * np.pi * y / period)))
        img[y, x - thick // 2:x - thick // 2 + thick] = 0
    tip = ys[0] - 12
    pts = np.array([[xc, tip], [xc - 5, tip + 12], [xc + 5, tip + 12]], np.int32)
    cv2.fillPoly(img, [pts], 0)


def tab_page(w=1920, h=484, line_x0=113, line_x1=1880, bars=(600, 822, 1176, 1306),
             arpeggios=(470, 1047), digits_every=60, bracket_x=99, start_line=True, final_double=None):
    """A page like iteration 1's: chord grids, a staff and one tab row broken by fret numbers.

    Returns (image, truth) where truth holds the exact positions drawn.
    """
    img = blank(h, w)
    draw_chord_grids(img, range(270, w - 150, 300))
    draw_staff(img, line_x0, line_x1)
    ys = tab_line_ys()
    draw_tab_lines(img, line_x0, line_x1, ys)
    if bracket_x is not None:
        vline(img, bracket_x, 205, 480, thick=7)
    for x in bars:
        draw_bar(img, x, ys, extend_up=ys[0] - STAFF_TOP_Y)
    bar_centres = [x + 0.5 for x in bars]
    if start_line:
        vline(img, line_x0, STAFF_TOP_Y, ys[-1] + LINE_THICK, thick=2)
        bar_centres = [line_x0 + 0.5] + bar_centres
    if final_double is not None:
        thin, thick_x = final_double, final_double + 7
        draw_bar(img, thin, ys)
        draw_bar(img, thick_x, ys, thick=6)
        bar_centres.append((thin + thick_x + 5) / 2)
    for x in arpeggios:
        draw_arpeggio(img, x, ys)
    avoid = list(bars) + list(arpeggios) + ([final_double] if final_double else [])
    for k, x in enumerate(range(line_x0 + 60, line_x1 - 30, digits_every)):
        if all(abs(x - a) > 25 for a in avoid):
            for s, y in enumerate(ys):
                if (k + s) % 2 == 0:
                    draw_digit(img, x, y, str((k * 7 + s) % 13))
    truth = {"line_y": [y + (LINE_THICK - 1) / 2 for y in ys], "s": float(TAB_GAP),
             "left": line_x0, "right": line_x1 - 1, "bar_x": bar_centres}
    return img, truth


def fade_edges(img, left_px=0, right_px=0):
    """Blend the outer columns towards white, linearly, as the video fades tab at a page edge."""
    out = img.astype(np.float64)
    w = img.shape[1]
    for k in range(left_px):
        a = (k + 1) / (left_px + 1)                  # 0 = white at the edge, 1 = untouched inside
        out[:, k] = a * out[:, k] + (1 - a) * 255
    for k in range(right_px):
        a = (k + 1) / (right_px + 1)
        out[:, w - 1 - k] = a * out[:, w - 1 - k] + (1 - a) * 255
    return np.round(out).astype(np.uint8)


def strip_pages(strip, xs, w, fade_px=0):
    """Windows of width w cut from a long strip at xs, faded on every edge where the music continues."""
    last = len(xs) - 1
    return [fade_edges(strip[:, x:x + w], fade_px if k > 0 else 0, fade_px if k < last else 0)
            for k, x in enumerate(xs)]
