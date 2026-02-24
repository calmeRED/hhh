import pyautogui


def press_scroll_lock():
    """模拟按下 Scroll Lock 键"""
    pyautogui.press('scrolllock')
    # pyautogui.moveTo(100, 100)
    # pyautogui.moveTo(500, 500)
    pyautogui.FAILSAFE = False


def K_to_C(values):
    return [v - 273.15 for v in values]
def C_to_K(values):
    return [v + 273.15 for v in values]