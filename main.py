import Xlib.xobject
import numpy as np
import cv2
import mss
import Xlib
import Xlib.display
import sys

def map_tree(f, nodes: list):
    for node in nodes:
        if out := f(node):
            return out
        elif children := node.query_tree().children:
            if out := map_tree(f, children):
                return out
    return None

class Application:
    def __init__(self):
        self.map_screenshot = None
    
    def select_region(self):
        bounds = None
        # get project gorgon window bounds
        disp = Xlib.display.Display()
        root = disp.screen().root
        tree = root.query_tree()
        def find_gorgon(node):
            if hasattr(node, "get_wm_name"):
                if name := node.get_wm_name():
                    if name == "Project Gorgon":
                        geom = node.get_geometry()
                        return (geom.x, geom.y, geom.width, geom.height)
            return None
        bounds = map_tree(find_gorgon, tree.children)
        if bounds is None:
            print("Project Gorgon window not found!")
            sys.exit(1)
        # screenshot the window
        with mss.mss() as sct:
            screenshot = sct.grab(bounds)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # display the screenshot
        cv2.imshow("Select Region", img)

        # allow user to drag-select a region and confirm when done
        self.region = cv2.selectROI("Select Region", img, fromCenter=False, showCrosshair=True)

        # destroy that window
        cv2.destroyWindow("Select Region")


# region select
#  1. screenshot the window
#  2. display the screenshot
#  3. allow user to drag-select a region and confirm when done
#  4. return the region selected
def select_region(state: State) -> State:
    bounds = None
    # get project gorgon window bounds
    disp = Xlib.display.Display()
    root = disp.screen().root
    tree = root.query_tree()
    def find_gorgon(node):
        if hasattr(node, "get_wm_name"):
            if name := node.get_wm_name():
                if name == "Project Gorgon":
                    geom = node.get_geometry()
                    return (geom.x, geom.y, geom.width, geom.height)
        return None
    bounds = map_tree(find_gorgon, tree.children)
    if bounds is None:
        print("Project Gorgon window not found!")
        sys.exit(1)
    # screenshot the window
    with mss.mss() as sct:
        screenshot = sct.grab(bounds)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # display the screenshot
    cv2.imshow("Select Region", img)

    # allow user to drag-select a region and confirm when done
    region = cv2.selectROI("Select Region", img, fromCenter=False, showCrosshair=True)

    cv2.destroyAllWindows()
    return region



def main():
    print("Select the region to capture...")
    region = select_region()
    A
    print(region)

if __name__ == "__main__":
    main()