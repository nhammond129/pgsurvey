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
            # sys.exit(1)
            bounds = (0, 0, 1920, 1080)  # temp for debugging :)
        # screenshot the window
        with mss.mss() as sct:
            screenshot = sct.grab(bounds)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # display the screenshot
        cv2.imshow("Select Region", img)

        # allow user to drag-select a region and confirm when done
        self.region = cv2.selectROI("Select Region", img, fromCenter=False, showCrosshair=True)

        # crop the screenshot to the selected region
        self.map_screenshot = img[int(self.region[1]):int(self.region[1] + self.region[3]), int(self.region[0]):int(self.region[0] + self.region[2])]

        # destroy that window
        cv2.destroyWindow("Select Region")
    
    def select_points(self):
        mapcopy = self.map_screenshot.copy()
        print("Select the points to capture...")
        cv2.imshow("Select Points", mapcopy)
        self.points = []
        def callback(event, x, y, flags, param):
            print(event, x, y, flags, param)
            if event == cv2.EVENT_LBUTTONUP:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                cv2.circle(mapcopy, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Points", mapcopy)
        cv2.setMouseCallback("Select Points", callback)

        while True:

            key = cv2.waitKey(0)
            if key == ord("q"):
                self.points = []
                break
            elif key == ord("r"):
                self.points = []
            elif key == ord(" "):
                break

            # if window gone, break
            if cv2.getWindowProperty("Select Points", cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyWindow("Select Points")


def main():
    print("Select the region to capture...")
    app = Application()
    region = app.select_region()
    app.select_points()
if __name__ == "__main__":
    main()