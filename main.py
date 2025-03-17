import Xlib.xobject
import numpy as np
import cv2
import mss
import Xlib
import Xlib.display
import sys
import networkx as nx
import time
import pyautogui as pag

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
        self.map_ss = None
        self.windowbounds = None
        # get project gorgon window bounds
        disp = Xlib.display.Display()
        root = disp.screen().root
        tree = root.query_tree()
        def find_gorgon(node):
            if hasattr(node, "get_wm_name"):
                if name := node.get_wm_name():
                    if name == "Project Gorgon":
                        geom = node.get_geometry()
                        print(f"Found Project Gorgon window: {geom.x}, {geom.y}, {geom.width}, {geom.height}")
                        return (geom.x, geom.y, geom.width, geom.height)
            return None
        self.windowbounds = map_tree(find_gorgon, tree.children)
        self.windowbounds = mss.mss().monitors[1]
        if self.windowbounds is None:
            print("Project Gorgon window not found!")
            sys.exit(1)
        
        #self.select_region()
        self.region = (6, 63, 522, 544) # where i like to put *my* map lol
        print (f"Selected region: {self.region}")
    
    def select_region(self):
        self.capture_window()
        # allow user to drag-select a region and confirm when done
        self.region = cv2.selectROI("Select Region", self.winimg, fromCenter=False, showCrosshair=True)
        # destroy that window
        cv2.destroyWindow("Select Region")
    
    def capture_region(self):
        with mss.mss() as sct:
            screenshot = sct.grab(self.windowbounds)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.map_ss = img[int(self.region[1]):int(self.region[1] + self.region[3]), int(self.region[0]):int(self.region[0] + self.region[2])]

    def capture_window(self):
        self.winimg = None
        with mss.mss() as sct:
            win = np.array(sct.grab(self.windowbounds))
            self.winimg = cv2.cvtColor(win, cv2.COLOR_BGRA2BGR)

    def click_all_maps(self):
        self.capture_window()
        # find all instances of 'mapicon.png' in the window screenshot
        template = cv2.imread('mapicon.png', cv2.IMREAD_UNCHANGED)
        w, h = template.shape[1], template.shape[0]
        result = cv2.matchTemplate(self.winimg, template, cv2.TM_CCOEFF_NORMED)
        maxval = np.max(result)

        threshold = 0.98
        result = cv2.threshold(result, threshold, 1.0, cv2.THRESH_BINARY)[1]
        cv2.imshow("Map mask", result)

        loc = np.where(result >= threshold)
        # add the width and height of the template to the x and y coordinates
        loc = list(zip(*loc[::-1]))
        surveys = [(x + w//2, y + h//2) for (x, y) in loc]
        
        self.points = []
        for i, pt in enumerate(surveys):
            # click on the map icon
            x,y = pt[0] + self.windowbounds['left'], pt[1] + self.windowbounds['top']

            print(f"Clicking on survey {i} at {x}, {y}")
            # simulate a mouse right-click at the location of the map icon
            
            pag.moveTo(x, y, 0.05)
            time.sleep(0.01); pag.click(x, y, button='right')
            pag.moveRel(-110,-125)
            time.sleep(0.01); pag.click(button='left')
            time.sleep(1.0)

            # capture the map region
            self.register_point()
        
        # show the map with the points
        # for x,y in self.points:
        #     cv2.circle(self.map_ss, (x, y), 5, (0, 255, 255), 2)
        #     
        # cv2.imshow("Map", self.map_ss)
        # cv2.waitKey(0)

    def register_point(self):
        self.capture_region()
        lower = np.array([0, 0, 170])
        upper = np.array([100, 100, 255])
        masked = cv2.inRange(self.map_ss, lower, upper)
        #cv2.imshow("Map", masked)#cv2.bitwise_and(self.map_ss, self.map_ss, mask=masked))
        # find the centroid of the red pixels
        centeroid = np.argwhere(masked > 0)
        centroid = np.mean(centeroid, axis=0)
        # draw a circle on the original image at the centroid
        cv2.circle(self.map_ss, (int(centroid[1]), int(centroid[0])), 5, (0, 255, 0), -1)
        self.points.append((int(centroid[1]), int(centroid[0])))
        return (int(centroid[1]), int(centroid[0]))
    
    def run_tsp(self):
        # run tsp
        G = nx.Graph()
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                dist = np.linalg.norm(np.array(self.points[i]) - np.array(self.points[j]))
                G.add_edge(i, j, weight=dist)
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
        self.tsp_path = [i for i in tsp_path]

    def register_points_manual(self):
        self.points = []

        print("Double-click surveys from left-to-right, top-to-bottom. Press Spacebar after each one, and Esc when done.")

        self.capture_region()
        cv2.imshow("Map", self.map_ss)
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                break
            elif key == 32: # Spacebar
                point = self.register_point()
                for x,y in self.points:
                    cv2.circle(self.map_ss, (x, y), 5, (0, 255, 255), 2)
                cv2.circle(self.map_ss, point, 5, (0, 255, 0), 2)
                cv2.imshow("Map", self.map_ss)
                print(f"Registered point: {self.points[-1]}")
            elif key == 13:
                self.run_tsp()
                break
            
        
    def show_path(self, hi=None):
        self.capture_region()
        for i,(x,y) in enumerate(self.points):
            cv2.circle(self.map_ss, (x, y), 5, (0, 255, 255), 1)
            cv2.putText(self.map_ss, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(self.map_ss, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for p1,p2 in zip(self.tsp_path, self.tsp_path[1:] + [self.tsp_path[0]]):
            cv2.line(self.map_ss, self.points[p1], self.points[p2], (255, 0, 0), 2)
        if hi is not None:
            cv2.circle(self.map_ss, self.points[hi], 10, (255, 0, 0), 2)
        cv2.imshow("Map", self.map_ss)

    def follow_path(self):
        selected_point = 0
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_point
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, (px, py) in enumerate(self.points):
                    if abs(px - x) < 10 and abs(py - y) < 10:
                        selected_point = i
                        break
                print(f"Selected point: {selected_point}")
        cv2.namedWindow("Map")
        cv2.setMouseCallback("Map", mouse_callback)
        while True:
            self.show_path(selected_point)
            key = cv2.waitKey(1)
            if key == ord('d'):
                # delete the selected point
                if len(self.points) > 1:
                    self.points.pop(selected_point)
                    selected_point = 0
                    self.run_tsp()
                    continue
            elif key == ord('n'):
                selected_point = (selected_point + 1) % len(self.points)
            elif key == 'p':
                selected_point = (selected_point - 1) % len(self.points)
            elif key == 27:  # ESC key
                break



    
def main():
    print("Select the region to capture...")
    app = Application()
    app.click_all_maps()
    app.run_tsp()
    app.follow_path()
    # region = app.select_region()
    # app.register_points()
if __name__ == "__main__":
    main()