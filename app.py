import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Helper Functions ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# --- Streamlit App ---
st.set_page_config(page_title=" Document Scanner", layout="wide")
st.title(" Automated Document Scanner")

# File uploader (drag and drop or select)
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig = image.copy()

    # --- Document Detection ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        st.warning("Could not detect document edges. Try a clearer background.")
    else:
        # --- Warp & Scan ---
        warped = four_point_transform(orig, screenCnt.reshape(4,2))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # --- Display Results Side by Side ---
        col1, col2 = st.columns(2)

        with col1:
            # Draw the contour on original
            cv2.drawContours(image, [screenCnt], -1, (0,255,0), 5)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original with Detected Edges")

        with col2:
            st.image(scanned, caption="Scanned Result", use_column_width=True)
