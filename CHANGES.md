# Changes

Short log of the changes made to the project and why.

## 1. Camera setup for Camera Module 3 Wide
**File:** `src/yolo_raspPi.py`

**What changed:**
- Capture size changed from `640x480` to `640x360`.
- Added `sensor={"output_size": (2304, 1296)}` so the camera uses the whole sensor.
- Turned on continuous autofocus (`AfMode Continuous`).
- Changed the format to `RGB888` and removed the `cvtColor` line.

**Why:**
- The Camera Module 3 Wide has a 16:9 sensor with a ~120° view. A 4:3 size (640x480) cuts off the sides, so we were losing part of the wide view.
- This camera has autofocus. Continuous mode keeps objects sharp as they get closer.
- In picamera2, `RGB888` already gives the frame in BGR order (what OpenCV and YOLO use), so the color conversion isn't needed anymore.

## 2. Direction and distance don't depend on frame size
**File:** `src/analyze.py`

**What changed:**
- Removed the `FRAME_W = 640` and `FRAME_H = 480` constants.
- `get_directie` and `get_distanta` now use `box.xyxyn` (coordinates from 0 to 1) instead of `box.xyxy` (pixels).

**Why:**
- The frame size changed to 640x360, so the old constants gave wrong results for near/far and left/right.
- With normalized coordinates the code works at any camera resolution, and nothing needs updating when the size changes.

## 3. Human class (and pillar)
**File:** `src/analyze.py`

**What changed:**
- Replaced `CLASS_NAMES` (number -> name) with `NUME_SCURT` (model name -> short name).
- `analizeaza(boxes, names)` now gets the class names from the model (`r.names`).
- Added `human` and `pillar`. `pillar` is mapped to `stalp`, like the two post classes.

**Why:**
- The old model had no human class, so the people logic never ran.
- Roboflow sorts classes alphabetically, so adding a class changes the numbers of the others. Looking classes up by name keeps working after every retrain.
- For someone walking, a pillar is the same kind of obstacle as a post, so it reuses the `stalp_*.mp3` files.

## 4. Objects are not skipped anymore
**File:** `src/yolo_raspPi.py`

**What changed:**
- Replaced `obiecte_anterioare` with `anuntate` (objects already announced that are still in view).
- An object goes into `anuntate` only when it is actually spoken. Objects that leave the view are removed, so they get announced again if they come back.

**Why:**
- Before, an object that appeared during the 2 second pause (or while talking) was saved as "seen" and never announced.

## 5. Priority 3 for near + straight ahead
**File:** `src/analyze.py`

**What changed:**
- New function `get_prioritate`: 3 = near and in front, 2 = near on the side, 1 = far.

**Why:**
- An object right in front of you is the most dangerous, so it should be announced first.

## 6. Near threshold per object
**File:** `src/analyze.py`

**What changed:**
- New dict `PRAG_APROAPE` with a threshold for each object. `get_distanta(box, nume)` uses it instead of a fixed 0.4.

**Why:**
- A bench is short and a post is tall, so one threshold didn't fit all of them. The numbers are starting values and need tuning with the camera.

## 7. `imgsz=512` in predict
**File:** `src/yolo_raspPi.py`

**What changed:**
- `model.predict(frame, imgsz=512, ...)`

**Why:**
- The model was trained and exported at 512, so the input size now matches it.
