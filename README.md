# Driver drowsiness detection — educational source

**Source archived; model efficacy and execution not verified.** Owner-reported Lebanese University Master 1 machine-learning coursework. Study status: paused; this is not a degree-completion claim.

Python/OpenCV/MediaPipe face landmarks, eye/mouth/head-pose rules, Windows alarms and CSV logging; TensorFlow/Keras, SVM, random-forest and MLP experiments are present. Existing CSV sessions, virtualenv, trained .h5/.pkl weights/scalers and private recordings are excluded. No independent accuracy claim or driving-safety assurance.

Known issues preserved: the base trainer uses [EAR, blinks_per_min, eye_closed_time], while final inference supplies [EAR, MAR, blink_rate]. Training/evaluation alternatives use self-generated status labels and random row splits; they are not independent ground-truth evaluation. The base trainer uses eight simulated samples. final_drowsiness_system.py opens drowsiness_log.csv in **w** mode and overwrites that log when run. Existing model provenance/rights are unresolved.

Setup references, **untested**: Python with a compatible environment, `pip install -r requirements.txt`. Runtime is Windows-specific through winsound. A model is intentionally absent, so launching the final script is not a working demo. Do not run the camera/training scripts against original data or overwrite evidence. No application, model training, tests or deployment were run for publication.
