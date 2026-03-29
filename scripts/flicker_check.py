import numpy as np

def calculate_flicker_intensity(frame1, frame2):
    """Calculates the brightness difference between two frames."""
    # Convert frames to average brightness (0-255)
    avg1 = np.mean(frame1)
    avg2 = np.mean(frame2)
    
    # The absolute difference is our 'flicker' signal
    return abs(avg1 - avg2)

# SIMULATION:
# Let's pretend we have a 10-frame video
# Safe Video = [100, 102, 101, 100, 103...] (Steady brightness)
# Trigger Video = [255, 0, 255, 0, 255...] (Strobe effect)

trigger_video = [np.full((64, 64), 255), np.full((64, 64), 0)] * 5

for i in range(len(trigger_video) - 1):
    intensity = calculate_flicker_intensity(trigger_video[i], trigger_video[i+1])
    if intensity > 100:
        print(f"Frame {i} to {i+1}: DANGER! High intensity flicker detected ({intensity})")
    else:
        print(f"Frame {i} to {i+1}: Safe.")