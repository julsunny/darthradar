from dataloader import RadarImageTargetSet
import peakDetection
import clustering_lib
import detection
import numpy as np
import plotting_stuff


#%%
if True or __name__ == "__main__":
    # create dataset
    ds = RadarImageTargetSet()

    # decorate dataset with box predictions
    for (img, target) in ds:
        # find peaks
        predicted_number_of_peaks, peaks_xs, peaks_ys, peak_strengths = peakDetection.detect_peaks(img, tx=10, ty=3, gx=2, gy=1, rate_fa=1e-2)

        # build large boxes around peaks
        predicted_boxes = peakDetection.return_box_bounds(peaks_xs, peaks_ys, img, 11, 27)
        assert predicted_number_of_peaks == len(predicted_boxes)
        
        # refine boxes using spectral graph clustering
        refined_boxes = np.array([clustering_lib.refine_box(img, box, strength, n_objects = 1, color_scaling = 1, gradient = False)[0] for box, strength in zip(predicted_boxes, peak_strengths)])

        # add refined boxes to dataset
        target['refined_boxes'] = refined_boxes

        # how much have we over-/ undercounted 
        target['overcount'] = len(target['refined_boxes']) - len(target['boxes'])

        # match predicted boxes to labeled boxes and calculate average intersection over union of box pairs
        labeled_boxes = target['boxes']
        goodness = detection.evaluate_boxes_pair(refined_boxes, labeled_boxes)
        target['box_goodness'] = goodness
#%%
sample_number = 8
samples = []
for i, (img, target) in enumerate(ds):
    samples.append((img, target))
    if i == sample_number:
        break
    
plotting_stuff.grid_plot(samples)

#%%
print(samples[0][1]["refined_boxes"])



