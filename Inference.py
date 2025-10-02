import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

import shutil

from scipy.spatial import ConvexHull
from scipy.stats.qmc import Halton
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import os
import csv
import random
import glob
from skimage.draw import polygon
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.colors as mcolors
import warnings
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from PIL import Image, ImageDraw

def generate_candidate_points(polygon_hull):
    candidates = []
    x_min, y_min, x_max, y_max = polygon_hull.bounds
    for i in range(0,2000):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        if polygon_hull.contains(point):
            candidates.append(point)
    
    return candidates

def evaluate_coverage(candidates, polygon_hull, radius):
    scored_candidates = []
    for point in candidates:
        circle = point.buffer(radius)
        intersection_area = polygon_hull.intersection(circle).area
        scored_candidates.append((intersection_area, point))
    
    return sorted(scored_candidates, reverse=True, key=lambda x: x[0])

def select_circle_centers(scored_candidates, n, radius):
    selected_centers = []
    
    for _, point in scored_candidates:
        if not selected_centers:  # First selection, no need to check distance
            selected_centers.append(point)
        else:
            used_tree = KDTree([p.coords[0] for p in selected_centers])
            if used_tree.query(point.coords[0], k=1, distance_upper_bound=2 * radius)[0] == np.inf:
                selected_centers.append(point)

        if len(selected_centers) >= n:
            break
    
    return selected_centers

def find_optimal_circle_centers(polygon, n, radius):
    candidates = generate_candidate_points(polygon)
    scored_candidates = evaluate_coverage(candidates, polygon, radius)
    centers = select_circle_centers(scored_candidates, n, radius)
    
    return [center.coords[0] for center in centers]


def evaluate_coverage_p(candidates, points_to_cover, radius):
    if len(points_to_cover) < 1:
        return []
    
    # Convert points_to_cover into a numpy array of points (2D)
    points_array = np.array([(p[0], p[1]) for p in points_to_cover])
    
    # Create the KDTree from the points
    tree = KDTree(points_array)
    
    scored_candidates = []
    for point in candidates:
        # Query the KDTree to get indices of points within the radius
        indices = tree.query_ball_point([point.x, point.y], radius)
        
        # Append the count of points within the radius and the candidate point
        scored_candidates.append((len(indices), point))
    
    # Sort candidates by the number of covered points in descending order
    return sorted(scored_candidates, reverse=True, key=lambda x: x[0])

def select_circle_centers_p(scored_candidates, n, radius):
    selected_centers = []
    #used_tree = KDTree([])
    
    for count, point in scored_candidates:
        if len(selected_centers) == 0:
            selected_centers.append(point)
            used_tree = KDTree([point.coords[0]])
        else:
            distance, _ = used_tree.query(point.coords[0], k=1, distance_upper_bound=2*radius)
            if distance > 2 * radius:
                selected_centers.append(point)
                coords = [p.coords[0] for p in selected_centers]
                used_tree = KDTree(coords)
        if len(selected_centers) >= n:
            break
    return selected_centers

def find_optimal_circle_centers_p(polygon, points_to_cover, n, radius):
    candidates = generate_candidate_points(polygon)
    scored_candidates = evaluate_coverage_p(candidates, points_to_cover, radius)
    centers = select_circle_centers_p(scored_candidates, n, radius)
    return [center.coords[0] for center in centers]


class SAMInference:
    def __init__(self,checkpoint="../sam_vit_b_01ec64.pth",model_type="vit_b",output_path='/media/vicorob/Filesystem2/YC/field_imagery/plot1_m1_250218_24mm_cc/masks'):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.output_path  = output_path
        os.makedirs(self.output_path, exist_ok=True)
        device = "cuda"
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.color_map_path = '/media/vicorob/Filesystem2/YC/field_imagery/color_mapping.csv'
        self.minmun_size = 2000
        if os.path.exists(self.color_map_path):
            map_file = open(self.color_map_path,'r')
            reader = csv.reader(map_file)
            self.color_map ={}
            self.color_list=[]
            for row in reader:
                class_name,hex_color = row
                color = mcolors.hex2color(hex_color)
                rgb_color = tuple(int(c * 255) for c in color)
                self.color_map[class_name] = rgb_color
                self.color_list.append(rgb_color)
            
        else:
            self.color_map ={}
            self.color_list=[]


    def ReadPoint(self,csv_path):
        # f = open(txt,'r',encoding='utf-8')
        csv_file = open(csv_path,mode='r')
        csv_reader = csv.reader(csv_file)
        points=[]
        class_list=[]
        # for line in f.readlines():
        #     cl,x,y=line.strip().split(' ')
        #     x = int(float(x))
        #     y = int(float(y))
        #     points.append([x,y])
        #     class_list.append(cl)
        for row in csv_reader:
            # print(row)
            cl,x,y = row
            if "png" in cl:
                continue
            x = int(float(x))
            y = int(float(y))
            points.append([x,y])
            class_list.append(cl)
        points =np.array(points)
        class_list =np.array(class_list)
        return points,class_list

    def Mean_shift(self,points):
        bandwidth = estimate_bandwidth(points, quantile=0.7, n_samples=500)
        meanshift = MeanShift(bandwidth=bandwidth,seeds=points[::500])
        labels = meanshift.fit_predict(points)
        centroids = meanshift.cluster_centers_
        return labels,centroids

    def MaskIOU(self,sam_mask,binary_hull_mask):

        intersection = np.logical_and(sam_mask, binary_hull_mask)
        union = np.logical_or(sam_mask, binary_hull_mask)
        # Compute IoU
        IOU = np.sum(intersection) / np.sum(union)
        return IOU

    def PointGenerator(self,polygon_hull,class_points=None,mode='p', num = 3 ,min_distance = 3):
        valid_points = []
        times = 0
        while len(valid_points)<1:
            x_min, y_min, x_max, y_max = polygon_hull.bounds
            x_min = x_min 
            y_min = y_min 
            x_max = x_max 
            y_max = y_max 
            # Halton
            #halton_sampler = Halton(d=2,optimization='lloyd')
            #halton_points = halton_sampler.random(5 * 2)
            #halton_points[:, 0] = halton_points[:, 0] * (x_max - x_min) + x_min
            #halton_points[:, 1] = halton_points[:, 1] * (y_max - y_min) + y_min

            #for point in halton_points:
            #    point_obj = Point(point)
            #    if polygon_hull.contains(point_obj) and polygon_hull.exterior.distance(point_obj) > min_distance:
            #        valid_points.append(point)


            #Maximin
            #x_min, y_min, x_max, y_max = polygon_hull.bounds
            #candidates = []
            #num_candidates = 1000
            #while len(candidates) < num_candidates:
            #    x = np.random.uniform(x_min+30, x_max-20)
            #    y = np.random.uniform(y_min+30, y_max-20)
            #    point = Point(x, y)
            #    if polygon_hull.contains(point):
            #        candidates.append([x, y])
            
            #candidates = np.array(candidates)
            #sampled_points = [candidates[np.random.randint(len(candidates))]] 
            #num_samples = 3
            #for _ in range(num_samples - 1):
            
             #   distances = cdist(candidates, np.array(sampled_points))  
             #   min_distances = np.min(distances, axis=1)  
                

              #  best_idx = np.argmax(min_distances)
              #  sampled_points.append(candidates[best_idx])
              #  valid_points.append(candidates[best_idx])
            
            #Circle cover
            #print(polygon_hull.area)
            if polygon_hull.area >1000000:
                radius = 150  # Circle radius
                n =8
            elif polygon_hull.area >500000:
                n =5
                radius = 150  # Circle radius
            elif polygon_hull.area >100000:
                n =3
                radius = 100  # Circle radius
            else :
                n=2
                radius = 90  # Circle radius
             ## Number of circles
            
            #print(f'n:{n}')
            if mode == 'p':
                valid_points = find_optimal_circle_centers_p(polygon_hull,class_points, n, radius)
            else:
                 valid_points = find_optimal_circle_centers(polygon_hull, n, radius)
            times +=1
            if times>10:
              return None
        #valid_points = valid_points[:num]
        return valid_points

    def Inference(self,txt):
        image_path =txt.replace('.csv','.jpg').replace('annotations','images')
        print(image_path)
        if not os.path.exists(image_path):
            return None
        points,labels =self.ReadPoint(txt)
        # labels,_ = self.Mean_shift(points)
        unique_labels = np.unique(labels)
        masks_total = []
        scores_total = []
        color_total = []
        label_total = []
        ori_image = cv2.imread(image_path)
        draw_img= Image.open(image_path)
        draw = ImageDraw.Draw(draw_img)
        for pt in points:
            x,y = pt
            x, y = map(int, (x,y))
            draw.ellipse((x - 10, y - 10, x + 10, y + 10),fill=(255,255,0))
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        for label in unique_labels:
            if 'chain'in label or 'scale'in label:
                mode = 'p'
                IOU_t = 0.1
            else:
                mode = 'l'
                IOU_t = 0.5
            class_points = points[labels == label]
            # label_index = int(label)
            binary_hull_mask = None
            class_points=np.unique(class_points, axis=0)
            point_list = None
            try:
              
                hull = ConvexHull(class_points)
                hull_vertices = class_points[hull.vertices]
                polygon_hull = Polygon(hull_vertices)
                polygon_points = [tuple(point) for point in hull_vertices]
                
                draw.polygon(polygon_points, outline="green", width=13) 

                mask_shape = image.shape[:2]
                binary_hull_mask = np.zeros(mask_shape, dtype=np.uint8)
                polygon_coords = np.array(polygon_hull.exterior.coords)
                rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], shape=mask_shape)

                # Mark the polygon area in the binary mask
                binary_hull_mask[rr, cc] = 1

                # plt.figure(figsize=(8, 8))
                # plt.imshow(binary_hull_mask)
                # plt.title("binary_hull_mask")
                # plt.axis('off')
                # plt.show()

                point_list = self.PointGenerator(polygon_hull,class_points,mode=mode)
            except:
                print(len(class_points))
                if(len(class_points)<3):
                    continue
                try:
                   point_list = random.sample(class_points, 1)  
                except:
                   continue
            #print(f'point:{point_list}')
            if point_list is None:
                continue
            
            for prompt in point_list:
                x,y = prompt
                x, y = map(int, (x,y))
            
                draw.ellipse((x - 20, y - 20, x + 20, y + 20),fill=(0,255,255))
                #print('draw')

            


            label_list = [1]*len(point_list)
            input_point = np.array(point_list)
            input_label = np.array(label_list)
            sam_mask, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,)
            if label.lstrip('0') not in self.color_map.keys():
                colors = np.random.rand(1,3)
                hex_color = mcolors.to_hex(colors) 

                color = mcolors.hex2color(hex_color)
                with open(self.color_map_path,mode='a+',newline='') as csv_file:
                    colorwriter = csv.writer(csv_file)
                    colorwriter.writerow([str(label.lstrip('0')),str(hex_color)])
                rgb_color = tuple(int(c * 255) for c in color)
                self.color_map[label.lstrip('0')] = rgb_color
                self.color_list.append(rgb_color)
            
            if binary_hull_mask is not None:
              IOU = self.MaskIOU(sam_mask,binary_hull_mask)
              

              if IOU>0.2:
                print(f'IOU:{IOU:.2f}')
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(binary_hull_mask, kernel, iterations=50)
                sam_mask = np.logical_and(sam_mask, dilated_mask)
                #sam_mask = dilated_mask
                masks_total.append(sam_mask)
                scores_total.append(scores)
                # current_color = self.color_list[label_index]
                current_color = self.color_map[label.lstrip('0')]
                color_total.append(current_color)
                # print(current_color)
                colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
                sam_mask = sam_mask.squeeze()
                
 
                for c in range(3):  # For each channel in RGB
                    colored_mask[:, :, c] += (sam_mask.astype(np.uint8) * current_color[c])
                
                # plt.figure(figsize=(10,10))
                # plt.imshow(colored_mask)
                # # # show_mask(masks, plt.gca())
                # # show_points(input_point, input_label, plt.gca())
                # plt.axis('off')
                # # name = name.replace('_marked.jpg','_vit_b.jpg')
                # # plt.savefig(os.path.join(output_path, name), bbox_inches='tight', pad_inches=0)
                # plt.show()

            else:
              masks_total.append(sam_mask)
              scores_total.append(scores)
              current_color = self.color_map[label.lstrip('0')]
              color_total.append(current_color)
              print(current_color)
              colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
              sam_mask = sam_mask.squeeze()
              num_labels, labels_m, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
              min_size = 20000
              filtered_mask = np.zeros_like(sam_mask)
              for label_m in range(1, num_labels):  # 从1开始，0是背景
                if stats[label_m, cv2.CC_STAT_AREA] >= min_size:
                    filtered_mask[labels_m == label_m] = 255
              for i in range(0,50,3):
                    kernel = np.ones((i, i), np.uint8)  
                    filled_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

              for c in range(3):  # For each channel in RGB
                colored_mask[:, :, c] += (filled_mask.astype(np.uint8) * current_color[c])
        # print(f'masks_total:{len(masks_total)}')
        if len(masks_total)==1:
            mask = masks_total[0]
            mask =mask.squeeze()
            color = color_total[0]
            colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
                # print(color)
                # Apply the mask to each color channel (R, G, B)s
            num_labels, labels_m, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            min_size = self.minmun_size
            filtered_mask = np.zeros_like(mask)
            for label_m in range(1, num_labels):  # 从1开始，0是背景
                print(f'size:{stats[label_m, cv2.CC_STAT_AREA]}')
                if stats[label_m, cv2.CC_STAT_AREA] >= min_size:
                    filtered_mask[labels_m == label_m] = 255
            for i in range(0,50,3):
                kernel = np.ones((i, i), np.uint8)  
                filled_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            for c in range(3):  # For each channel in RGB
                colored_mask[:, :, c] += (filled_mask.astype(np.uint8) * color[c])
            output_name = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','.png'))
            cv2.imwrite(output_name,colored_mask)
            alpha = 0.6  # transparency factor for the mask overlay
            overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

            # Plotting the original image with the mask overlay
            output_name_mask = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_mask.jpg'))
            cv2.imwrite(output_name_mask,overlayed_image)
            output_image_path = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_pts.jpg'))
            print(output_image_path)
            draw_img.save(output_image_path)


        elif len(masks_total)>1:
          
          mask_shape = masks_total[0].squeeze().shape  # Get the height and width of a single mask
          colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)  # RGB mask with shape (682, 1024, 3)

          # Use colormap or manually define colors
          #colors = [np.array([random.randint(0, 255) for _ in range(3)]) for _ in range(len(masks_total))]

          # Combine masks and apply colors
          for i, mask in enumerate(masks_total):
                mask = mask.squeeze()  # Ensure mask is 2D (height x width)
                color = color_total[i]
                # print(color)
                # Apply the mask to each color channel (R, G, B)s
                num_labels, labels_m, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                min_size = self.minmun_size
                filtered_mask = np.zeros_like(mask)
                for label_m in range(1, num_labels):  # 从1开始，0是背景
                    # print(f'size:{stats[label_m, cv2.CC_STAT_AREA]}')
                    if stats[label_m, cv2.CC_STAT_AREA] >= min_size:
                        filtered_mask[labels_m == label_m] = 255
                for i in range(0,50,3):
                    kernel = np.ones((i, i), np.uint8)  
                    filled_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                
                for c in range(3):  # For each channel in RGB
                    colored_mask[:, :, c] += (filled_mask.astype(np.uint8) * color[c])

          output_name = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','.png').replace('.JPG','.png'))
          cv2.imwrite(output_name,colored_mask)
          alpha = 0.6  # transparency factor for the mask overlay
          overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

          # Plotting the original image with the mask overlay
          output_name_mask = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_mask.jpg').replace('.JPG','_mask.jpg'))
          cv2.imwrite(output_name_mask,overlayed_image)
          print(output_name_mask)

          output_image_path = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_pts.jpg').replace('.JPG','_pts.jpg'))
          print(output_image_path)
          draw_img.save(output_image_path)

        #   output_image_path = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_pts.jpg').replace('.JPG','_ori.jpg'))
        #   shutil.copyfile(image_path,output_image_path)
         
          # Plotting the colored masks
          # alpha = 0.6  # transparency factor for the mask overlay
          # overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

          # # Plotting the original image with the mask overlay
          # plt.figure(figsize=(8, 8))
          # plt.imshow(overlayed_image)
          # plt.title("output")
          # plt.axis('off')
          # plt.show()


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=FutureWarning)
    Infer = SAMInference()
    
    # txts = ['/media/vicorob/Filesystem2/YC/field_imagery/plot2/annotations_plot2/IMG_9065.csv']
    txts = glob.glob('/media/vicorob/Filesystem2/YC/field_imagery/plot1_m1_250218_24mm_cc/annotations/*.csv')
    print(f'total:{len(txts)}')
    
    for txt in txts:
        print(txt)
        Infer.Inference(txt)
