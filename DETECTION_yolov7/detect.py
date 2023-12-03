import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import requests
import os

random.seed(1)

objects_detected = {}
coordinates = {}
weight = {}

# API related constants
API_NAMES = ["edamam"]
API = {
    "edamam": {
        "homepage": "https://developer.edamam.com/",
        "url": "https://api.edamam.com/api/nutrition-data",
        "auth": {
            "app_id": "e45165c7",  # Replace with your Edamam app ID
            "app_key": "6f6a07dea3cb759d2598efb4268675c5"  # Replace with your Edamam app key
        },
        "query_str": {
            "ingr": "",
            "nutrition-type": "logging",
        }
    }
}

def get_food_info_from_api(object_name):
    api_name = "edamam"
    api_dict = API[api_name]
    api_url = api_dict['url']
    api_auth = api_dict['auth']

    query_str = api_dict['query_str']
    query_str['ingr'] = object_name

    headers = {"Accept": "application/json"}

    response = requests.get(
        api_url,
        params={**api_auth, **query_str},
        headers=headers
    )

    data = response.json()
    enerc_kcal_quantity = round(data["totalNutrients"]["ENERC_KCAL"]["quantity"],2)
    procnt_quantity = round(data["totalNutrients"]["PROCNT"]["quantity"],2)
    fat_quantity = round(data["totalNutrients"]["FAT"]["quantity"],2)
    chocdf_quantity = round(data["totalNutrients"]["CHOCDF"]["quantity"],2)
    fibtg_quantity = round(data["totalNutrients"]["FIBTG"]["quantity"],2)
    # result = response_dict

    # food_info = result['food']

    # food_label = response_dict["text"]
    # food_id = food_info['foodId']
    # food_nutrients = food_info['nutrients']

    # calories = food_nutrients['ENERC_KCAL']
    # protein = food_nutrients['PROCNT']
    # fat = food_nutrients['FAT']
    # carbs = food_nutrients['CHOCDF']
    # fiber = food_nutrients['FIBTG']

    return {
        "name": object_name,
        "nutrients": {
            "calories": enerc_kcal_quantity,
            "protein": procnt_quantity,
            "fat": fat_quantity,
            "carbs": chocdf_quantity,
            "fiber": fibtg_quantity
        }
    }


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    startTime = 0
    for path, img, im0s, vid_cap in dataset:
        ##############################################################################
        Mmodel_type = "DPT_Large"
        Mimage_path = path
        MIimg = cv2.imread(Mimage_path)
        MIimg = cv2.cvtColor(MIimg, cv2.COLOR_BGR2RGB)

        # Load MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", Mmodel_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if Mmodel_type == "DPT_Large" or Mmodel_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Transform image
        input_batch = transform(MIimg).to(device)

        with torch.no_grad():
            # Get MiDaS prediction
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=MIimg.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert prediction to numpy array
        Moutput = prediction.cpu().numpy()
        # Get the shape of the output array
        height, width = Moutput.shape

        # Calculate the coordinates of the center
        center_x, center_y = width // 2, height // 2

        # Print the value at the center
        center_value = Moutput[center_y, center_x]
        print(f"Value at the center: {center_value}")
        ##############################################################################
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    ##########################################################
                    ele = names[int(cls)]
                    if ele not in coordinates:
                        coordinates[ele] = [[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])]]

                    if ele in objects_detected:
                        objects_detected[ele]+=1
                    else:
                        objects_detected[ele]=1

                    for key in coordinates:
                        inst = coordinates[key]
                        x0 = inst[0][0]
                        x1 = inst[0][1]
                        x2 = inst[1][0]
                        x3 = inst[1][1]
                        area = (x2-x0)*(x3-x1)
                        middle_point = [(x0 + x2) // 2, (x1 + x3) // 2]

                        rd = Moutput[middle_point[1], middle_point[0]]
                        weight[key] = float((((15-rd))*area*0.46)/1000)
                    ##########################################################

            # Stream results
            if dataset.mode != 'image':
                currentTime = time.time()

                fps = 1 / (currentTime - startTime)
                startTime = currentTime

                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                  
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    results_path = 'results.csv'
    with open(results_path, 'w') as results_file:
        results_file.write("No,object_name,count,Calories,Protein,Fat,Carbs,Fiber\n")

    x = 0
    for key, value in objects_detected.items():
        x = x + 1
        name = str(value) + key
        food_info = get_food_info_from_api(name)

        if food_info:
            objects_detected[key] = {
                            "count": value,
                            "food_info": food_info
                        }

            with open(results_path, 'a') as results_file:
                results_file.write(f"{x},")
                results_file.write(f"{key},")
                results_file.write(f"{value},")
                results_file.write(f"{food_info['nutrients']['calories']},")
                results_file.write(f"{food_info['nutrients']['protein']},")
                results_file.write(f"{food_info['nutrients']['fat']},")
                results_file.write(f"{food_info['nutrients']['carbs']},")
                results_file.write(f"{food_info['nutrients']['fiber']}\n")


    # Print the detected object counts
    # print("Detected Object Counts:")
    # for obj_name, obj_count in object_coordinates.items():
    #     print(f"{obj_name}: {len(obj_count)} objects detected")

    # if object_name in object_coordinates:
    #     if frame in object_coordinates[object_name]:
    #         object_coordinates[object_name][frame].append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
    #     else:
    #         object_coordinates[object_name][frame] = [(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))]
    # else:
    #     object_coordinates[object_name] = {frame: [(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()