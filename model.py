import onnxruntime
import cv2
import numpy as np
import time


class Model:
    def __init__(self):
        self.img = None
        self.yolo = onnxruntime.InferenceSession("yolov8s.onnx")
        self.Siamese = onnxruntime.InferenceSession("siamese.onnx")
        self.classes = ["big", "small"]
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, img: bytes):
        confidence_thres = 0.8
        iou_thres = 0.8
        model_inputs = self.yolo.get_inputs()
        input_shape = model_inputs[0].shape
        input_width = input_shape[2]
        input_height = input_shape[3]
        self.img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_ANYCOLOR)
        img_height, img_width = self.img.shape[:2]

        # 裁剪下半部分作为顺序图区域（假设为下 1/3）
        order_area = self.img[int(img_height * 9 / 10):, :]  # 你可以根据图片实际比例微调这个裁剪位置
        self.order_area = order_area.copy()

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_height, input_width))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        input = {model_inputs[0].name: image_data}
        output = self.yolo.run(None, input)
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        x_factor = img_width / input_width
        y_factor = img_height / input_height
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        new_boxes = [boxes[i] for i in indices]
        small_imgs, big_img_boxes = {}, []
        for i in new_boxes:
            cropped = self.img[i[1]: i[1] + i[3], i[0]: i[0] + i[2]]
            if cropped.shape[0] < 35 and cropped.shape[1] < 35:
                small_imgs[i[0]] = cropped
            else:
                big_img_boxes.append(i)
        return small_imgs, big_img_boxes

    def split_order_image(self, count: int):
        """
        直接从左到右切割 count 个小图，每个宽度为 30。
        :param count: 要切出的图像数量。
        """
        height, width = self.order_area.shape[:2]
        char_width = 30
        ordered_crops = []
        for i in range(count):
            x_start = i * char_width
            x_end = min(x_start + char_width, width)
            crop = self.order_area[:, x_start:x_end]
            ordered_crops.append(crop)
        return ordered_crops

    @staticmethod
    def preprocess_image(img, size=(105, 105)):
        img_resized = cv2.resize(img, size)
        img_normalized = np.array(img_resized) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_expanded = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return img_expanded

    def siamese_from_order(self, order_imgs, big_img_boxes):
        result_list = []
        for img1 in order_imgs:
            image_data_1 = self.preprocess_image(img1)
            matched = False
            for box in big_img_boxes:
                if [box[0], box[1]] in result_list:
                    continue
                cropped = self.img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                image_data_2 = self.preprocess_image(cropped)
                inputs = {'input': image_data_1, "input.53": image_data_2}
                output = self.Siamese.run(None, inputs)
                output_sigmoid = 1 / (1 + np.exp(-output[0]))
                res = output_sigmoid[0][0]
                if res >= 0.1:
                    result_list.append([box[0], box[1]])
                    matched = True
                    break
            if not matched:
                print("未匹配到一个顺序小图，可能需要调阈值或检查图像切割")
        # 可视化
        for i in result_list:
            cv2.circle(self.img, (i[0] + 30, i[1] + 30), 5, (0, 0, 255), 5)
        cv2.imwrite("result.jpg", self.img)
        return result_list

