import tensorflow as tf
import numpy as np
import os
import cv2

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

tf.compat.v1.enable_eager_execution()
root_dir = os.getcwd()
# patch tf.compat.v1 into `utils.ops`
utils_ops.tf.compat.v1 = tf.compat.v1.compat.v1

# Patch the location of gfile
tf.compat.v1.gfile = tf.compat.v1.io.gfile

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.compat.v1.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
              image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.compat.v1.cast(detection_masks_reframed > 0.5,
                                      tf.compat.v1.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def visualise(frame, model, category_index):
   image_np = np.array(frame)
   output_dict = run_inference_for_single_image(model,image_np)
   vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)


def main():
  # List of the strings that is used to add correct label for each box.
  h_category_index = label_map_util.create_category_index_from_labelmap('annotations/labelmap_helmet.pbtxt', use_display_name=True)
  m_category_index = label_map_util.create_category_index_from_labelmap('annotations/labelmap_motorbike.pbtxt', use_display_name=True)

  #model = tf.keras.models.load_model(str(model_dir))
  h_model = tf.compat.v1.saved_model.load_v2(str("helmet_model/saved_model"))
  m_model = tf.compat.v1.saved_model.load_v2(str("motorbike_model/saved_model"))

  # cap = cv2.VideoCapture(0)
  cap = cv2.VideoCapture("bikingclip.mp4")
  # cap.open("http://192.168.1.4:8080/video")
  writer = create_video_writer(cap, "output.mp4")

  while(cap.isOpened()):
    try:
  # while(True):
      # Capture frame-by-frame
        ret,frame = cap.read()
        if ret:
          visualise(frame, h_model, h_category_index)
          visualise(frame, m_model, m_category_index)
          cv2.namedWindow("output", cv2.WINDOW_NORMAL)
          cv2.imshow('output',frame)
          writer.write(frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        else:
          break
    except Exception as e:
      pass

  # When everything done, release the capture
  cap.release()
  writer.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()