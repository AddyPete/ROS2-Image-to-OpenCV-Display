import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from time import time
import numpy as np
import serial
from stable_baselines3 import PPO

# HEIGHT = 1080 #HD CAMERA
# WIDTH = 720 

HEIGHT = 896 #ZED CAMERA
WIDTH = 512


class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')

        self.start_time = time()
        self.frame_count = 0
        self.ball_x_pos = 0
        self.global_msg = Image()
        self.i = 0
        
        self.subscription = self.create_subscription(
            Image,
            '/zed2i/zed_node/rgb/image_rect_color',
            self.callback,
            10)
        self.bridge = CvBridge()

        #/zed2i/zed_node/rgb/image_rect_color
        #/camera/image_raw/uncompressed

        self.model = YOLO('yolov8l.pt')

        self.names = self.model.names
        # load RL model
        self.nav_model = PPO.load('/home/addy/Rover_SB3_Test/ballFollowOnly_NoAngleInfo_AutoStraight_PPO')
        # ESP32 Serial Communication
        try:
            self.arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1)
        except:
            print ("ESP32 NOT CONNECTED")
        self.timer = self.create_timer(0.01, self.timer_callback)  # 0.01 seconds = 10 milliseconds
        self.last_received_time = self.get_clock().now()
        

    def callback(self, msg):
        # Process the received message here
        self.last_received_time = self.get_clock().now()
        self.global_msg = msg

    def timer_callback(self):
        # Check if it's been 10 milliseconds since the last message
        current_time = self.get_clock().now()
        time_diff = current_time - self.last_received_time
        if time_diff.nanoseconds >= 10_000_000:
            # Do something here if no message received within 10 milliseconds
            #print("COUNT {0}".format(self.i))
            self.i+=1
            self.image_callback()

    def image_callback(self):
        msg = self.global_msg
        #self.start_time = time()
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            final_frame = self.follow_ball(cv_image)
        except Exception as e:
            self.get_logger().info(f'Error converting image: {str(e)}')
            return

        cv2.imshow('opencv image', final_frame)
        # cv2.imshow('opencv image', cv_image)
        cv2.waitKey(1)

    def follow_ball(self, cap):
        
        
        # Read a frame from the video
        frame = cap
        # frame = cv2.resize(frame,(320,240))

        #if success:
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        go = 0.0
        left_right = 0
        obj_detected = False

        #print ("RESULTS {0}".format(results.boxes[0]))
        for det_object in results[0].boxes.cpu().numpy():
            # print ("HOOOOOOOOOOOOOOOOOOOOOOOO")
            if self.names[int(det_object.cls)] == 'sports ball':
                obj_detected = True
                xyxy = det_object.xyxy[0]
                x1, y1, x2, y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])

                # Visualize Only the specific Target Class
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(frame, 'sports ball', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                # area_fraction = ((x2-x1)*(y2-y1))/(640*480)
                # x_normalized = (((x1+x2)/2)/640)*2-1
                area_fraction = ((x2-x1)*(y2-y1))/(HEIGHT*WIDTH)
                x_normalized = (((x1+x2)/2)/WIDTH)*2-1
                self.ball_x_pos = x_normalized
                # print("ball x pos: ",x_normalized)

        
                #### Compute Action and Send via ESP32
                observation = np.array([x_normalized,area_fraction],dtype=np.float64)
                action, _states = self.nav_model.predict(observation, deterministic=True)

                if action == 0:
                    left_right = -0.10
                elif action == 2:
                    left_right = 0.10
                else:
                    left_right = 0.00

                # send to Rover via ESP32
                go = 0.20
                # cmd_string = f"0.5,{left_right},0.,0.,0.,0.,0.,0&\n"
                # self.arduino.write(cmd_string.encode())
                # print("steer: ",action)
            # else:
            #     cmd_string = f"0.,0.,0.,0.,0.,0.,0.,0&\n"
        
        # if not obj_detected and self.ball_x_pos > 0:
        #     action = 2 # RIGHT
        # elif not obj_detected and self.ball_x_pos < 0:
        #     action = 0 # LEFT
            
        cmd_string = f"{go},{left_right},0.,0.,0.,0.,0.,0&\n"

        try:
            self.arduino.write(cmd_string.encode())
        except:
            print ("ESP32 not connected, can't write to arduino")
            # self.arduino.write(cmd_string.encode())
        # Visualize ALL detected on the frame
        # annotated_frame = results[0].plot()

        # end_time = time()
        # fps = 1/np.round(end_time - self.start_time, 2)

        # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        self.frame_count += 1
        current_time = time()
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

        
        if 'action' not in locals():
            action = ""
            final_action = ""
        elif action == 0:
            final_action = "Left"
        elif action == 1:
            final_action = "Center"
        else:
            final_action = "Right"

        
        cv2.putText(frame, f'Action: {action} Steer: {final_action}', (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 6, 185), 2, cv2.LINE_AA, False)
        #print (obj_detected)
        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        # cv2.imshow("YOLOv8 Inference", frame)
        return frame

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             return
    #     else:
    #         # Break the loop if the end of the video is reached
    #         return

    # # Release the video capture object and close the display window
    #     cap.release()
    #     cv2.destroyAllWindows()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()