;; Auto-generated. Do not edit!


(when (boundp 'sensor_cdf::Array)
  (if (not (find-package "SENSOR_CDF"))
    (make-package "SENSOR_CDF"))
  (shadow 'Array (find-package "SENSOR_CDF")))
(unless (find-package "SENSOR_CDF::ARRAY")
  (make-package "SENSOR_CDF::ARRAY"))

(in-package "ROS")
;;//! \htmlinclude Array.msg.html


(defclass sensor_cdf::Array
  :super ros::object
  :slots (_data _rows _cols ))

(defmethod sensor_cdf::Array
  (:init
   (&key
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:rows __rows) 0)
    ((:cols __cols) 0)
    )
   (send-super :init)
   (setq _data __data)
   (setq _rows (round __rows))
   (setq _cols (round __cols))
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:rows
   (&optional __rows)
   (if __rows (setq _rows __rows)) _rows)
  (:cols
   (&optional __cols)
   (if __cols (setq _cols __cols)) _cols)
  (:serialization-length
   ()
   (+
    ;; float32[] _data
    (* 4    (length _data)) 4
    ;; uint32 _rows
    4
    ;; uint32 _cols
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; uint32 _rows
       (write-long _rows s)
     ;; uint32 _cols
       (write-long _cols s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; uint32 _rows
     (setq _rows (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _cols
     (setq _cols (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get sensor_cdf::Array :md5sum-) "62b5b442ae701eb893fd7e389d955602")
(setf (get sensor_cdf::Array :datatype-) "sensor_cdf/Array")
(setf (get sensor_cdf::Array :definition-)
      "float32[] data
uint32 rows
uint32 cols

")



(provide :sensor_cdf/Array "62b5b442ae701eb893fd7e389d955602")


