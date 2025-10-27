; Auto-generated. Do not edit!


(cl:in-package obstacle_based_cdf-msg)


;//! \htmlinclude Array.msg.html

(cl:defclass <Array> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0))
   (rows
    :reader rows
    :initarg :rows
    :type cl:integer
    :initform 0)
   (cols
    :reader cols
    :initarg :cols
    :type cl:integer
    :initform 0))
)

(cl:defclass Array (<Array>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Array>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Array)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name obstacle_based_cdf-msg:<Array> is deprecated: use obstacle_based_cdf-msg:Array instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <Array>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader obstacle_based_cdf-msg:data-val is deprecated.  Use obstacle_based_cdf-msg:data instead.")
  (data m))

(cl:ensure-generic-function 'rows-val :lambda-list '(m))
(cl:defmethod rows-val ((m <Array>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader obstacle_based_cdf-msg:rows-val is deprecated.  Use obstacle_based_cdf-msg:rows instead.")
  (rows m))

(cl:ensure-generic-function 'cols-val :lambda-list '(m))
(cl:defmethod cols-val ((m <Array>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader obstacle_based_cdf-msg:cols-val is deprecated.  Use obstacle_based_cdf-msg:cols instead.")
  (cols m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Array>) ostream)
  "Serializes a message object of type '<Array>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'data))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'rows)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'rows)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'rows)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'rows)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'cols)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'cols)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'cols)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'cols)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Array>) istream)
  "Deserializes a message object of type '<Array>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'rows)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'rows)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'rows)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'rows)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'cols)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'cols)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'cols)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'cols)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Array>)))
  "Returns string type for a message object of type '<Array>"
  "obstacle_based_cdf/Array")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Array)))
  "Returns string type for a message object of type 'Array"
  "obstacle_based_cdf/Array")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Array>)))
  "Returns md5sum for a message object of type '<Array>"
  "62b5b442ae701eb893fd7e389d955602")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Array)))
  "Returns md5sum for a message object of type 'Array"
  "62b5b442ae701eb893fd7e389d955602")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Array>)))
  "Returns full string definition for message of type '<Array>"
  (cl:format cl:nil "float32[] data~%uint32 rows~%uint32 cols~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Array)))
  "Returns full string definition for message of type 'Array"
  (cl:format cl:nil "float32[] data~%uint32 rows~%uint32 cols~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Array>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Array>))
  "Converts a ROS message object to a list"
  (cl:list 'Array
    (cl:cons ':data (data msg))
    (cl:cons ':rows (rows msg))
    (cl:cons ':cols (cols msg))
))
