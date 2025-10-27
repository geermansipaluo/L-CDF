
(cl:in-package :asdf)

(defsystem "obstacle_based_cdf-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Array" :depends-on ("_package_Array"))
    (:file "_package_Array" :depends-on ("_package"))
  ))