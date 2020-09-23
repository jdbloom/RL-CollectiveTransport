(ql:quickload "cl-csv")
(require "cl-csv")

(defun evaluate-episode (filepath)
  (let ((reward 0)
	(is-header 't))
    (cl-csv:do-csv (row filepath)
      (let ((entry (first row)))
	(if is-header
	    (setf is-header 'nil)
	    (incf reward (read-from-string entry t nil :start 1 :end (search "," entry))))))
      reward))

(defun find-best-model (experiment-dirname)
  (let ((episode-counter 0)
	(model-avg -99999999)
	(best-model 'nil)
	(best-model-avg -999999999))
    (flet ((episode-number (pathname)
	     (parse-integer
	      (pathname-name pathname)
	      :start (+ 1 (position #\_ (pathname-name pathname) :from-end t)))))
      (loop
	 for reward in (mapcar #'evaluate-episode
			       (sort (directory (pathname (concatenate 'string experiment-dirname "/*.csv")))
				     #'<
				     :key #'episode-number))
	 do (progn
	      (when (= (mod episode-counter 10) 0)
		;(format t "Model avg: ~d" model-avg)
		(when (> model-avg best-model-avg)
		  (setf best-model-avg model-avg)
		  (setf best-model (- episode-counter 10)))
		(setq model-avg 0))
	      (incf model-avg (/ reward 10))
	      (incf episode-counter 1)))
	 best-model)))

(defun find-all-best-models (train-dirname)
  (loop
     for experiment-folder in (directory (pathname (concatenate 'string train-dirname "/*")))
     do (format t "~A~%" (namestring experiment-folder))
       (format t "~D~%~%" (find-best-model
			   (concatenate 'string (namestring experiment-folder) "Data")))))
