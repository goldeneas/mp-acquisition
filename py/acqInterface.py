#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import pickle
import uuid
import mediapipe as mp
import time
import datetime

class Interface:

    def __init__(self,window_title = "Example", w_width = 600, w_height = 400):
        self.window_title=window_title
        self.width = w_width
        self.height = w_height
        #main gui
        self.camera_label = None
        
        #setup
        self.output_type = 'pkl'
        self.output_path = os.path.abspath(os.getcwd())
        self.sequence_type = 0
        self.max_acq = 0
        
        self.valid_model = False
        self.start_acq = False
        #outputs
        self.output_vectors = []
        self.output_classes = []
        #thread
        self.thread = None
        self.stopEvent = None
        
    def __initGUI__(self):
        self.gui = tk.Tk(className=self.window_title) 
        # set window size
        self.gui.geometry(str(self.width)+"x"+str(self.height))
        self.gui.bind('<Escape>', lambda e: self.gui.quit())
        
        # Main layout: Left frame for camera, Right frame for controls
        self.left_frame = tk.Frame(self.gui, width=360, height=360, bg="#f0f0f0")
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.right_frame = tk.Frame(self.gui)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.loading_text = tk.Label(self.left_frame, text = "Loading Camera...")
        self.loading_text.pack(expand=True)
        
        # Label name section
        tk.Label(self.right_frame, text = "Label name:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.class_text = tk.Text(self.right_frame, width=23, height=1)
        self.class_text.pack(pady=(0, 10))
        
        # Action buttons section
        buttons_frame = tk.Frame(self.right_frame)
        buttons_frame.pack(fill="x", pady=(0, 10))
        
        self.start_button = tk.Button(buttons_frame, text ="Start", width = 10, bg="#e1f5fe", command = self.update_main_window)
        self.start_button.pack(side="left", expand=True, padx=(0, 5))

        self.stop_button = tk.Button(buttons_frame, text ="Stop", width = 10, state = 'disabled', bg="#ffebee", command = self.update_main_window)
        self.stop_button.pack(side="right", expand=True, padx=(5, 0))

        # Real-time counter
        self.samples_label = tk.Label(self.right_frame, text="Samples: 0", font=("Arial", 12, "bold"), fg="#2196f3")
        self.samples_label.pack(pady=5)

        # Output log section
        tk.Label(self.right_frame, text = "Log Output:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5, 2))
        self.output_log = tk.Text(self.right_frame, width=23, height=8, state='disabled', bg="#fafafa")
        self.output_log.pack(fill="both", expand=True, pady=(0, 10))
        
        # Setup button at the bottom
        self.setup_button = tk.Button(self.right_frame, text ="Settings", width = 23, command = self.setup_window)
        self.setup_button.pack(side="bottom", pady=5)
        
        # camera thread
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.gui_camera, args=())
        self.thread.start()
        
        self.gui.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gui.mainloop()
    
    # defining the callback function (observer)
    def max_acq_callback(self,button_max_acq,ins_max_acq):
        if button_max_acq['text'] == 'Enable':
            button_max_acq['text'] = 'Disable'
            ins_max_acq['state'] = 'normal'
            ins_max_acq.delete('1.0',str(float(len(ins_max_acq.get("1.0", 'end-1c')))))
        else:
            button_max_acq['text'] = 'Enable'
            ins_max_acq.delete('1.0',str(float(len(ins_max_acq.get("1.0", 'end-1c')))))
            ins_max_acq['state'] = 'disabled'
    
    def setup_window(self):
        setup_gui = tk.Toplevel(self.gui)
        setup_gui.title("Setup")
        setup_gui.geometry("420x450")
        setup_gui.padx = 20
        setup_gui.pady = 20
        
        main_container = tk.Frame(setup_gui, padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # Output path row
        tk.Label(main_container, text="Output Path:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", pady=10)
        path = tk.Text(main_container, width=25, height=1)
        path.insert('1.0', self.output_path)
        path.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        
        # Output type row
        tk.Label(main_container, text="Output Type:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w", pady=10)
        type_frame = tk.Frame(main_container)
        type_frame.grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        type_var = tk.StringVar(value=self.output_type)
        tk.Radiobutton(type_frame, text="Pickle", value='pkl', variable=type_var).pack(side="left")
        tk.Radiobutton(type_frame, text="Json", value='json', variable=type_var).pack(side="left", padx=(10, 0))
        
        # Save sequence row
        tk.Label(main_container, text="Save Sequence:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky="nw", pady=10)
        seq_frame = tk.Frame(main_container)
        seq_frame.grid(row=2, column=1, sticky="w", padx=(10, 0))
        
        sequence_var = tk.IntVar(value=self.sequence_type)
        tk.Radiobutton(seq_frame, text="Save every Stop", value=0, variable=sequence_var).pack(anchor="w")
        tk.Radiobutton(seq_frame, text="Save on app quit", value=1, variable=sequence_var).pack(anchor="w", pady=(5, 0))
        
        # Max acquisitions row
        tk.Label(main_container, text="Max Acq:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky="w", pady=10)
        max_acq_frame = tk.Frame(main_container)
        max_acq_frame.grid(row=3, column=1, sticky="w", padx=(10, 0))
        
        ins_max_acq = tk.Text(max_acq_frame, width=10, height=1, state='disabled')
        ins_max_acq.insert('1.0', self.max_acq)
        ins_max_acq.pack(side="left")
        
        button_max_acq = tk.Button(max_acq_frame, text="Enable", width=6, 
                                  command=lambda: self.max_acq_callback(button_max_acq, ins_max_acq))
        button_max_acq.pack(side="left", padx=(5, 0))

        # Save button at the bottom
        save_button = tk.Button(main_container, text="Save Settings", width=20, bg="#c8e6c9",
                                command=lambda: self.save_setup(path.get("1.0",'end-1c'), type_var.get(), sequence_var.get(), ins_max_acq.get("1.0",'end-1c')))
        save_button.grid(row=4, column=0, columnspan=2, pady=(30, 0))
        
        main_container.columnconfigure(1, weight=1)
      
    def save_setup(self,o_path,o_type,o_seq,o_max):
        try:
            self.output_path = o_path
            self.output_type = o_type
            self.sequence_type = o_seq
            if o_max == '' : self.max_acq = 0
            else: self.max_acq = int(o_max)
            print("Setup saved!\npath: {0},\noutput_type: {1},\nsequence_type: {2},\nmax_vectors: {3}".format(o_path,o_type,o_seq,o_max))
            messagebox.showinfo("Info", "Setup saved!\npath: {0},\noutput_type: {1},\nsequence_type: {2},\nmax_vectors: {3}".format(o_path,o_type,o_seq,o_max))
        except Exception as e:
            messagebox.showerror("Error", "Check if all parameters are correct.\n" + str(e))

    def reset_outputs(self):
        self.output_vectors = []
        self.output_classes = []
        if hasattr(self, 'samples_label'):
            self.samples_label.config(text="Samples: 0")
    
    def return_outputs(self):
        return self.output_vectors,self.output_classes
    
    def if_vectors(self):
        if len(self.output_vectors) == 0 or len(self.output_classes) == 0:
            #messagebox.showwarning("Warning", "There aren't vectors to save.")
            return False
        else:
            return True
    
    def save_outputs(self, output_type = "pkl"):
        if not self.if_vectors(): return
        str_id = datetime.datetime.now()
        if output_type == "pkl":
            try:
                vectors_out = os.path.join(str(self.output_path), "output", "vectors_"+str(str_id)+"_"+str(len(self.output_vectors))+".pkl")
                classes_out = os.path.join(str(self.output_path), "output", "classes_"+str(str_id)+"_"+str(len(self.output_vectors))+".pkl")

                print(vectors_out)
                with open(vectors_out, "wb") as v_outfile:
                    print("Saving vectors pikle file...")
                    pickle.dump(self.output_vectors, v_outfile)
                with open(classes_out, "wb") as c_outfile:
                    print("Saving classes pikle file...")
                    pickle.dump(self.output_classes, c_outfile)                    
                messagebox.showinfo("Info", "All output files correctly saved.")
                self.reset_outputs()
            except Exception as e:
                messagebox.showerror("Error", "Can't save output files.\n" + str(e))
        if output_type == "json":
            self.reset_outputs()
            messagebox.showerror("Error", "The json output file is not implemented yet.")
            
    #method used for manage main actions
    def update_main_window(self):
        if self.start_button['state'] == 'disabled':
            #onClickSTOP
            self.start_button['state'] = 'active'
            self.stop_button['state'] =  'disabled'
            self.setup_button['state'] = 'active'
            self.class_text['state'] = 'normal'
            self.start_acq = False
            self.output_log['state'] = 'normal'
            self.class_text['state'] = 'normal'
            self.output_log.insert('1.0' ,"Added: "+str(len(self.output_vectors))+" vectors.\nUnique classes:"+str(set(self.output_classes)))
            if self.sequence_type == 0:
                self.save_outputs(output_type = self.output_type)
        else:
            #onClickSTART
            self.start_button['state'] = 'disabled'
            self.stop_button['state'] =  'active'
            self.setup_button['state'] = 'disabled'
            self.class_text['state'] = 'disabled'
            self.class_text['state'] = 'disabled'
            self.class_string = self.class_text.get("1.0",'end-1c')
            self.start_acq = True
            self.output_log.delete('1.0',str(float(len(self.output_log.get("1.0", 'end-1c')))))
            self.output_log['state'] = 'disabled'
                
    def on_closing(self):
        print("Closing...")
        if self.sequence_type == 1:
            self.save_outputs(output_type = self.output_type)
        self.valid_model = False
        self.cap.release()
        self.stopEvent.set()
        self.gui.destroy()
        
    def set_mediapipe_model(self,m_type):
        try:
            if m_type == 'Hand':
                from MediapipeModels import MediapipeHandModel
                self.model_type = m_type
                mp_class = MediapipeHandModel()
                self.mp_model = mp_class.return_hand_model()
                self.mp_hands = mp_class.return_mp_hands()
                self.mp_drawing = mp_class.return_mp_drawing()
                self.mp_drawing_styles = mp_class.return_mp_drawing_styles()
            else:
                print("Select a valid Model type.")
                return
        except Exception as e:
            messagebox.showerror("Error", "Error to select model type.\n" + str(e))
            
        self.valid_model = True
    
    def get_mediapipe_keypoints(self,landmark,window_w,window_h):
        #in mediapipe i use window_w,window_h to remove normalization
        #input: mediapipe landmark, width and height of gui window
        #return: vector
        if self.model_type == "Hand":
            vector = []
            for markers in landmark:
                for mark in markers:
                    vector.append(mark.x*window_w)#x
                    vector.append(mark.y*window_h)#y
            return vector
        else:
            print("Error to model_type.")
            return []
    
    def show_hand_keypoints(self, cv2image):
        # MediaPipe Tasks expects mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2image)
        timestamp_ms = int(time.time() * 1000)
        
        # Use detect_for_video for video streams
        results = self.mp_model.detect_for_video(mp_image, timestamp_ms)
        
        if results.hand_landmarks:
            #load keypoints to vectors
            if self.start_acq == True:
                self.output_vectors.append(self.get_mediapipe_keypoints(results.hand_landmarks,cv2image.shape[1],cv2image.shape[0]))
                self.output_classes.append(self.class_string)
                self.samples_label.config(text=f"Samples: {len(self.output_vectors)}")
                print(f"Captured sample {len(self.output_vectors)} for class: {self.class_string}")
                #if there is a max value in settings
                if self.max_acq > 0 and len(self.output_vectors) >= self.max_acq:
                    self.save_outputs(output_type = self.output_type)
                    self.update_main_window()
            #show keypoints        
            for hand_landmarks in results.hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    cv2image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return cv2image
    
    def gui_camera(self):
        width, height = 360, 360
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        try:
            # keep looping over frames until instructed to stop
            while not self.stopEvent.is_set():

                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if self.valid_model == True:
                        #show frame with keypoints
                        cv2image = self.show_hand_keypoints(cv2image)

                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # if the panel is not None, initialize it
                    if self.camera_label is None:
                        if self.loading_text:
                            self.loading_text.destroy()
                        self.camera_label = tk.Label(self.left_frame)
                        self.camera_label.pack(expand=True)
                    # otherwise, simply update
                    else:
                        self.camera_label.configure(image=imgtk)
                        self.camera_label.image = imgtk
        except Exception as e:
            print("Camera Error.\n", str(e))

