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
        
        self.loading_text = tk.Label(self.gui, text = "Loading Camera...")
        self.loading_text.place(relx = 0.2,rely = 0.3)
        
        #start button
        self.start_button = tk.Button(self.gui, text ="Start", width = 10,command = self.update_main_window)
        self.start_button.pack()
        self.start_button.place(relx = 0.65,rely = 0.4)

        #stop button
        self.stop_button = tk.Button(self.gui, text ="Stop", width = 10, state = 'disabled',command = self.update_main_window)
        self.stop_button.pack()
        self.stop_button.place(relx = 0.80,rely = 0.4)

        #class text
        class_text_title = tk.Label(self.gui, text = "Label name:")
        class_text_title.place(relx = 0.65,rely = 0.3)
        self.class_text = tk.Text(self.gui,width=10,height=1)
        self.class_text.pack()
        self.class_text.place(relx = 0.78,rely = 0.3)
        
        #output log text
        self.output_log = tk.Text(self.gui,width=21,height=3,state='disabled')
        self.output_log.pack()
        self.output_log.place(relx = 0.65,rely = 0.5)
        
        #setup button
        self.setup_button = tk.Button(self.gui, text ="Setup", width = 23,command = self.setup_window)
        self.setup_button.pack()
        self.setup_button.place(relx = 0.65,rely = 0.65)
        
        #camera thread
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
        setup_gui.geometry("400x400")
        
        #output path
        path_title = tk.Label(setup_gui, text = "OutputPath:")
        path_title.place(relx = 0.05,rely = 0.1)
        path = tk.Text(setup_gui,width=20,height=1)
        path.insert('1.0' ,self.output_path)
        path.pack()
        path.place(relx = 0.3,rely = 0.1)
        
        #output type file
        type_title = tk.Label(setup_gui, text = "OutputType:")
        type_title.place(relx = 0.05,rely = 0.2)
        type_var = tk.StringVar()
        type_var.set(self.output_type)
        pkl = tk.Radiobutton(setup_gui, text="Pickle", value='pkl', variable=type_var)
        pkl.pack()
        pkl.place(relx = 0.3,rely = 0.2)
        json = tk.Radiobutton(setup_gui, text="Json", value='json', variable=type_var)
        json.pack()
        json.place(relx = 0.45,rely = 0.2)
        
        #output save sequence
        sequence_title = tk.Label(setup_gui, text = "SaveSequence:")
        sequence_title.place(relx = 0.05,rely = 0.3)
        sequence_var = tk.IntVar()
        sequence_var.set(self.sequence_type)
        seq_00 = tk.Radiobutton(setup_gui, text="Save output file every Stop.", value=0, variable=sequence_var)
        seq_00.pack()
        seq_00.place(relx = 0.3,rely = 0.3)
        seq_01 = tk.Radiobutton(setup_gui, text="Save output file on app quit.", value=1, variable=sequence_var)
        seq_01.pack()
        seq_01.place(relx = 0.3,rely = 0.35)
        
        #set max number of acquisitions
        title_max_acq = tk.Label(setup_gui, text = "Max acquisitions:")
        title_max_acq.place(relx = 0.05,rely = 0.45)
        ins_max_acq = tk.Text(setup_gui,width=10,height=1,state='disabled')
        ins_max_acq.insert('1.0' ,self.max_acq)
        ins_max_acq.pack()
        ins_max_acq.place(relx = 0.45, rely = 0.45)
        button_max_acq = tk.Button(setup_gui, text ="Enable", width = 5,command = lambda: self.max_acq_callback(button_max_acq,ins_max_acq))
        button_max_acq.pack()
        button_max_acq.place(relx = 0.3,rely = 0.45)

        #save button
        save_button = tk.Button(setup_gui, text ="Save", width = 10,
                                command = lambda: self.save_setup(path.get("1.0",'end-1c'),type_var.get(),sequence_var.get(),ins_max_acq.get("1.0",'end-1c')))
        save_button.pack()
        save_button.place(relx = 0.45,rely = 0.7)
      
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
        str_id = str(uuid.uuid4())
        if output_type == "pkl":
            try:
                vectors_out = os.path.join(str(self.output_path), "vectors_"+str_id+"_"+str(len(self.output_vectors))+".pkl")
                classes_out = os.path.join(str(self.output_path), "classes_"+str_id+"_"+str(len(self.output_vectors))+".pkl")

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
                        self.camera_label = tk.Label(self.gui)
                        self.camera_label.pack()
                        self.camera_label.place(relx = 0,rely = 0)
                    # otherwise, simply update
                    else:
                        self.camera_label.configure(image=imgtk)
                        self.camera_label.image = imgtk
        except Exception as e:
            print("Camera Error.\n", str(e))

