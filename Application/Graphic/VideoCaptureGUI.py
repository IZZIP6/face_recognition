import tkinter as tk
from Application.VideoCaptureController import VideoCaptureController


class VideoCaptureGUI:
    def __init__(self, version):
        self.controller = VideoCaptureController(version=version)
        self.root = tk.Tk()
        self.var_check4use = tk.IntVar(self.root)
        self.var_check4pca = tk.IntVar(self.root)
        self.var_checkpred = tk.IntVar(self.root, value=1)
        self.name_f_id = tk.StringVar
        self.panel4video = tk.Label(self.root, background="#141414")
        self.threshold1 = tk.Scale(self.root)
        self.threshold3 = tk.Scale(self.root)
        self.check4use_component = tk.Checkbutton(self.root, variable=self.var_check4use)
        self.check4pca = tk.Checkbutton(self.root, variable=self.var_check4pca)
        self.warning_pca = tk.StringVar(self.root)
        self.text_name = tk.StringVar(self.root)
        self.prediction = tk.Checkbutton(self.root, variable=self.var_checkpred)
        self.create_gui()
        print("VERSION:\t"+version)

    def create_gui(self):
        self.root.title('FR System')
        self.root.config(background="#262626")
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar)
        file_menu.add_command(label="New")
        file_menu.add_command(label="Open")
        menu_bar.add_cascade(label="File", menu=file_menu)
        gallery_menu = tk.Menu(menu_bar)
        gallery_menu.add_command(label="Show")
        gallery_menu.add_command(label="Add Item")
        menu_bar.add_cascade(label="Gallery", menu=gallery_menu)
        help_menu = tk.Menu(self.root)
        help_menu.add_command(label="Help")
        help_menu.add_command(label="About")
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menu_bar)
        self.panel4video.grid(row=0,
                              column=0,
                              columnspan=2,
                              rowspan=10,
                              ipadx=1, ipady=1,
                              padx=4,
                              pady=9)
        panel4option = tk.Label(self.root,
                                background="#262626",
                                width=50)
        panel4option.grid(row=0,
                          column=2,
                          sticky="n",
                          pady=9, padx=5,
                          rowspan=2)
        button4play = tk.Button(self.root,
                                text="Play",
                                command=self.controller.next_video)
        button4play.grid(row=11,
                         column=0,
                         sticky="e",
                         padx=2,
                         pady=5)
        self.threshold1.config(from_=0.3,
                               to=1,
                               resolution=0.01,
                               orient="horizontal",
                               length=300,
                               sliderlength=20,
                               activebackground='#474747',
                               background='#262626',
                               foreground="#ffffff",
                               cursor='hand1',
                               highlightbackground='#262626',
                               troughcolor="#ffffff",
                               font=10,
                               label='1st threshold',
                               command=self.controller.set_threshold4temp_template)
        self.threshold1.grid(row=0,
                             column=2,
                             sticky="n",
                             pady=6,
                             columnspan=3)
        self.threshold3.config(from_=10,
                               to=1500,
                               resolution=10,
                               orient="horizontal",
                               length=300,
                               sliderlength=20,
                               activebackground='#474747',
                               background='#262626',
                               foreground="#ffffff",
                               cursor='hand1',
                               highlightbackground='#262626',
                               troughcolor="#ffffff",
                               font=10,
                               label='n-dim 4 PCA',
                               command=self.controller.change_pca_dimensions)
        self.threshold3.grid(row=2,
                             column=2,
                             sticky="n",
                             pady=6,
                             columnspan=3)
        self.check4pca.config(text="PCA",
                              background='#262626',
                              foreground='#ffffff',
                              selectcolor='#262626',
                              command=lambda: self.controller.use_pca(self.var_check4pca.get())
                              )
        self.check4pca.grid(row=4,
                            column=2,
                            sticky="nw",
                            padx=18)
        self.check4use_component.config(text="Using Loaded PCA-component",
                                        background='#262626',
                                        foreground='#ffffff',
                                        selectcolor='#262626',
                                        command=lambda: self.controller.ask4load_components(self.var_check4use.get()))
        self.check4use_component.grid(row=5,
                                      column=2,
                                      sticky="nw",
                                      padx=18)
        self.prediction.config(text='Prediction',
                               background='#262626',
                               foreground='#ffffff',
                               selectcolor='#262626',
                               command=lambda: self.controller.use_prediction(self.var_checkpred.get()))
        self.prediction.grid(row=3,
                             column=2,
                             sticky='nw',
                             padx=18)
        self.start_video()

    def start_video(self):
        current_frame = self.controller.Logic.run_video()
        self.panel4video.imgtk = current_frame
        self.panel4video.config(image=current_frame)
        self.root.after(1, self.start_video)
