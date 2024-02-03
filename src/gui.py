import io
import numpy as np

from tkinter import *
from PIL import Image
from neural_network import MNISTNeuralNetwork


class MNISTGui(object):
    DEFAULT_PEN_SIZE = 25.0
    DEFAULT_COLOR = 'black'

    def __init__(self, model: MNISTNeuralNetwork):
        self.old_y = None
        self.old_x = None
        self.model = model
        self.color = self.DEFAULT_COLOR
        self.line_width = self.DEFAULT_PEN_SIZE

        self.root = Tk()
        self.root.title(f"Mnist predictor ({self.model.evaluate() * 100:.2f} accuracy)")
        self.root.resizable(width=False, height=False)

        self.prediction_label = Text(self.root, fg='white', height=2, width=43,
                                     borderwidth=0, highlightthickness=0, relief='ridge', state='disabled')
        self.prediction_label.tag_configure("center", justify='center', font=("Helvetica", 22))
        self.prediction_label.grid(row=0, column=0, columnspan=2)

        self.canvas = Canvas(self.root, bg='white', width=300, height=300)
        self.canvas.grid(row=1, column=0, columnspan=2)

        self.predict_button = Button(self.root, text='Predict', command=self.Predict, pady=5)
        self.predict_button.grid(row=2, column=0)

        self.clear_button = Button(self.root, text='Clear', command=self.use_eraser, pady=5)
        self.clear_button.grid(row=2, column=1)

        self.setup()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def start(self):
        self.root.mainloop()

    def Predict(self):
        img = self.get_image_from_canvas()
        img_gray = img.convert("L")
        img_resized = img_gray.resize((28, 28), Image.LANCZOS)
        img_array = (255.0 - np.array(img_resized, dtype=np.float32)) / 255.0

        prediction = self.model.predict(img_array)

        self.prediction_label.config(state='normal')
        self.prediction_label.delete(1.0, "end")
        self.prediction_label.insert("end", f"Predicted value: {prediction}", "center")
        self.prediction_label.config(state='disabled')

    def get_image_from_canvas(self):
        ps_data = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps_data.encode('utf-8')))
        return img

    def use_eraser(self):
        self.prediction_label.delete(1.0, END)
        self.canvas.delete("all")

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=self.color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
