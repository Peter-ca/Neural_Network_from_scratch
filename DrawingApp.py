import tkinter as tk
from PIL import Image, ImageDraw
import os
import NeuralNetwork

class DrawingApp:
    def __init__(self, root, width=280, height=280, line_width=6):
        self.root = root
        self.root.title("Simple Drawing App")

        self.width = width
        self.height = height
        self.line_width = line_width

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        self.image = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonPress-1>", self.set_start_pos)

        self.save_button = tk.Button(root, text="Save as PNG", command=self.save_image)
        self.save_button.pack(pady=10)

        self.last_x, self.last_y = None, None

        self.nn = NeuralNetwork.NeuralNetwork()

    def set_start_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=self.line_width)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=self.line_width)
        self.last_x, self.last_y = x, y

    def save_image(self):
        save_path = os.path.join(os.path.dirname(__file__), "number.png")
        if os.path.exists(save_path):
            os.remove(save_path)
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        resized_image.save(save_path)

        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.image)

        print(self.nn.test_self_drawn_img())


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
