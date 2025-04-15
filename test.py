import numpy as np
from PIL import Image, ImageTk
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def create_initial_population(size, genome_length):
    return [np.random.randint(0, 256, genome_length).astype(np.uint8) for _ in range(size)]

def fitness(key, target):
    return np.sum(key == target)

def selection(population, scores, k=3):
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2):
    c = random.randint(1, len(p1) - 2)
    child1 = np.concatenate([p1[:c], p2[c:]])
    child2 = np.concatenate([p2[:c], p1[c:]])
    return child1, child2

def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = np.uint8(random.randint(0, 255))
    return individual

def genetic_algorithm(target, pop_size, genome_length, n_generations):
    population = create_initial_population(pop_size, genome_length)
    best = None
    best_eval = 0
    for gen in range(n_generations):
        scores = [fitness(ind, target) for ind in population]
        for i in range(len(population)):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                print(f"New best score: {best_eval}")
                if best_eval == len(target):
                    return best
        selected = [selection(population, scores) for _ in range(len(population))]
        children = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                p1, p2 = selected[i], selected[i+1]
                child1, child2 = crossover(p1, p2)
                children.append(mutation(child1))
                children.append(mutation(child2))
        population = children
    return best

def encrypt_decrypt_image(image, key):
    image_data = np.array(image)
    flat_image = image_data.flatten()
    encrypted_decrypted = np.bitwise_xor(flat_image, key[:len(flat_image)]).reshape(image_data.shape)
    return Image.fromarray(encrypted_decrypted)

class ImageEncryptorDecryptorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Encryptor/Decryptor")
        self.root.geometry("800x600")
        
        self.input_image = None
        self.encrypted_image = None
        self.decrypted_image = None
        self.found_key = None

        self.frame1 = tk.Frame(root)
        self.frame2 = tk.Frame(root)
        self.frame3 = tk.Frame(root)

        self.create_frame1()
        self.create_frame2()
        self.create_frame3()

        self.show_frame(self.frame1)

    def show_frame(self, frame):
        frame.tkraise()

    def create_frame1(self):
        self.frame1.grid(row=0, column=0, sticky='nsew')
        self.frame1.grid_rowconfigure(0, weight=1)
        self.frame1.grid_columnconfigure(0, weight=1)
        upload_button = tk.Button(self.frame1, text="Upload Image", command=self.upload_image)
        upload_button.grid(row=0, column=0, padx=20, pady=20, sticky="n")

    def create_frame2(self):
        self.frame2.grid(row=0, column=0, sticky='nsew')
        self.frame2.grid_rowconfigure(0, weight=1)
        self.frame2.grid_columnconfigure(0, weight=1)
        encrypt_button = tk.Button(self.frame2, text="Encrypt", command=self.encrypt_image)
        encrypt_button.grid(row=0, column=0, padx=20, pady=20, sticky="n")

    def create_frame3(self):
        self.frame3.grid(row=0, column=0, sticky='nsew')
        self.frame3.grid_rowconfigure(0, weight=1)
        self.frame3.grid_rowconfigure(1, weight=1)
        self.frame3.grid_rowconfigure(2, weight=1)
        self.frame3.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.frame3, width=300, height=300)
        self.canvas.grid(row=0, column=0, pady=10)
        self.key_label = tk.Label(self.frame3, text="Key: ")
        self.key_label.grid(row=1, column=0, pady=10)
        
        self.ssim_label = tk.Label(self.frame3, text="SSIM: ")
        self.ssim_label.grid(row=2, column=0, pady=10)
        
        self.psnr_label = tk.Label(self.frame3, text="PSNR: ")
        self.psnr_label.grid(row=3, column=0, pady=10)
        
        decrypt_button = tk.Button(self.frame3, text="Decrypt Image", command=self.decrypt_image)
        decrypt_button.grid(row=4, column=0, pady=10)
        back_button = tk.Button(self.frame3, text="Back", command=lambda: self.show_frame(self.frame2))
        back_button.grid(row=5, column=0, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_image = Image.open(file_path).convert('L')
            self.show_frame(self.frame2)

    def encrypt_image(self):
        if self.input_image:
            input_image_data = np.array(self.input_image)
            pop_size = 100
            genome_length = input_image_data.size
            n_generations = 50
            target_key = np.random.randint(0, 256, genome_length).astype(np.uint8)
            self.found_key = genetic_algorithm(target_key, pop_size, genome_length, n_generations)
            self.encrypted_image = encrypt_decrypt_image(self.input_image, self.found_key)
            self.display_image(self.encrypted_image)
            self.key_label.config(text=f"Key: {self.found_key}")
            self.show_frame(self.frame3)
        else:
            messagebox.showerror("Error", "Please upload an image first.")

    def decrypt_image(self):
        if self.encrypted_image and self.found_key is not None:
            self.decrypted_image = encrypt_decrypt_image(self.encrypted_image, self.found_key)
            self.display_image(self.decrypted_image)
            self.plot_histograms_and_correlations()
            self.calculate_and_display_scores()
        else:
            messagebox.showerror("Error", "Please encrypt an image first.")

    def display_image(self, img):
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def calculate_correlation(self, data, direction):
        if direction == 'horizontal':
            pairs = np.vstack((data[:, :-1].flatten(), data[:, 1:].flatten())).T
        elif direction == 'vertical':
            pairs = np.vstack((data[:-1, :].flatten(), data[1:, :].flatten())).T
        elif direction == 'diagonal':
            pairs = np.vstack((data[:-1, :-1].flatten(), data[1:, 1:].flatten())).T
        else:
            raise ValueError("Invalid direction for correlation calculation")
        return np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]

    def plot_histograms_and_correlations(self):
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Histograms, Correlations, and Coefficients")
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        original_data = np.array(self.input_image).flatten()
        encrypted_data = np.array(self.encrypted_image).flatten()
        decrypted_data = np.array(self.decrypted_image).flatten()
        
        axs[0, 0].hist(original_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
        axs[0, 0].set_title('Original Image Histogram')
        axs[0, 0].set_xlabel('Pixel Value')
        axs[0, 0].set_ylabel('Frequency')
        
        axs[0, 1].hist(encrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
        axs[0, 1].set_title('Encrypted Image Histogram')
        axs[0, 1].set_xlabel('Pixel Value')
        axs[0, 1].set_ylabel('Frequency')
        
        axs[0, 2].hist(decrypted_data, bins=256, range=(0, 256), color='blue', alpha=0.7)
        axs[0, 2].set_title('Decrypted Image Histogram')
        axs[0, 2].set_xlabel('Pixel Value')
        axs[0, 2].set_ylabel('Frequency')
        
        original_pairs = np.vstack((original_data[:-1], original_data[1:])).T
        encrypted_pairs = np.vstack((encrypted_data[:-1], encrypted_data[1:])).T
        decrypted_pairs = np.vstack((decrypted_data[:-1], decrypted_data[1:])).T
        
        axs[1, 0].scatter(original_pairs[:, 0], original_pairs[:, 1], color='blue', alpha=0.5, s=1)
        axs[1, 0].set_title('Original Image Correlation')
        axs[1, 0].set_xlabel('Pixel Value (i)')
        axs[1, 0].set_ylabel('Pixel Value (i+1)')
        
        axs[1, 1].scatter(encrypted_pairs[:, 0], encrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
        axs[1, 1].set_title('Encrypted Image Correlation')
        axs[1, 1].set_xlabel('Pixel Value (i)')
        axs[1, 1].set_ylabel('Pixel Value (i+1)')
        
        axs[1, 2].scatter(decrypted_pairs[:, 0], decrypted_pairs[:, 1], color='blue', alpha=0.5, s=1)
        axs[1, 2].set_title('Decrypted Image Correlation')
        axs[1, 2].set_xlabel('Pixel Value (i)')
        axs[1, 2].set_ylabel('Pixel Value (i+1)')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        original_data_2d = np.array(self.input_image)
        encrypted_data_2d = np.array(self.encrypted_image)
        decrypted_data_2d = np.array(self.decrypted_image)

        original_corrs = [
            self.calculate_correlation(original_data_2d, 'horizontal'),
            self.calculate_correlation(original_data_2d, 'vertical'),
            self.calculate_correlation(original_data_2d, 'diagonal')
        ]

        encrypted_corrs = [
            self.calculate_correlation(encrypted_data_2d, 'horizontal'),
            self.calculate_correlation(encrypted_data_2d, 'vertical'),
            self.calculate_correlation(encrypted_data_2d, 'diagonal')
        ]

        decrypted_corrs = [
            self.calculate_correlation(decrypted_data_2d, 'horizontal'),
            self.calculate_correlation(decrypted_data_2d, 'vertical'),
            self.calculate_correlation(decrypted_data_2d, 'diagonal')
        ]
        
        corr_table = tk.Toplevel(self.root)
        corr_table.title("Correlation Coefficients Table")
        
        table_text = tk.Text(corr_table, height=10, width=50)
        table_text.insert(tk.END, "Direction\tOriginal\tEncrypted\tDecrypted\n")
        directions = ["Horizontal", "Vertical", "Diagonal"]
        
        for i, direction in enumerate(directions):
            table_text.insert(tk.END, f"{direction}\t{original_corrs[i]:.4f}\t{encrypted_corrs[i]:.4f}\t{decrypted_corrs[i]:.4f}\n")
        
        table_text.pack()

    def calculate_and_display_scores(self):
        ssim_score = ssim(np.array(self.input_image), np.array(self.decrypted_image))
        psnr_score = psnr(np.array(self.input_image), np.array(self.decrypted_image))
        self.ssim_label.config(text=f"SSIM: {ssim_score:.4f}")
        self.psnr_label.config(text=f"PSNR: {psnr_score:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEncryptorDecryptorApp(root)
    root.mainloop()
